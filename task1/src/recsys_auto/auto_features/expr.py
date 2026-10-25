"""Расширенный DSL: агрегаты с условиями/затуханием и бинарные выражения.

Плоская спецификация ``(entity, metric, event, window)`` остаётся листом.
Дополнительно разрешены:

- ``condition`` — предикат внутри агрегата (``complete``/``skip``/``non_organic``);
- ``decay_count`` + ``decay_days`` — счётчик с экспоненциальным затуханием;
- бинарные операции над двумя агрегатами: ``ratio``, ``delta``, ``product``.

Выражения вычисляются поверх уже материализованных листьев, поэтому SQL
остаётся ограниченным и проверяемым, а не пишется моделью.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..features.dsl import CONDITIONS, ENTITY_KEYS, FeatureSpec
from .generator import is_valid_combination

EXPR_OPS: tuple[str, ...] = ("ratio", "delta", "product")
EPS = 1e-6


@dataclass(frozen=True)
class Expr:
    """Бинарное выражение над двумя агрегатами."""

    op: str
    a: FeatureSpec
    b: FeatureSpec

    def __post_init__(self) -> None:
        if self.op not in EXPR_OPS:
            raise ValueError(f"unknown op: {self.op}; allowed: {EXPR_OPS}")

    @property
    def name(self) -> str:
        return f"expr__{self.op}__{self.a.name}__{self.b.name}"

    @property
    def entity(self) -> str:
        """Основная сущность выражения — используется для квоты по сущностям."""
        return self.a.entity

    def leaves(self) -> list[FeatureSpec]:
        return [self.a, self.b]

    def evaluate(self, frame: pd.DataFrame) -> pd.Series:
        left = frame[self.a.name].astype("float64")
        right = frame[self.b.name].astype("float64")
        if self.op == "ratio":
            return left / (right.abs() + EPS)
        if self.op == "delta":
            return left - right
        return left * right

    def describe(self) -> dict:
        return {
            "name": self.name,
            "op": self.op,
            "entity": self.entity,
            "a": self.a.describe(),
            "b": self.b.describe(),
            "metric": f"{self.op}({self.a.metric},{self.b.metric})",
            "event_type": None,
            "window_days": None,
            "condition": None,
            "decay_days": None,
        }


def _clean(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, str) and value.strip().lower() in ("", "none", "null", "any"):
        return None if value.strip().lower() != "any" else "any"
    return value


def spec_from_mapping(data: dict) -> FeatureSpec:
    """Собрать лист DSL из JSON-представления (с валидацией)."""
    condition = _clean(data.get("condition"))
    decay = data.get("decay_days")
    if decay in (0, "0", "", None):
        decay = None
    window = data.get("window_days")
    if window in (0, None, "", "0"):
        window = None
    event = _clean(data.get("event_type"))
    if event == "any":
        event = None
    return FeatureSpec(
        entity=str(data["entity"]).strip(),
        metric=str(data["metric"]).strip(),
        window_days=int(window) if window is not None else None,
        event_type=event,
        condition=condition,
        decay_days=float(decay) if decay is not None else None,
    )


def expression_from_mapping(data: dict) -> FeatureSpec | Expr:
    """Лист или бинарное выражение."""
    op = str(data.get("op") or "agg").strip().lower()
    if op in ("agg", "leaf", "aggregate"):
        return spec_from_mapping(data)
    if op not in EXPR_OPS:
        raise ValueError(f"unknown op: {op}; allowed: agg, {', '.join(EXPR_OPS)}")
    a = data.get("a") or {}
    b = data.get("b") or {}
    if not isinstance(a, dict) or not isinstance(b, dict):
        raise ValueError("a and b must be aggregate objects")
    return Expr(op=op, a=spec_from_mapping(a), b=spec_from_mapping(b))


def parse_expressions(raw: Any) -> tuple[list[tuple[FeatureSpec | Expr, str | None]], list[dict]]:
    """Разобрать батч от агента: список листьев и/или бинарных выражений."""
    accepted: list[tuple[FeatureSpec | Expr, str | None]] = []
    errors: list[dict] = []
    if isinstance(raw, dict):
        raw = raw.get("candidates") or raw.get("expressions") or []
    if not isinstance(raw, list):
        return accepted, [{"error": "candidates must be a list"}]

    seen: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            errors.append({"index": index, "input": item, "error": "candidate must be an object"})
            continue
        try:
            expr = expression_from_mapping(item)
        except (ValueError, KeyError, TypeError) as exc:
            errors.append({"index": index, "input": item, "error": str(exc)})
            continue
        leaves = expr.leaves() if isinstance(expr, Expr) else [expr]
        bad = None
        for leaf in leaves:
            if leaf.entity not in ENTITY_KEYS:
                bad = f"unsupported entity: {leaf.entity}"
                break
            if not is_valid_combination(leaf.entity, leaf.metric, leaf.event_type):
                bad = (
                    f"metric {leaf.metric} is not defined for event_type {leaf.event_type}"
                )
                break
        if bad:
            errors.append({"index": index, "input": item, "error": bad})
            continue
        if expr.name in seen:
            errors.append({"index": index, "input": item, "error": "duplicate in batch"})
            continue
        seen.add(expr.name)
        accepted.append((expr, item.get("hypothesis")))
    return accepted, errors


def describe_expression_space() -> dict:
    return {
        "ops": list(EXPR_OPS),
        "op_meaning": {
            "ratio": "a / |b| — доля, интенсивность, тренд",
            "delta": "a - b — прирост/падение между окнами",
            "product": "a * b — совместный сигнал (популярность × лояльность)",
        },
        "conditions": {
            name: sql for name, sql in CONDITIONS.items()
        },
        "decay": {
            "metric": "decay_count",
            "requires": "decay_days (half-life в днях)",
            "meaning": "sum(exp(-ln2 * age_days / half_life)) — свежесть важнее давности",
        },
        "limits": {
            "leaves_per_expression": 2,
            "max_window_days": 730,
            "note": "вложенные выражения (ratio от ratio) не поддерживаются",
        },
    }
