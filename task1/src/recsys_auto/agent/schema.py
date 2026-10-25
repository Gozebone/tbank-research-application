"""Схема предложения агента и её валидация.

Единственный контракт между LLM и проектом — ``(entity, metric, event_type,
window_days)``. Всё, что не проходит ``FeatureSpec``/``is_valid_combination``,
отбрасывается с понятной ошибкой, но без исключения: агент должен получить
фидбэк и исправиться на следующей итерации.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

from ..auto_features.generator import DEFAULT_EVENTS, DEFAULT_WINDOWS, is_valid_combination
from ..features.dsl import ENTITIES, ENTITY_METRICS, METRICS, FeatureSpec

EVENT_CHOICES: tuple[str, ...] = ("listen", "like", "any")
ANY_EVENT = "any"
MAX_WINDOW_DAYS = 730


class SpecCandidate(BaseModel):
    """Один предложенный признак.

    ``window_days = 0`` означает «вся история» (в DSL это ``None``), чтобы не
    требовать от модели null-значений в JSON.
    """

    entity: str
    metric: str
    event_type: str = ANY_EVENT
    window_days: int = Field(default=0, ge=0, le=MAX_WINDOW_DAYS)
    hypothesis: str | None = None

    @field_validator("event_type")
    @classmethod
    def _event(cls, value: str) -> str:
        value = (value or ANY_EVENT).strip().lower()
        if value not in EVENT_CHOICES:
            raise ValueError(f"event_type must be one of {EVENT_CHOICES}, got {value!r}")
        return value

    @field_validator("entity", "metric")
    @classmethod
    def _strip(cls, value: str) -> str:
        return value.strip()

    def to_spec(self) -> FeatureSpec:
        event = None if self.event_type == ANY_EVENT else self.event_type
        window = self.window_days or None
        return FeatureSpec(
            entity=self.entity,
            metric=self.metric,
            window_days=window,
            event_type=event,
        )


def parse_candidates(raw: Any) -> tuple[list[tuple[FeatureSpec, str | None]], list[dict]]:
    """Разобрать ответ агента.

    Возвращает ``(принятые, ошибки)``; в принятых рядом со спецификацией лежит
    гипотеза агента (если была).
    """
    accepted: list[tuple[FeatureSpec, str | None]] = []
    errors: list[dict] = []
    if isinstance(raw, dict):
        raw = raw.get("candidates", [])
    if not isinstance(raw, list):
        return accepted, [{"error": "candidates must be a list"}]

    seen: set[str] = set()
    for index, item in enumerate(raw):
        try:
            candidate = SpecCandidate.model_validate(item)
        except ValidationError as exc:
            errors.append(
                {
                    "index": index,
                    "input": item,
                    "error": "; ".join(
                        f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
                        for err in exc.errors()
                    ),
                }
            )
            continue
        try:
            spec = candidate.to_spec()
        except ValueError as exc:  # несовместимая пара entity/metric
            errors.append({"index": index, "input": item, "error": str(exc)})
            continue
        if not is_valid_combination(
            spec.entity, spec.metric, spec.event_type
        ):
            errors.append(
                {
                    "index": index,
                    "input": item,
                    "error": (
                        f"metric {spec.metric} is not defined for "
                        f"event_type {spec.event_type}"
                    ),
                }
            )
            continue
        if spec.name in seen:
            errors.append({"index": index, "input": item, "error": "duplicate in batch"})
            continue
        seen.add(spec.name)
        accepted.append((spec, candidate.hypothesis))
    return accepted, errors


def describe_dsl() -> dict:
    """Полное описание пространства поиска для промпта и тула ``get_schema``."""
    return {
        "entities": {
            entity: {
                "metrics": list(ENTITY_METRICS[entity]),
                "keys": _ENTITY_KEY_DESCRIPTION[entity],
            }
            for entity in ENTITIES
        },
        "metrics": {
            name: {"sql": expr, "note": _METRIC_NOTES.get(name, "")}
            for name, expr in METRICS.items()
        },
        "event_types": list(EVENT_CHOICES),
        "windows_days": {
            "grid": [w for w in DEFAULT_WINDOWS if w is not None],
            "all_time": 0,
            "note": (
                "Сетка не обязательна: можно предлагать любые окна 1..730 дней. "
                "0 означает всю историю."
            ),
        },
        "event_grid": [e or ANY_EVENT for e in DEFAULT_EVENTS],
        "rules": [
            "window_days=0 -> вся история (None в DSL)",
            "played_ratio-метрики (sum_played/mean_played/max_played) допустимы только для listen",
            "nunique_* допустимы только для listen",
            "last_ts/first_ts не подаются в модель напрямую: из них строятся recency/tenure",
            "агрегаты считаются только по событиям строго до cutoff — утечки нет по построению",
        ],
        "naming": "{entity}__{metric}__{event}__{window}, window = 'all' или '<N>d'",
    }


_ENTITY_KEY_DESCRIPTION: dict[str, str] = {
    "user": "uid",
    "item": "item_id (глобальная популярность объекта)",
    "user_item": "(uid, item_id) — прямая связь пользователь-объект",
    "user_artist": "(uid, primary_artist_id) — косвенная связь через исполнителя",
    "user_album": "(uid, primary_album_id) — косвенная связь через альбом",
}

_METRIC_NOTES: dict[str, str] = {
    "sum_played": "суммарная доля прослушивания, прокси силы интереса",
    "mean_played": "средняя доля прослушивания (скипы vs дослушивания)",
    "max_played": "лучшее прослушивание пары",
    "nunique_items": "широта интереса (сколько разных объектов)",
    "nunique_users": "широта аудитории объекта",
    "last_ts": "источник recency = (cutoff - last_ts) / 86400",
    "first_ts": "источник tenure = (cutoff - first_ts) / 86400",
    "organic_count": "число органических прослушиваний (is_organic = 1)",
}


def describe_dsl_text() -> str:
    """Компактное текстовое описание DSL для системного промпта."""
    lines: list[str] = []
    lines.append("Entities and available metrics:")
    for entity, info in describe_dsl()["entities"].items():
        lines.append(f"- {entity} [{info['keys']}]: {', '.join(info['metrics'])}")
    lines.append("")
    lines.append("Metrics (SQL aggregate):")
    for name, info in describe_dsl()["metrics"].items():
        note = f" — {info['note']}" if info["note"] else ""
        lines.append(f"- {name} = {info['sql']}{note}")
    lines.append("")
    lines.append(
        "event_type: listen | like | any. window_days: 1..730 days, or 0 for all history."
    )
    lines.append(
        "Default grid: windows "
        + ", ".join(str(w) for w in DEFAULT_WINDOWS if w is not None)
        + "; events listen/like/any. Proposals outside this grid are welcome."
    )
    return "\n".join(lines)
