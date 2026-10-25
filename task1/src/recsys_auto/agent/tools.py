"""Инструменты LLM-агента.

Ключевой принцип: агент не получает доступ к произвольному SQL или к
файловой системе. Он может только предлагать спецификации DSL и получать
обратную связь от того же скрининга, что используется в
``scripts/06_auto_search.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..auto_features.expr import Expr, parse_expressions
from ..auto_features.expr_search import ExpressionScreener
from ..auto_features.search import screen_specs
from ..config import Config, resolve
from ..features.dsl import INDEX_ONLY_METRICS, FeatureSpec
from ..utils.io import ensure_dir, get_logger, load_json, save_json
from .schema import describe_dsl, parse_candidates

log = get_logger("agent")

MAX_SCREEN_PER_CALL_HINT = 12


class AgentTools:
    """Состояние диалога: кеш скрининга, принятый пул, история итераций."""

    def __init__(
        self,
        cfg: Config,
        tag: str = "quota",
        sample_users: int | None = None,
        batch_size: int | None = None,
        min_user_auc_gain: float | None = None,
        max_specs_per_call: int | None = None,
        pool_path: Path | None = None,
        cache_path: Path | None = None,
        require_novel: bool = False,
        mode: str = "flat",
    ) -> None:
        self.cfg = cfg
        agent_cfg = cfg.get("agent", {}) or {}
        self.tag = tag
        self.sample_users = int(sample_users or agent_cfg.get("sample_users", 200))
        self.batch_size = int(batch_size or agent_cfg.get("batch_size", 8))
        self.min_gain = float(
            min_user_auc_gain
            if min_user_auc_gain is not None
            else agent_cfg.get("min_user_auc_gain", 0.015)
        )
        self.max_specs_per_call = int(
            max_specs_per_call or agent_cfg.get("max_specs_per_call", MAX_SCREEN_PER_CALL_HINT)
        )
        self.max_per_entity_metric = int(agent_cfg.get("max_per_entity_metric", 3))
        self.require_novel = bool(require_novel or agent_cfg.get("require_novel", False))
        if mode not in ("flat", "expr"):
            raise ValueError(f"unknown agent mode: {mode}")
        self.mode = mode
        self.extended = mode == "expr"
        self.max_pool_size = int(agent_cfg.get("max_pool_size", 80))
        self.screener: ExpressionScreener | None = None

        base = ensure_dir(resolve(cfg.paths.experiments) / "auto_search")
        self.base = base
        self.pool_path = pool_path or base / f"agent_pool_{tag}.json"
        self.cache_path = cache_path or base / "agent_screen_cache.json"
        self.transcript_path = base / f"agent_transcript_{tag}.json"

        self.cache: dict[str, dict] = load_json(self.cache_path) if self.cache_path.exists() else {}
        self.pool: dict[str, dict] = {}
        if self.pool_path.exists():
            payload = load_json(self.pool_path)
            items = payload.get("specs") or payload.get("expressions") or []
            self.pool = {item["name"]: item for item in items}
        self.history: list[dict] = []
        self.finished = False
        self.summary: str | None = None

    # ------------------------------------------------------------------ tools

    def _schema_tools(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_schema",
                    "description": "Полное описание DSL: сущности, метрики, события, окна и правила.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_context",
                    "description": (
                        "Текущее состояние поиска: сильнейшие признаки сетки, важности "
                        "моделей, уже принятые агентом признаки и покрытие пространства."
                    ),
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
        ]

    def _finish_tool(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "finish",
                "description": "Завершить поиск и зафиксировать итоговый пул.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "Что найдено, что не сработало и почему.",
                        }
                    },
                    "required": ["summary"],
                },
            },
        }

    def _agg_properties(self, extended: bool) -> dict:
        props = {
            "entity": {
                "type": "string",
                "enum": ["user", "item", "user_item", "user_artist", "user_album"],
            },
            "metric": {
                "type": "string",
                "enum": [
                    "count",
                    "sum_played",
                    "mean_played",
                    "max_played",
                    "nunique_items",
                    "nunique_users",
                    "organic_count",
                ]
                + (["decay_count"] if extended else []),
            },
            "event_type": {"type": "string", "enum": ["listen", "like", "any"]},
            "window_days": {
                "type": "integer",
                "minimum": 0,
                "maximum": 730,
                "description": (
                    "Окно в днях. 0 означает всю историю. "
                    "Сетка 1/3/7/14/30/60/90/180/365 не обязательна."
                ),
            },
        }
        if extended:
            props["condition"] = {
                "type": "string",
                "enum": ["none", "complete", "skip", "non_organic"],
                "description": (
                    "Условие внутри агрегата: complete = played_ratio>=0.9, "
                    "skip = played_ratio<=0.2, non_organic = is_organic=0."
                ),
            }
            props["decay_days"] = {
                "type": "integer",
                "minimum": 0,
                "maximum": 365,
                "description": "Half-life в днях; только для metric=decay_count.",
            }
        return props

    def _flat_propose_tool(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "propose_specs",
                "description": (
                    "Проверить батч предложенных признаков: валидация DSL + скрининг "
                    f"на {self.sample_users} пользователях. Возвращает user_auc, auc, "
                    "nan_rate и статус accepted/rejected по каждому кандидату."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "candidates": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": self.max_specs_per_call,
                            "items": {
                                "type": "object",
                                "properties": {
                                    **self._agg_properties(extended=False),
                                    "hypothesis": {
                                        "type": "string",
                                        "description": "Почему этот признак должен помочь ранжированию.",
                                    },
                                },
                                "required": ["entity", "metric", "event_type", "window_days"],
                            },
                        }
                    },
                    "required": ["candidates"],
                },
            },
        }

    def _extended_propose_tools(self) -> list[dict]:
        agg = self._agg_properties(extended=True)
        hypothesis = {
            "type": "string",
            "description": "Почему это должно помочь ранжированию (один механизм).",
        }
        return [
            {
                "type": "function",
                "function": {
                    "name": "propose_aggregates",
                    "description": (
                        "Проверить батч агрегатов, в том числе с condition и decay_count. "
                        f"Скрининг на {self.sample_users} пользователях."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "candidates": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": self.max_specs_per_call,
                                "items": {
                                    "type": "object",
                                    "properties": {**agg, "hypothesis": hypothesis},
                                    "required": ["entity", "metric", "event_type", "window_days"],
                                },
                            }
                        },
                        "required": ["candidates"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "propose_expressions",
                    "description": (
                        "Проверить батч бинарных выражений над двумя агрегатами "
                        f"(ratio/delta/product). Скрининг на {self.sample_users} пользователях."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "candidates": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": self.max_specs_per_call,
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "op": {
                                            "type": "string",
                                            "enum": ["ratio", "delta", "product"],
                                        },
                                        "a": {"type": "object", "properties": agg},
                                        "b": {"type": "object", "properties": agg},
                                        "hypothesis": hypothesis,
                                    },
                                    "required": ["op", "a", "b"],
                                },
                            }
                        },
                        "required": ["candidates"],
                    },
                },
            },
        ]

    def tool_schemas(self) -> list[dict]:
        tools = self._schema_tools()
        if self.extended:
            tools.extend(self._extended_propose_tools())
        else:
            tools.append(self._flat_propose_tool())
        tools.append(self._finish_tool())
        return tools

    def dispatch(self, name: str, arguments: str | dict) -> dict:
        try:
            args = arguments if isinstance(arguments, dict) else json.loads(arguments or "{}")
        except json.JSONDecodeError as exc:
            return {"error": f"invalid JSON arguments: {exc}"}
        handler = {
            "get_schema": self.get_schema,
            "get_context": self.get_context,
            "propose_specs": self.propose_specs,
            "propose_aggregates": self.propose_aggregates,
            "propose_expressions": self.propose_expressions,
            "finish": self.finish,
        }.get(name)
        if handler is None:
            return {"error": f"unknown tool: {name}"}
        try:
            return handler(**args)
        except TypeError as exc:
            return {"error": f"bad arguments for {name}: {exc}"}

    def get_schema(self) -> dict:
        payload = describe_dsl()
        if self.extended:
            from ..auto_features.expr import describe_expression_space

            payload["expressions"] = describe_expression_space()
        return payload

    def get_context(self) -> dict:
        screen = self._grid_screen()
        context: dict = {
            "n_grid_specs_screened": int(len(screen)) if screen is not None else 0,
            "n_accepted_by_agent": len(self.pool),
            "sample_users_for_screening": self.sample_users,
            "acceptance_rule": (
                f"|user_auc - 0.5| >= {self.min_gain} и nan_rate < 0.999"
            ),
            "iterations_used": len(self.history),
            "require_novel": self.require_novel,
        }
        if screen is not None and not screen.empty:
            top = screen.head(25)[["name", "user_auc", "auc", "nan_rate"]]
            context["grid_top25_by_user_auc"] = top.round(4).to_dict(orient="records")
            context["grid_coverage"] = {
                "by_entity": screen["entity"].value_counts().to_dict(),
                "by_metric": screen["metric"].value_counts().to_dict(),
                "by_window": screen["window_days"].astype(str).value_counts().to_dict(),
            }
        accepted = sorted(
            self.pool.values(), key=lambda x: abs(x.get("user_auc", 0.5) - 0.5), reverse=True
        )
        context["accepted_features"] = [
            {k: item.get(k) for k in ("name", "user_auc", "nan_rate", "hypothesis")}
            for item in accepted[:25]
        ]
        if self.history:
            context["recent_iterations"] = self.history[-3:]
        context["model_top_gain"] = self._model_importance()
        if self.extended:
            from ..auto_features.expr import describe_expression_space

            context["expressions"] = describe_expression_space()
            context["pool_by_op"] = self._pool_counts(by="op")
            context["pool_size_limit"] = self.max_pool_size
        return context

    def propose_specs(self, candidates: list[dict]) -> dict:
        if self.finished:
            return {"error": "agent already finished"}
        accepted_specs, errors = parse_candidates(candidates)

        # отдельно отбрасываем index-only метрики: они не идут в модель напрямую
        grid_names: set[str] = set()
        if self.require_novel:
            grid = self._grid_screen()
            if grid is not None:
                grid_names = set(grid["name"])

        kept: list[tuple[FeatureSpec, str | None]] = []
        for spec, hypothesis in accepted_specs:
            if spec.name in self.pool:
                errors.append(
                    {"input": spec.name, "error": "already accepted earlier; skip duplicates"}
                )
                continue
            if spec.metric in INDEX_ONLY_METRICS:
                errors.append(
                    {
                        "input": spec.name,
                        "error": (
                            "last_ts/first_ts are index-only: они не подаются в модель. "
                            "recency и tenure уже строятся из них автоматически для "
                            "отобранных признаков; предложи другую метрику."
                        ),
                    }
                )
                continue
            if self.require_novel and spec.name in grid_names:
                errors.append(
                    {
                        "input": spec.name,
                        "error": (
                            "already covered by the fixed grid; propose a combination "
                            "the grid does not cover (other window, event or entity/metric)"
                        ),
                    }
                )
                continue
            kept.append((spec, hypothesis))

        to_screen = [s for s, _ in kept if s.name not in self.cache]
        if to_screen:
            log.info("screening %d new agent specs on %d users", len(to_screen), self.sample_users)
            frame = screen_specs(
                self.cfg,
                to_screen,
                sample_users=self.sample_users,
                seed=42,
                save=False,
            )
            for row in frame.to_dict(orient="records"):
                self.cache[row["name"]] = {
                    "auc": float(row["auc"]),
                    "user_auc": float(row["user_auc"]),
                    "nan_rate": float(row["nan_rate"]),
                    "source": "agent",
                }
            self._save_cache()

        feedback: list[dict] = []
        for spec, hypothesis in kept:
            stats = self.cache.get(spec.name)
            if stats is None:
                feedback.append({"name": spec.name, "status": "error", "error": "not screened"})
                continue
            strength = abs(stats["user_auc"] - 0.5)
            ok = stats["nan_rate"] < 0.999 and strength >= self.min_gain
            entry = {
                "name": spec.name,
                "entity": spec.entity,
                "metric": spec.metric,
                "event_type": spec.event_type,
                "window_days": spec.window_days,
                "user_auc": round(stats["user_auc"], 4),
                "auc": round(stats["auc"], 4),
                "nan_rate": round(stats["nan_rate"], 4),
                "status": "accepted" if ok else "rejected",
                "reason": (
                    "ok" if ok else f"weak: |user_auc-0.5|={strength:.4f} < {self.min_gain}"
                ),
            }
            if stats["nan_rate"] >= 0.999:
                entry["reason"] = "almost always missing (nan_rate >= 0.999)"
            if ok:
                # защита от вырождения пула: квота на пару (entity, metric)
                same_pair = sum(
                    1
                    for item in self.pool.values()
                    if item["entity"] == spec.entity and item["metric"] == spec.metric
                )
                if same_pair >= self.max_per_entity_metric:
                    ok = False
                    entry["status"] = "rejected"
                    entry["reason"] = (
                        f"diversity cap: already {same_pair} accepted features for "
                        f"{spec.entity}/{spec.metric}; try another entity or metric"
                    )
            if ok:
                self.pool[spec.name] = {
                    **entry,
                    "hypothesis": hypothesis,
                    "iteration": len(self.history) + 1,
                }
            feedback.append(entry)
        self._save_pool()
        self.history.append(
            {
                "iteration": len(self.history) + 1,
                "n_proposed": len(candidates),
                "n_valid": len(kept),
                "n_accepted": sum(1 for f in feedback if f["status"] == "accepted"),
                "n_errors": len(errors),
                "pool_size": len(self.pool),
            }
        )
        return {
            "feedback": feedback,
            "errors": errors,
            "pool_size": len(self.pool),
            "pool_by_entity_metric": self._pool_counts(),
            "note": (
                f"At most {self.max_per_entity_metric} accepted features per (entity, metric) pair. "
                "Rejected признаки можно переформулировать (другая сущность/окно/метрика). "
                "Не повторяй уже принятые имена."
            ),
        }

    def propose_aggregates(self, candidates: list[dict]) -> dict:
        return self._propose_extended(candidates, kind="agg")

    def propose_expressions(self, candidates: list[dict]) -> dict:
        return self._propose_extended(candidates, kind="expr")

    def _propose_extended(self, candidates: list[dict], kind: str) -> dict:
        """Приём батча в расширенном DSL: листья и бинарные выражения."""
        if self.finished:
            return {"error": "agent already finished"}
        if len(self.pool) >= self.max_pool_size:
            return {
                "error": (
                    f"pool is full ({self.max_pool_size} features); "
                    "stop proposing and call finish"
                )
            }
        parsed, errors = parse_expressions(candidates)
        grid_names: set[str] = set()
        if self.require_novel:
            grid = self._grid_screen()
            if grid is not None:
                grid_names = set(grid["name"])

        kept: list[tuple[FeatureSpec | Expr, str | None]] = []
        for expr, hypothesis in parsed:
            if expr.name in self.pool:
                errors.append({"input": expr.name, "error": "already accepted earlier"})
                continue
            leaves = expr.leaves() if isinstance(expr, Expr) else [expr]
            index_only = [leaf.name for leaf in leaves if leaf.metric in INDEX_ONLY_METRICS]
            if index_only:
                errors.append(
                    {
                        "input": expr.name,
                        "error": (
                            f"index-only metrics are not model features: {index_only}. "
                            "recency/tenure are derived automatically."
                        ),
                    }
                )
                continue
            if self.require_novel and not isinstance(expr, Expr) and expr.name in grid_names:
                errors.append(
                    {
                        "input": expr.name,
                        "error": (
                            "aggregate already covered by the fixed grid; use an "
                            "expression or a different window/condition"
                        ),
                    }
                )
                continue
            kept.append((expr, hypothesis))

        if self.screener is None:
            self.screener = ExpressionScreener(
                self.cfg, sample_users=self.sample_users, seed=42
            )
        scores = (
            self.screener.score([expr for expr, _ in kept]).set_index("name")
            if kept
            else pd.DataFrame()
        )

        feedback: list[dict] = []
        for expr, hypothesis in kept:
            if expr.name not in scores.index:
                feedback.append({"name": expr.name, "status": "error", "error": "not screened"})
                continue
            row = scores.loc[expr.name]
            strength = abs(float(row["user_auc"]) - 0.5)
            ok = float(row["nan_rate"]) < 0.999 and strength >= self.min_gain
            entry = {
                "name": expr.name,
                "op": row["op"],
                "entity": row["entity"],
                "user_auc": round(float(row["user_auc"]), 4),
                "auc": round(float(row["auc"]), 4),
                "nan_rate": round(float(row["nan_rate"]), 4),
                "status": "accepted" if ok else "rejected",
                "reason": (
                    "ok" if ok else f"weak: |user_auc-0.5|={strength:.4f} < {self.min_gain}"
                ),
            }
            if float(row["nan_rate"]) >= 0.999:
                entry["reason"] = "almost always missing (nan_rate >= 0.999)"
            if ok:
                signature = self._signature(expr)
                same = sum(1 for item in self.pool.values() if item.get("signature") == list(signature))
                if same >= self.max_per_entity_metric:
                    ok = False
                    entry["status"] = "rejected"
                    entry["reason"] = (
                        f"diversity cap: already {same} accepted features for this "
                        "aggregate pair; change entity, metric or operator"
                    )
            if ok:
                self.pool[expr.name] = {
                    **entry,
                    "signature": list(self._signature(expr)),
                    "expr": self._payload(expr),
                    "hypothesis": hypothesis,
                    "iteration": len(self.history) + 1,
                }
            feedback.append(entry)

        self._save_pool()
        self.history.append(
            {
                "iteration": len(self.history) + 1,
                "kind": kind,
                "n_proposed": len(candidates),
                "n_valid": len(kept),
                "n_accepted": sum(1 for f in feedback if f["status"] == "accepted"),
                "n_errors": len(errors),
                "pool_size": len(self.pool),
            }
        )
        return {
            "feedback": feedback,
            "errors": errors,
            "pool_size": len(self.pool),
            "pool_by_op": self._pool_counts(by="op"),
            "note": (
                f"At most {self.max_per_entity_metric} accepted features per aggregate "
                f"pair; pool limit {self.max_pool_size}. "
                "Rejected expressions can be reformulated (different op, metric or window)."
            ),
        }

    def finish(self, summary: str) -> dict:
        self.finished = True
        self.summary = summary
        self._save_pool()
        return {"status": "finished", "pool_size": len(self.pool), "path": str(self.pool_path)}

    # ------------------------------------------------------------- internals

    def _grid_screen(self) -> pd.DataFrame | None:
        path = self.base / "screen.csv"
        return pd.read_csv(path) if path.exists() else None

    def _model_importance(self) -> list[dict]:
        candidates = [
            ("auto_all", self.base / self.tag / "models" / "auto_all" / "feature_importance.csv"),
            (
                "auto_gain_top40",
                self.base / self.tag / "models" / "auto_gain_top40" / "feature_importance.csv",
            ),
            ("manual", resolve(self.cfg.paths.experiments) / "baseline_manual" / "feature_importance.csv"),
        ]
        out: list[dict] = []
        for name, path in candidates:
            if not path.exists():
                continue
            try:
                frame = pd.read_csv(path).head(10)
            except Exception:  # pragma: no cover - битый артефакт не должен ронять агента
                continue
            out.append(
                {
                    "model": name,
                    "top_features": frame["feature"].tolist(),
                }
            )
        return out

    @staticmethod
    def _signature(expr: FeatureSpec | Expr) -> tuple:
        if isinstance(expr, Expr):
            return (
                expr.op,
                expr.a.entity,
                expr.a.metric,
                expr.b.entity,
                expr.b.metric,
            )
        return ("agg", expr.entity, expr.metric)

    @staticmethod
    def _agg_payload(spec: FeatureSpec) -> dict:
        return {
            "entity": spec.entity,
            "metric": spec.metric,
            "event_type": spec.event_type,
            "window_days": spec.window_days,
            "condition": spec.condition,
            "decay_days": spec.decay_days,
        }

    @classmethod
    def _payload(cls, expr: FeatureSpec | Expr) -> dict:
        if isinstance(expr, Expr):
            return {"op": expr.op, "a": cls._agg_payload(expr.a), "b": cls._agg_payload(expr.b)}
        return {"op": "agg", **cls._agg_payload(expr)}

    def _pool_counts(self, by: str = "entity_metric") -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.pool.values():
            if by == "op":
                key = str(item.get("op", "agg"))
            else:
                key = f"{item['entity']}/{item.get('metric', '')}"
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def _save_cache(self) -> None:
        save_json(self.cache, self.cache_path)

    def _save_pool(self) -> None:
        key = "expressions" if self.extended else "specs"
        payload = {
            "tag": self.tag,
            "mode": self.mode,
            "sample_users": self.sample_users,
            "min_user_auc_gain": self.min_gain,
            "n_specs": len(self.pool),
            key: list(self.pool.values()),
            "iterations": self.history,
        }
        if self.summary:
            payload["summary"] = self.summary
        save_json(payload, self.pool_path)
