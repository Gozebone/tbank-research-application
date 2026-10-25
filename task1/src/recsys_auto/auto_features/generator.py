"""Генератор кандидатов-признаков.

Перебирает декартово произведение допустимых компонент DSL. Допустимость
метрики для сущности задаётся ``ENTITY_METRICS``, поэтому заведомо
бессмысленные признаки (например, ``nunique_items`` для пары user-item) не
попадают в поиск.
"""

from __future__ import annotations

from itertools import product

from ..features.dsl import DECAY_METRICS, ENTITIES, ENTITY_METRICS, FeatureSpec

DEFAULT_WINDOWS: tuple[int | None, ...] = (1, 3, 7, 14, 30, 60, 90, 180, 365, None)
DEFAULT_EVENTS: tuple[str | None, ...] = (None, "listen", "like")

# метрики, которые определены только для прослушиваний (played_ratio и длина)
PLAYED_METRICS = frozenset({"sum_played", "mean_played", "max_played", "mean_track_len"})


def is_valid_combination(entity: str, metric: str, event_type: str | None) -> bool:
    if event_type is not None and event_type != "listen" and metric in PLAYED_METRICS:
        return False
    if event_type is not None and metric.startswith("nunique") and event_type != "listen":
        return False
    return True


def generate_specs(
    entities: tuple[str, ...] = ENTITIES,
    windows: tuple[int | None, ...] = DEFAULT_WINDOWS,
    event_types: tuple[str | None, ...] = DEFAULT_EVENTS,
    exclude: set[str] | None = None,
) -> list[FeatureSpec]:
    """Сгенерировать все допустимые спецификации."""
    exclude = exclude or set()
    specs: list[FeatureSpec] = []
    for entity, event_type, window in product(entities, event_types, windows):
        for metric in ENTITY_METRICS[entity]:
            if metric in DECAY_METRICS:
                # decay_count требует параметр half-life и доступен только агенту
                continue
            if not is_valid_combination(entity, metric, event_type):
                continue
            spec = FeatureSpec(entity=entity, metric=metric, window_days=window, event_type=event_type)
            if spec.name not in exclude:
                specs.append(spec)
    return specs
