"""Декларативный DSL признаков.

Признак описывается четвёркой:

    (сущность, метрика, тип события, временное окно)

Например, ``user_artist / count / listen / 30d`` — сколько раз пользователь
слушал треки этого исполнителя за 30 дней до cutoff. Одна и та же спецификация
используется и ручным набором, и автоматическим генератором — это важно для
честного сравнения подходов.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Entity = Literal["user", "item", "user_item", "user_artist", "user_album"]

ENTITIES: tuple[Entity, ...] = ("user", "item", "user_item", "user_artist", "user_album")

# SQL-выражения агрегатов. Все они работают поверх таблиц events / events_enriched.
METRICS: dict[str, str] = {
    "count": "count(*)",
    "sum_played": "sum(played_ratio)",
    "mean_played": "avg(played_ratio)",
    "max_played": "max(played_ratio)",
    "nunique_items": "count(distinct item_id)",
    "nunique_users": "count(distinct uid)",
    "last_ts": "max(ts)",
    "first_ts": "min(ts)",
    "mean_track_len": "avg(track_length_seconds)",
    "organic_count": "count(*)",
    # взвешенный по времени счётчик: sum(exp(-ln2 * age / tau)).
    # SQL строится в aggregates._window_expr, потому что зависит от cutoff и tau.
    "decay_count": "sum(exp(-ln(2) * age_days / half_life))",
}

# условия внутри агрегата (расширение DSL): whitelist SQL-предикатов
CONDITIONS: dict[str, str] = {
    "complete": "played_ratio >= 0.9",
    "skip": "played_ratio <= 0.2",
    "non_organic": "is_organic = 0",
}

# метрики, для которых обязателен параметр decay_days
DECAY_METRICS = frozenset({"decay_count"})

# дополнительные условия, входящие в агрегат (а не в общий WHERE)
METRIC_CONDITIONS: dict[str, str | None] = {
    "organic_count": "is_organic = 1",
}

# какие метрики имеют смысл для сущности (защита от бессмысленных признаков)
ENTITY_METRICS: dict[str, tuple[str, ...]] = {
    "user": (
        "count",
        "sum_played",
        "mean_played",
        "nunique_items",
        "last_ts",
        "first_ts",
        "mean_track_len",
        "organic_count",
        "decay_count",
    ),
    "item": (
        "count",
        "sum_played",
        "mean_played",
        "max_played",
        "nunique_users",
        "last_ts",
        "first_ts",
        "mean_track_len",
        "organic_count",
        "decay_count",
    ),
    "user_item": (
        "count",
        "sum_played",
        "mean_played",
        "max_played",
        "last_ts",
        "first_ts",
        "decay_count",
    ),
    "user_artist": (
        "count",
        "sum_played",
        "mean_played",
        "nunique_items",
        "last_ts",
        "first_ts",
        "organic_count",
        "decay_count",
    ),
    "user_album": (
        "count",
        "sum_played",
        "mean_played",
        "nunique_items",
        "last_ts",
        "first_ts",
        "decay_count",
    ),
}

ENTITY_KEYS: dict[str, tuple[str, ...]] = {
    "user": ("uid",),
    "item": ("item_id",),
    "user_item": ("uid", "item_id"),
    "user_artist": ("uid", "primary_artist_id"),
    "user_album": ("uid", "primary_album_id"),
}

ENTITY_TABLE: dict[str, str] = {
    "user": "events",
    "item": "events",
    "user_item": "events",
    "user_artist": "events_enriched",
    "user_album": "events_enriched",
}

KEY_TABLE_NAMES: dict[str, str] = {
    "user": "keys_user",
    "item": "keys_item",
    "user_item": "keys_user_item",
    "user_artist": "keys_user_artist",
    "user_album": "keys_user_album",
}

# агрегаты, у которых отсутствие данных означает 0, а не NaN
ZERO_FILL_METRICS = frozenset(
    {"count", "nunique_items", "nunique_users", "organic_count", "decay_count"}
)

# «сырые» временные метки: используются только для производных recency-признаков
# и не подаются в модель напрямую (абсолютное значение сдвигается между снимками).
INDEX_ONLY_METRICS = frozenset({"last_ts", "first_ts"})


@dataclass(frozen=True)
class FeatureSpec:
    entity: Entity
    metric: str
    window_days: int | None = None
    event_type: str | None = None
    # расширение DSL: условие внутри агрегата и временное затухание
    condition: str | None = None
    decay_days: float | None = None

    def __post_init__(self) -> None:
        if self.entity not in ENTITIES:
            raise ValueError(f"unknown entity: {self.entity}")
        if self.metric not in METRICS:
            raise ValueError(f"unknown metric: {self.metric}")
        if self.metric not in ENTITY_METRICS[self.entity]:
            raise ValueError(f"metric {self.metric} is not supported for {self.entity}")
        if self.event_type not in (None, "listen", "like", "dislike", "unlike", "undislike"):
            raise ValueError(f"unknown event_type: {self.event_type}")
        if self.window_days is not None and self.window_days <= 0:
            raise ValueError("window_days must be positive or None")
        if self.condition is not None and self.condition not in CONDITIONS:
            raise ValueError(
                f"unknown condition: {self.condition}; allowed: {sorted(CONDITIONS)}"
            )
        if self.metric in DECAY_METRICS and self.decay_days is None:
            raise ValueError(f"metric {self.metric} requires decay_days")
        if self.metric not in DECAY_METRICS and self.decay_days is not None:
            raise ValueError("decay_days is only supported for decay_count")
        if self.decay_days is not None and self.decay_days <= 0:
            raise ValueError("decay_days must be positive")

    @property
    def name(self) -> str:
        window = "all" if self.window_days is None else f"{self.window_days}d"
        event = self.event_type or "any"
        suffix = ""
        if self.condition:
            suffix += f"__{self.condition}"
        if self.decay_days is not None:
            suffix += f"__decay{self.decay_days:g}"
        return f"{self.entity}__{self.metric}__{event}__{window}{suffix}"

    @property
    def fill_zero(self) -> bool:
        return self.metric in ZERO_FILL_METRICS

    def with_window(self, window_days: int | None) -> "FeatureSpec":
        return FeatureSpec(
            self.entity, self.metric, window_days, self.event_type, self.condition, self.decay_days
        )

    def describe(self) -> dict:
        return {
            "name": self.name,
            "entity": self.entity,
            "metric": self.metric,
            "event_type": self.event_type,
            "window_days": self.window_days,
            "condition": self.condition,
            "decay_days": self.decay_days,
        }
