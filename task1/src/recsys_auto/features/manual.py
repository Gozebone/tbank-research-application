"""Ручной (curated) набор признаков — сильный baseline.

Набор намеренно покрывает все группы из постановки: пользователь, объект,
прямая пара user-item и косвенные связи через исполнителя/альбом. Позже
автоматический генератор будет сравниваться именно с ним.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .dsl import FeatureSpec

WINDOWS = (1, 7, 30, 90, None)  # None == вся история


def _many(entity: str, metric: str, event: str | None, windows) -> list[FeatureSpec]:
    return [FeatureSpec(entity, metric, w, event) for w in windows]


def manual_specs() -> list[FeatureSpec]:
    specs: list[FeatureSpec] = []
    # --- пользователь ---
    specs += _many("user", "count", "listen", WINDOWS)
    specs += _many("user", "count", "like", (7, 30, 90, None))
    specs += _many("user", "count", "dislike", (30, None))
    specs += _many("user", "mean_played", "listen", (7, 30, 90, None))
    specs += _many("user", "sum_played", "listen", (30, 90))
    specs += _many("user", "nunique_items", "listen", (7, 30, 90, None))
    specs += _many("user", "organic_count", "listen", (30, None))
    specs += _many("user", "mean_track_len", "listen", (90, None))
    specs.append(FeatureSpec("user", "last_ts"))
    specs.append(FeatureSpec("user", "first_ts"))

    # --- объект ---
    specs += _many("item", "count", "listen", WINDOWS)
    specs += _many("item", "nunique_users", "listen", WINDOWS)
    specs += _many("item", "mean_played", "listen", (7, 30, 90, None))
    specs += _many("item", "count", "like", (7, 30, 90, None))
    specs += _many("item", "organic_count", "listen", (7, 30, None))
    specs.append(FeatureSpec("item", "last_ts"))
    specs.append(FeatureSpec("item", "first_ts"))
    specs.append(FeatureSpec("item", "mean_track_len", None, "listen"))

    # --- прямая связь user-item ---
    specs += _many("user_item", "count", "listen", (1, 7, 30, 90, None))
    specs.append(FeatureSpec("user_item", "count", None, "like"))
    specs += _many("user_item", "max_played", "listen", (None,))
    specs += _many("user_item", "mean_played", "listen", (None,))
    specs.append(FeatureSpec("user_item", "last_ts"))
    specs.append(FeatureSpec("user_item", "first_ts"))

    # --- косвенная связь через исполнителя ---
    specs += _many("user_artist", "count", "listen", (7, 30, 90, None))
    specs += _many("user_artist", "nunique_items", "listen", (30, 90, None))
    specs += _many("user_artist", "count", "like", (90, None))
    specs.append(FeatureSpec("user_artist", "last_ts"))

    # --- косвенная связь через альбом ---
    specs += _many("user_album", "count", "listen", (30, 90, None))
    return specs


def _safe_div(a: pd.Series, b: pd.Series, eps: float = 1.0) -> pd.Series:
    return a / (b + eps)


def add_derived(df: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    """Производные признаки: конверсии, доли, recency, cross-features.

    Все формулы используют только агрегаты, посчитанные до cutoff.
    """
    day = 86_400.0
    out: dict[str, pd.Series] = {}

    def col(name: str, default: float = 0.0) -> pd.Series:
        if name in df.columns:
            return df[name]
        # Агрегаты DSL названы полным именем ``entity__metric__event__window``.
        # Для recency/tenure в коде ниже используется короткое имя, поэтому
        # поддерживаем оба варианта; иначе missing-колонка молча превращалась
        # бы в константу cutoff / 86400.
        full = f"{name}__any__all"
        if full in df.columns:
            return df[full]
        return pd.Series(default, index=df.index, dtype="float64")

    # конверсии лайков
    for w in (7, 30, 90, None):
        tag = "all" if w is None else f"{w}d"
        out[f"user__like_rate__{tag}"] = _safe_div(
            col(f"user__count__like__{tag}"), col(f"user__count__listen__{tag}")
        )
        out[f"item__like_rate__{tag}"] = _safe_div(
            col(f"item__count__like__{tag}"), col(f"item__count__listen__{tag}")
        )

    # recency / tenure
    out["user__days_since_last"] = np.where(
        col("user__last_ts").isna(), 9999.0, (cutoff - col("user__last_ts")) / day
    )
    out["user__tenure_days"] = np.where(
        col("user__first_ts").isna(), 0.0, (cutoff - col("user__first_ts")) / day
    )
    out["item__days_since_last"] = np.where(
        col("item__last_ts").isna(), 9999.0, (cutoff - col("item__last_ts")) / day
    )
    out["item__age_days"] = np.where(
        col("item__first_ts").isna(), 9999.0, (cutoff - col("item__first_ts")) / day
    )
    out["user_item__days_since_last"] = np.where(
        col("user_item__last_ts").isna(), 9999.0, (cutoff - col("user_item__last_ts")) / day
    )
    out["user_artist__days_since_last"] = np.where(
        col("user_artist__last_ts").isna(), 9999.0, (cutoff - col("user_artist__last_ts")) / day
    )

    # доли: насколько объект/исполнитель важны для пользователя
    for w in (7, 30, 90, None):
        tag = "all" if w is None else f"{w}d"
        out[f"user_item__share__{tag}"] = _safe_div(
            col(f"user_item__count__listen__{tag}"), col(f"user__count__listen__{tag}")
        )
        out[f"user_artist__share__{tag}"] = _safe_div(
            col(f"user_artist__count__listen__{tag}"), col(f"user__count__listen__{tag}")
        )
    for w in (30, 90, None):
        tag = "all" if w is None else f"{w}d"
        out[f"user_album__share__{tag}"] = _safe_div(
            col(f"user_album__count__listen__{tag}"), col(f"user__count__listen__{tag}")
        )

    # предпочтения по длине трека и доле прослушивания
    out["user__organic_share__30d"] = _safe_div(
        col("user__organic_count__listen__30d"), col("user__count__listen__30d")
    )
    out["user_item__played_vs_user"] = col("user_item__mean_played__listen__all") - col(
        "user__mean_played__listen__all"
    )
    out["item__played_vs_user_mean"] = col("item__mean_played__listen__all") - col(
        "user__mean_played__listen__all"
    )
    out["user__short_track_affinity"] = col("user__mean_track_len__listen__all") - col(
        "item__mean_track_len__listen__all"
    )

    # популярность объекта, нормированная на активность пользователя
    out["item__pop_vs_user_activity__30d"] = col("item__count__listen__30d") / (
        col("user__count__listen__30d") + 100.0
    )
    out["user_artist__pop_vs_user__30d"] = col("user_artist__count__listen__30d") / (
        col("user__count__listen__30d") + 100.0
    )

    # давность прямого контакта с учётом частоты
    out["user_item__recent_share__30d"] = col("user_item__count__listen__30d") / (
        col("user_item__count__listen__all") + 1.0
    )

    derived = pd.DataFrame(out, index=df.index)
    return pd.concat([df, derived], axis=1)
