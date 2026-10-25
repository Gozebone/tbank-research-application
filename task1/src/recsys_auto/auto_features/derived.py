"""Автоматические производные признаки (отношения и recency).

Базовые агрегаты из DSL — это только половина ручного набора. Вторая половина
— отношения: like rate, доля объекта в активности пользователя, тренд окна к
истории, давность контакта. Здесь эти операции шаблонно генерируются из уже
посчитанных базовых признаков, чтобы автоматический поиск мог конкурировать с
человеком не только по «сырым» агрегатам.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REL_PREFIX = "auto_rel__"


def _parse(name: str) -> tuple[str, str, str, str] | None:
    parts = name.split("__")
    if len(parts) != 4:
        return None
    return parts[0], parts[1], parts[2], parts[3]


def derive_relations(df: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    """Сгенерировать производные признаки из базовых колонок.

    Возвращает только новые колонки; исходный frame не меняется.
    """
    available = set(df.columns)
    out: dict[str, pd.Series] = {}
    day = 86_400.0

    def safe_div(num: pd.Series, den: pd.Series, eps: float = 1.0) -> pd.Series:
        return num / (den + eps)

    def add(name: str, values: pd.Series) -> None:
        if name not in available:
            out[name] = values

    for name in list(available):
        parsed = _parse(name)
        if parsed is None:
            continue
        entity, metric, event, window = parsed

        # давность последнего контакта (raw last_ts в модель не идёт)
        if metric == "last_ts" and event == "any":
            last = df[name]
            recency = np.where(last.isna(), 9999.0, (cutoff - last) / day)
            add(f"{REL_PREFIX}{entity}__days_since_last", pd.Series(recency, index=df.index))
            continue
        if metric == "first_ts" and event == "any":
            first = df[name]
            tenure = np.where(first.isna(), 0.0, (cutoff - first) / day)
            add(f"{REL_PREFIX}{entity}__tenure_days", pd.Series(tenure, index=df.index))
            continue

        if metric == "count" and event == "like":
            listen = f"{entity}__count__listen__{window}"
            if listen in available:
                add(
                    f"{REL_PREFIX}{entity}__like_rate__{window}",
                    safe_div(df[name], df[listen]),
                )

        if metric == "organic_count" and event == "listen":
            listen = f"{entity}__count__listen__{window}"
            if listen in available:
                add(
                    f"{REL_PREFIX}{entity}__organic_share__{window}",
                    safe_div(df[name], df[listen]),
                )

        if metric == "count" and event == "listen":
            # доля сущности в активности пользователя
            if entity in ("user_item", "user_artist", "user_album"):
                user_col = f"user__count__listen__{window}"
                if user_col in available:
                    add(
                        f"{REL_PREFIX}{entity}__share__{window}",
                        safe_div(df[name], df[user_col]),
                    )
            # тренд окна к полной истории
            all_col = f"{entity}__count__listen__all"
            if window != "all" and all_col in available:
                add(
                    f"{REL_PREFIX}{entity}__trend__{window}",
                    safe_div(df[name], df[all_col]),
                )

    if not out:
        return pd.DataFrame(index=df.index)
    return pd.DataFrame(out, index=df.index)
