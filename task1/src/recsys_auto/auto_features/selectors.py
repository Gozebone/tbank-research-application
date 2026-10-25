"""Стратегии отбора признаков.

Все стратегии работают на валидационной выборке; тест не используется.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def per_user_auc(
    df: pd.DataFrame,
    columns: list[str],
    target: str = "label",
    uid: str = "uid",
    fillna: float = -999.0,
) -> pd.Series:
    """Средний AUC внутри пользователя (ranking-метрика, а не глобальная).

    Для рекомендаций важна способность упорядочивать кандидатов одного
    пользователя, поэтому глобальный AUC может недооценивать признаки вроде
    повторных прослушиваний, которые редки, но очень точны.
    """
    y = df[target].astype(float)
    uid_values = df[uid]
    n_pos = y.groupby(uid_values).transform("sum")
    n_neg = y.groupby(uid_values).transform("count") - n_pos
    valid = (n_pos > 0) & (n_neg > 0)

    pos_per_user = n_pos[valid].groupby(uid_values[valid]).first()
    neg_per_user = n_neg[valid].groupby(uid_values[valid]).first()

    scores: dict[str, float] = {}
    for col in columns:
        values = df[col].astype("float64").fillna(fillna)
        ranks = values.groupby(uid_values).rank(method="average")
        pos_rank_sum = ranks[y == 1].groupby(uid_values[y == 1]).sum()
        pos_rank_sum = pos_rank_sum.reindex(pos_per_user.index)
        auc = (pos_rank_sum - pos_per_user * (pos_per_user + 1) / 2) / (
            pos_per_user * neg_per_user
        )
        scores[col] = float(auc.mean())
    return pd.Series(scores, name="user_auc")


def univariate_auc(
    df: pd.DataFrame,
    columns: list[str],
    target: str = "label",
    fillna: float = -999.0,
) -> pd.Series:
    """AUC каждого признака как отдельного скорера.

    Пропуски заменяются на ``fillna`` (по умолчанию — ниже любого значения).
    Константные признаки получают AUC = 0.5.
    """
    y = df[target].to_numpy()
    scores: dict[str, float] = {}
    for col in columns:
        values = df[col].astype("float64").to_numpy(copy=True)
        if np.isnan(values).any():
            values[np.isnan(values)] = fillna
        if np.unique(values).size <= 1:
            scores[col] = 0.5
        else:
            scores[col] = float(roc_auc_score(y, values))
    return pd.Series(scores, name="auc")


def auc_strength(auc: pd.Series) -> pd.Series:
    """Насколько признак отличает классы в любую сторону."""
    return (auc - 0.5).abs()


def correlation_prune(
    df: pd.DataFrame,
    columns: list[str],
    threshold: float = 0.98,
) -> list[str]:
    """Убрать почти дублирующие признаки (жадно, по порядку columns)."""
    if not columns:
        return []
    values = df[columns].astype("float64").fillna(0.0)
    corr = values.corr().abs()
    keep: list[str] = []
    for col in columns:
        if all(corr.loc[col, other] < threshold for other in keep):
            keep.append(col)
    return keep


def rank_by_auc(scores: pd.Series) -> list[str]:
    """Имена признаков, отсортированные по силе одномерного сигнала."""
    return list(auc_strength(scores).sort_values(ascending=False).index)
