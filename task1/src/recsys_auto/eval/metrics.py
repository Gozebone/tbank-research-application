"""Метрики классификации и ранжирования."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score


def classification_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "log_loss": float(log_loss(y_true, np.clip(scores, 1e-7, 1 - 1e-7))),
    }


def _per_user_arrays(df: pd.DataFrame, score_col: str = "score") -> list[tuple[np.ndarray, np.ndarray]]:
    # перемешиваем перед сортировкой, чтобы равные скоры не наследовали
    # порядок строк датасета (в кандидатах позитивы идут первыми)
    shuffled = df.sample(frac=1.0, random_state=42)
    ordered = shuffled.sort_values(["uid", score_col], ascending=[True, False], kind="mergesort")
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for _, grp in ordered.groupby("uid", sort=False):
        out.append((grp[score_col].to_numpy(), grp["label"].to_numpy()))
    return out


def ndcg_at_k(labels: np.ndarray, k: int) -> float:
    if labels.sum() == 0:
        return 0.0
    top = labels[:k]
    discounts = 1.0 / np.log2(np.arange(2, len(top) + 2))
    dcg = float((top * discounts).sum())
    ideal_len = min(int(labels.sum()), k)
    idcg = float((1.0 / np.log2(np.arange(2, ideal_len + 2))).sum())
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(labels: np.ndarray, k: int) -> float:
    n_pos = labels.sum()
    if n_pos == 0:
        return 0.0
    return float(labels[:k].sum() / n_pos)


def map_at_k(labels: np.ndarray, k: int) -> float:
    top = labels[:k]
    if top.sum() == 0:
        return 0.0
    positions = np.arange(1, len(top) + 1)
    precisions = np.cumsum(top) / positions
    return float((precisions * top).sum() / min(top.sum(), k))


def ranking_metrics(df: pd.DataFrame, ks: tuple[int, ...] = (5, 10, 20)) -> dict[str, float]:
    """NDCG/Recall/MAP@K, усреднённые по пользователям."""
    per_user = _per_user_arrays(df)
    out: dict[str, float] = {}
    for k in ks:
        out[f"ndcg@{k}"] = float(np.mean([ndcg_at_k(labels, k) for _, labels in per_user]))
        out[f"recall@{k}"] = float(np.mean([recall_at_k(labels, k) for _, labels in per_user]))
        out[f"map@{k}"] = float(np.mean([map_at_k(labels, k) for _, labels in per_user]))
    out["n_users"] = int(len(per_user))
    out["avg_candidates"] = float(len(df) / max(len(per_user), 1))
    out["avg_positives"] = float(df.groupby("uid")["label"].sum().mean())
    return out


def evaluate(df: pd.DataFrame, ks: tuple[int, ...] = (5, 10, 20)) -> dict[str, float]:
    y = df["label"].to_numpy()
    scores = df["score"].to_numpy()
    metrics = classification_metrics(y, scores)
    metrics.update(ranking_metrics(df, ks))
    return metrics
