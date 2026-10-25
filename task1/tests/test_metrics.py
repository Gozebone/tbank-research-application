from __future__ import annotations

import numpy as np
import pandas as pd

from recsys_auto.eval.metrics import (
    map_at_k,
    ndcg_at_k,
    ranking_metrics,
    recall_at_k,
)


def test_ranking_metrics_on_perfect_order():
    labels = np.array([1, 1, 0, 0, 0])
    assert ndcg_at_k(labels, 5) == 1.0
    assert recall_at_k(labels, 2) == 1.0
    assert recall_at_k(labels, 5) == 1.0
    assert map_at_k(labels, 5) == 1.0


def test_ndcg_penalizes_bad_order():
    labels = np.array([0, 1, 0, 0, 0])
    assert ndcg_at_k(labels, 5) < 1.0



def test_ranking_metrics_are_zero_without_positives():
    labels = np.zeros(5, dtype=int)
    assert ndcg_at_k(labels, 5) == 0.0
    assert recall_at_k(labels, 5) == 0.0
    assert map_at_k(labels, 5) == 0.0


def test_ranking_metrics_average_per_user():
    df = pd.DataFrame(
        {
            "uid": [1, 1, 2, 2],
            "label": [1, 0, 0, 1],
            "score": [0.9, 0.1, 0.9, 0.1],
        }
    )
    metrics = ranking_metrics(df, ks=(1,))
    assert metrics["ndcg@1"] == 0.5
    assert metrics["recall@1"] == 0.5
    assert metrics["n_users"] == 2


def test_equal_scores_do_not_inherit_positive_first_order():
    # в датасете позитив всегда идёт первым; при равных скорах это не должно
    # превращаться в идеальное ранжирование
    rows = []
    for uid in range(200):
        for rank in range(5):
            rows.append((uid, 1 if rank == 0 else 0, 0.5))
    df = pd.DataFrame(rows, columns=["uid", "label", "score"])
    metrics = ranking_metrics(df, ks=(1,))
    assert metrics["ndcg@1"] < 0.5
