from __future__ import annotations

import numpy as np
import polars as pl

from recsys_auto.data.candidates import build_candidates
from recsys_auto.data.split import Snapshot

DAY = 86_400


def _synthetic_events(seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    n_users, n_items = 50, 200
    for uid in range(n_users):
        # история до cutoff: каждый слушает 20 случайных объектов
        for _ in range(20):
            rows.append((uid, int(rng.integers(0, n_items)), int(rng.integers(0, 40 * DAY)), "listen"))
        # горизонт: 3 прослушивания
        for _ in range(3):
            rows.append((uid, int(rng.integers(0, n_items)), int(rng.integers(40 * DAY, 47 * DAY)), "listen"))
        # лайки в истории
        for _ in range(5):
            rows.append((uid, int(rng.integers(0, n_items)), int(rng.integers(0, 40 * DAY)), "like"))
    return pl.DataFrame(
        rows,
        schema={
            "uid": pl.UInt32,
            "item_id": pl.UInt32,
            "ts": pl.Int64,
            "event_type": pl.Utf8,
        },
        orient="row",
    )


def test_candidates_have_no_positive_leak_into_negatives():
    events = _synthetic_events()
    snap = Snapshot(
        name="val",
        split="val",
        cutoff=40 * DAY,
        label_start=40 * DAY,
        label_end=47 * DAY,
    )
    cand = build_candidates(
        events,
        snap,
        max_positives_per_user=10,
        negatives_per_positive=5,
        pool_size=100,
        min_popularity=1,
        seed=1,
    )
    assert cand.label.isin([0, 1]).all()
    pos = cand[cand.label == 1][["uid", "item_id"]].drop_duplicates()
    neg = cand[cand.label == 0][["uid", "item_id"]].drop_duplicates()
    merged = pos.merge(neg, on=["uid", "item_id"])
    assert merged.empty


def test_negatives_respect_budget():
    events = _synthetic_events(seed=3)
    snap = Snapshot(
        name="val",
        split="val",
        cutoff=40 * DAY,
        label_start=40 * DAY,
        label_end=47 * DAY,
    )
    cand = build_candidates(
        events,
        snap,
        max_positives_per_user=10,
        negatives_per_positive=5,
        pool_size=100,
        min_popularity=1,
        seed=2,
    )
    counts = cand.groupby(["uid", "label"]).size().unstack(fill_value=0)
    assert (counts[0] <= 5 * counts[1]).all()
    assert (counts[1] > 0).all()


def test_candidates_do_not_depend_on_event_row_order():
    events = _synthetic_events(seed=7)
    shuffled = events.sample(fraction=1.0, shuffle=True, seed=11)
    snap = Snapshot(
        name="val",
        split="val",
        cutoff=40 * DAY,
        label_start=40 * DAY,
        label_end=47 * DAY,
    )
    kwargs = dict(
        max_positives_per_user=10,
        negatives_per_positive=5,
        pool_size=100,
        min_popularity=1,
        seed=3,
    )
    a = build_candidates(events, snap, **kwargs).sort_values(["uid", "item_id", "label"])
    b = build_candidates(shuffled, snap, **kwargs).sort_values(["uid", "item_id", "label"])
    assert a.reset_index(drop=True).equals(b.reset_index(drop=True))
