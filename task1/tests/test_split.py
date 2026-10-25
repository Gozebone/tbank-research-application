from __future__ import annotations

import polars as pl

from recsys_auto.data.split import make_time_split

DAY = 86_400


def _events(n_days: int = 100) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "uid": [1, 1, 2],
            "item_id": [10, 11, 10],
            "ts": [0, n_days * DAY - 1, n_days * DAY // 2],
            "event_type": ["listen", "listen", "listen"],
        }
    )


def test_boundaries_are_ordered_and_gapped():
    split = make_time_split(_events(), test_days=7, val_days=7, gap_days=7)
    assert split.train_start < split.train_end < split.val_start < split.val_end
    assert split.val_end + 7 * DAY == split.test_start
    assert split.train_end + 7 * DAY == split.val_start


def test_snapshots_cover_train_and_eval_periods():
    split = make_time_split(_events(), test_days=7, val_days=7, gap_days=7)
    snaps = split.snapshots(train_weeks=3, horizon_days=7)
    train = [s for s in snaps if s.split == "train"]
    assert len(train) == 3
    for s in train:
        assert s.label_end - s.label_start == 7 * DAY
        assert s.label_end <= split.train_end
        assert s.cutoff == s.label_start
    val = next(s for s in snaps if s.split == "val")
    assert val.cutoff == split.val_start
    assert val.label_end == split.val_end
    # снимки train не пересекаются с val
    assert max(s.label_end for s in train) <= split.train_end < val.cutoff
