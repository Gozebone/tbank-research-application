from __future__ import annotations

import polars as pl

from recsys_auto.data.load import sample_users


def _frame(n_users: int = 6, events_per_user: int = 5) -> pl.LazyFrame:
    rows = []
    for uid in range(n_users):
        for offset in range(events_per_user):
            rows.append(
                {
                    "uid": uid,
                    "item_id": uid * 10 + offset,
                    "ts": offset,
                    "event_type": "listen",
                }
            )
    return pl.DataFrame(rows).lazy()


def test_sample_users_excludes_previous_sample():
    lf = _frame()
    _, first = sample_users(lf, n_users=3, min_events=3, seed=1)
    first_uids = set(first.get_column("uid").to_list())

    _, second = sample_users(
        lf, n_users=2, min_events=3, seed=2, exclude=first.get_column("uid").to_numpy()
    )
    second_uids = set(second.get_column("uid").to_list())

    assert len(first_uids) == 3
    assert len(second_uids) == 2
    assert not (first_uids & second_uids)


def test_sample_users_is_deterministic():
    lf = _frame()
    _, a = sample_users(lf, n_users=3, min_events=3, seed=7)
    _, b = sample_users(lf, n_users=3, min_events=3, seed=7)
    assert a.get_column("uid").to_list() == b.get_column("uid").to_list()
