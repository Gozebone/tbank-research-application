from __future__ import annotations

import pandas as pd

from recsys_auto.features.manual import add_derived

DAY = 86_400
CUTOFF = 100 * DAY


def test_manual_recency_uses_full_aggregate_names():
    frame = pd.DataFrame(
        {
            "user__last_ts__any__all": [99 * DAY, 97 * DAY],
            "user__first_ts__any__all": [90 * DAY, 80 * DAY],
            "item__last_ts__any__all": [98 * DAY, 96 * DAY],
            "item__first_ts__any__all": [50 * DAY, 40 * DAY],
            "user_item__last_ts__any__all": [95 * DAY, 93 * DAY],
            "user_artist__last_ts__any__all": [92 * DAY, 91 * DAY],
        }
    )

    out = add_derived(frame, CUTOFF)

    assert out["user__days_since_last"].tolist() == [1.0, 3.0]
    assert out["user__tenure_days"].tolist() == [10.0, 20.0]
    assert out["item__days_since_last"].tolist() == [2.0, 4.0]
    assert out["item__age_days"].tolist() == [50.0, 60.0]
    assert out["user_item__days_since_last"].tolist() == [5.0, 7.0]
    assert out["user_artist__days_since_last"].tolist() == [8.0, 9.0]
