from __future__ import annotations

import pandas as pd
import polars as pl

from recsys_auto.features.aggregates import compute_specs, create_connection
from recsys_auto.features.dsl import FeatureSpec

DAY = 86_400
CUTOFF = 100 * DAY


def _write(tmp_path):
    events = pl.DataFrame(
        {
            "uid": [1, 1, 1, 1, 2],
            "item_id": [10, 11, 10, 10, 10],
            "ts": [CUTOFF - DAY, CUTOFF - 10 * DAY, CUTOFF - 2 * DAY, CUTOFF, CUTOFF - 3 * DAY],
            "event_type": ["listen", "listen", "like", "listen", "listen"],
            "is_organic": [1, 0, 0, 1, 0],
            "played_ratio": [80, 40, None, 100, 50],
            "track_length_seconds": [200, 100, None, 200, 200],
        },
        schema_overrides={"played_ratio": pl.UInt16},
    )
    items = pd.DataFrame(
        {
            "item_id": [10, 11],
            "primary_artist_id": [100, 200],
            "primary_album_id": [1000, 1000],
        }
    )
    events_path = tmp_path / "events.parquet"
    items_path = tmp_path / "items.parquet"
    events.write_parquet(events_path)
    items.to_parquet(items_path, index=False)
    return events_path, items_path


def test_aggregates_are_strictly_before_cutoff(tmp_path):
    con = create_connection(*_write(tmp_path))
    specs = [
        FeatureSpec("user", "count", 7, "listen"),
        FeatureSpec("user", "count", None, "listen"),
        FeatureSpec("user", "count", None, "like"),
        FeatureSpec("item", "count", None, "listen"),
        FeatureSpec("user_item", "count", None, "listen"),
        FeatureSpec("user_item", "mean_played", None, "listen"),
        FeatureSpec("user_artist", "count", None, "listen"),
    ]
    out = compute_specs(con, specs, CUTOFF)

    user = out["user"].set_index("uid")
    # событие ровно на cutoff не попадает ни в один признак
    assert user.loc[1, "user__count__listen__all"] == 2
    assert user.loc[1, "user__count__listen__7d"] == 1
    assert user.loc[1, "user__count__like__all"] == 1
    assert user.loc[2, "user__count__listen__all"] == 1

    item = out["item"].set_index("item_id")
    assert item.loc[10, "item__count__listen__all"] == 2
    assert item.loc[11, "item__count__listen__all"] == 1

    ui = out["user_item"].set_index(["uid", "item_id"])
    assert ui.loc[(1, 10), "user_item__count__listen__all"] == 1
    assert ui.loc[(1, 10), "user_item__mean_played__listen__all"] == 80

    artist = out["user_artist"].set_index(["uid", "primary_artist_id"])
    assert artist.loc[(1, 100), "user_artist__count__listen__all"] == 1
    assert artist.loc[(1, 200), "user_artist__count__listen__all"] == 1
    con.close()
