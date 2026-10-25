from __future__ import annotations

import numpy as np
import pandas as pd

from recsys_auto.auto_features.derived import derive_relations
from recsys_auto.auto_features.generator import generate_specs, is_valid_combination
from recsys_auto.auto_features.selectors import per_user_auc, univariate_auc


def test_generator_skips_invalid_metric_event_combinations():
    assert not is_valid_combination("user", "mean_played", "like")
    assert not is_valid_combination("item", "nunique_users", "like")
    assert is_valid_combination("user", "count", "like")
    specs = generate_specs()
    names = {s.name for s in specs}
    assert "user__count__listen__7d" in names
    assert "user__mean_played__listen__7d" in names
    assert "user__mean_played__like__7d" not in names
    assert len(specs) == len(names)


def test_per_user_auc_matches_global_for_single_user():
    df = pd.DataFrame(
        {
            "uid": [1, 1, 1, 1],
            "label": [1, 1, 0, 0],
            "perfect": [0.9, 0.8, 0.2, 0.1],
            "inverse": [0.1, 0.2, 0.8, 0.9],
        }
    )
    user = per_user_auc(df, ["perfect", "inverse"])
    glob = univariate_auc(df, ["perfect", "inverse"])
    assert user["perfect"] == glob["perfect"] == 1.0
    assert user["inverse"] == glob["inverse"] == 0.0


def test_per_user_auc_averages_users():
    df = pd.DataFrame(
        {
            "uid": [1, 1, 2, 2],
            "label": [1, 0, 0, 1],
            "feature": [0.9, 0.1, 0.9, 0.1],
        }
    )
    # первый пользователь отранжирован идеально, второй — наоборот
    assert per_user_auc(df, ["feature"]).loc["feature"] == 0.5


def test_derive_relations_builds_shares_and_recency():
    day = 86_400
    df = pd.DataFrame(
        {
            "user__count__listen__30d": [10.0],
            "user__last_ts__any__all": [100 * day],
            "user_item__count__listen__30d": [2.0],
            "user_item__count__listen__all": [8.0],
            "user_item__count__like__30d": [1.0],
            "user_item__last_ts__any__all": [95 * day],
        }
    )
    derived = derive_relations(df, cutoff=100 * day)
    assert abs(derived["auto_rel__user_item__share__30d"].iloc[0] - 2 / 11) < 1e-9
    assert derived["auto_rel__user_item__like_rate__30d"].iloc[0] == 1 / 3
    assert derived["auto_rel__user_item__days_since_last"].iloc[0] == 5.0
    assert derived["auto_rel__user_item__trend__30d"].iloc[0] == 2 / 9
