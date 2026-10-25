from __future__ import annotations

import pandas as pd
import pytest

from recsys_auto.auto_features.expr import Expr, parse_expressions, spec_from_mapping
from recsys_auto.auto_features.generator import generate_specs
from recsys_auto.features.dsl import FeatureSpec


def test_grid_does_not_include_parameterised_metrics():
    specs = generate_specs()
    assert len(specs) == 900
    assert all(spec.metric != "decay_count" for spec in specs)


def test_condition_and_decay_are_encoded_in_name():
    complete = FeatureSpec("user_item", "count", 30, "listen", condition="complete")
    assert complete.name == "user_item__count__listen__30d__complete"
    decay = FeatureSpec("user_artist", "decay_count", 30, "listen", decay_days=7)
    assert decay.name == "user_artist__decay_count__listen__30d__decay7"
    assert decay.fill_zero


def test_unknown_condition_rejected():
    with pytest.raises(ValueError, match="unknown condition"):
        FeatureSpec("item", "count", 7, "listen", condition="maybe")


def test_decay_requires_and_forbids_half_life():
    with pytest.raises(ValueError, match="requires decay_days"):
        FeatureSpec("item", "decay_count", 7, "listen")
    with pytest.raises(ValueError, match="only supported for decay_count"):
        FeatureSpec("item", "count", 7, "listen", decay_days=7)


def test_expression_name_and_evaluation():
    a = FeatureSpec("user_item", "count", 7, "listen")
    b = FeatureSpec("user_item", "count", 30, "listen")
    expr = Expr("ratio", a, b)
    assert expr.name.startswith("expr__ratio__user_item__count__listen__7d")
    frame = pd.DataFrame({a.name: [1.0, 3.0], b.name: [2.0, 3.0]})
    values = expr.evaluate(frame)
    assert values.iloc[0] == pytest.approx(0.5)
    assert values.iloc[1] == pytest.approx(1.0)


def test_delta_and_product():
    a = FeatureSpec("item", "nunique_users", 7, "listen")
    b = FeatureSpec("item", "nunique_users", 30, "listen")
    frame = pd.DataFrame({a.name: [5.0], b.name: [20.0]})
    assert Expr("delta", a, b).evaluate(frame).iloc[0] == -15.0
    assert Expr("product", a, b).evaluate(frame).iloc[0] == 100.0


def test_parse_expressions_accepts_leaf_and_binary():
    payload = [
        {
            "op": "ratio",
            "a": {"entity": "user_item", "metric": "count", "event_type": "listen", "window_days": 7},
            "b": {"entity": "user_item", "metric": "count", "event_type": "listen", "window_days": 30},
        },
        {"entity": "item", "metric": "nunique_users", "event_type": "listen", "window_days": 14},
    ]
    accepted, errors = parse_expressions(payload)
    assert not errors
    assert len(accepted) == 2
    assert isinstance(accepted[0][0], Expr)
    assert isinstance(accepted[1][0], FeatureSpec)


def test_parse_expressions_reports_invalid_leaf():
    accepted, errors = parse_expressions(
        [
            {
                "op": "ratio",
                "a": {"entity": "item", "metric": "nunique_users", "event_type": "like", "window_days": 7},
                "b": {"entity": "item", "metric": "count", "event_type": "listen", "window_days": 7},
            }
        ]
    )
    assert not accepted
    assert errors


def test_spec_from_mapping_handles_none_markers():
    spec = spec_from_mapping(
        {
            "entity": "item",
            "metric": "count",
            "event_type": "any",
            "window_days": 0,
            "condition": "none",
            "decay_days": 0,
        }
    )
    assert spec.event_type is None
    assert spec.window_days is None
    assert spec.condition is None
    assert spec.decay_days is None
