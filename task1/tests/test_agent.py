from __future__ import annotations

import json

import pandas as pd
import pytest

from recsys_auto.agent.schema import SpecCandidate, parse_candidates
from recsys_auto.agent.tools import AgentTools
from recsys_auto.config import load_config
from recsys_auto.features.dsl import FeatureSpec


def test_candidate_maps_any_and_zero_window():
    candidate = SpecCandidate(
        entity="user_item", metric="count", event_type="any", window_days=0
    )
    spec = candidate.to_spec()
    assert spec.event_type is None
    assert spec.window_days is None
    assert spec.name == "user_item__count__any__all"


def test_candidate_accepts_non_grid_window():
    candidate = SpecCandidate(
        entity="item", metric="nunique_users", event_type="listen", window_days=45
    )
    assert candidate.to_spec().name == "item__nunique_users__listen__45d"


def test_parse_candidates_reports_errors_without_raising():
    payload = [
        {"entity": "item", "metric": "count", "event_type": "listen", "window_days": 7},
        {"entity": "bogus", "metric": "count", "event_type": "listen", "window_days": 7},
        {"entity": "item", "metric": "sum_played", "event_type": "like", "window_days": 7},
        {"entity": "item", "metric": "count", "event_type": "listen", "window_days": 7},
    ]
    accepted, errors = parse_candidates(payload)
    assert [spec.name for spec, _ in accepted] == ["item__count__listen__7d"]
    # неизвестная сущность, несовместимая метрика и дубль внутри батча
    assert len(errors) == 3


def test_parse_candidates_rejects_bad_event_type():
    accepted, errors = parse_candidates(
        [{"entity": "item", "metric": "count", "event_type": "skip", "window_days": 7}]
    )
    assert not accepted
    assert "event_type" in errors[0]["error"] or "must be one of" in errors[0]["error"]


@pytest.fixture()
def tools(tmp_path, monkeypatch):
    cfg = load_config()
    instance = AgentTools(
        cfg,
        tag="test",
        sample_users=10,
        pool_path=tmp_path / "pool.json",
        cache_path=tmp_path / "cache.json",
    )
    monkeypatch.setattr(instance, "_grid_screen", lambda: None)

    def fake_screen(cfg_, specs, sample_users=10, seed=42, save=True, **_):
        rows = []
        for spec in specs:
            strong = spec.metric == "count"
            rows.append(
                {
                    "name": spec.name,
                    "entity": spec.entity,
                    "metric": spec.metric,
                    "event_type": spec.event_type,
                    "window_days": spec.window_days,
                    "auc": 0.8 if strong else 0.51,
                    "user_auc": 0.75 if strong else 0.51,
                    "nan_rate": 0.05,
                }
            )
        return pd.DataFrame(rows)

    monkeypatch.setattr("recsys_auto.agent.tools.screen_specs", fake_screen)
    return instance


def test_propose_specs_accepts_strong_and_rejects_weak(tools):
    result = tools.propose_specs(
        [
            {"entity": "item", "metric": "count", "event_type": "listen", "window_days": 10},
            {"entity": "item", "metric": "mean_played", "event_type": "listen", "window_days": 10},
        ]
    )
    statuses = {item["name"]: item["status"] for item in result["feedback"]}
    assert statuses["item__count__listen__10d"] == "accepted"
    assert statuses["item__mean_played__listen__10d"] == "rejected"
    assert result["pool_size"] == 1
    assert tools.pool_path.exists()


def test_propose_specs_rejects_index_only_metric(tools):
    result = tools.propose_specs(
        [{"entity": "user_item", "metric": "last_ts", "event_type": "any", "window_days": 0}]
    )
    # last_ts нет в enum тула, но валидация схемы всё равно должна отсечь его
    assert result["pool_size"] == 0
    assert result["errors"]


def test_second_propose_uses_cache(tools, monkeypatch):
    calls = {"n": 0}
    original = pd.DataFrame

    def counting_screen(*args, **kwargs):
        calls["n"] += 1
        return original(
            [
                {
                    "name": "item__count__listen__10d",
                    "entity": "item",
                    "metric": "count",
                    "event_type": "listen",
                    "window_days": 10,
                    "auc": 0.8,
                    "user_auc": 0.75,
                    "nan_rate": 0.05,
                }
            ]
        )

    monkeypatch.setattr("recsys_auto.agent.tools.screen_specs", counting_screen)
    payload = [{"entity": "item", "metric": "count", "event_type": "listen", "window_days": 10}]
    tools.propose_specs(payload)
    tools.propose_specs(payload)
    assert calls["n"] == 1  # второй раз берём из кеша


def test_diversity_cap_limits_pair(tools):
    specs = [
        {
            "entity": "user_artist",
            "metric": "count",
            "event_type": "listen",
            "window_days": window,
        }
        for window in (10, 20, 30, 40)
    ]
    result = tools.propose_specs(specs)
    accepted = [f for f in result["feedback"] if f["status"] == "accepted"]
    assert len(accepted) == tools.max_per_entity_metric
    assert any("diversity cap" in f["reason"] for f in result["feedback"])


def test_repeated_accepted_spec_is_rejected_without_growth(tools):
    payload = [
        {"entity": "item", "metric": "count", "event_type": "listen", "window_days": 10}
    ]
    tools.propose_specs(payload)
    result = tools.propose_specs(payload)
    assert result["pool_size"] == 1
    assert any("already accepted" in err["error"] for err in result["errors"])


def test_pool_roundtrip(tools):
    tools.propose_specs(
        [{"entity": "item", "metric": "count", "event_type": "listen", "window_days": 10}]
    )
    payload = json.loads(tools.pool_path.read_text(encoding="utf-8"))
    assert payload["n_specs"] == 1
    spec = FeatureSpec(
        entity=payload["specs"][0]["entity"],
        metric=payload["specs"][0]["metric"],
        window_days=payload["specs"][0]["window_days"],
        event_type=payload["specs"][0]["event_type"],
    )
    assert spec.name == "item__count__listen__10d"
