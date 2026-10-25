"""CLI: выживаемость признаков агента на KION относительно полной сетки 900."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys_auto.auto_features.expr import expression_from_mapping
from recsys_auto.auto_features.expr_search import ExpressionScreener
from recsys_auto.auto_features.search import screen_specs
from recsys_auto.config import load_config, resolve
from recsys_auto.features.dsl import INDEX_ONLY_METRICS, FeatureSpec
from recsys_auto.utils.io import ensure_dir, get_logger

log = get_logger("kion_survival")


def _spec(item: dict) -> FeatureSpec:
    return FeatureSpec(
        entity=item["entity"],
        metric=item["metric"],
        window_days=item.get("window_days"),
        event_type=item.get("event_type"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/kion.yaml")
    parser.add_argument("--origin", default="experiments/auto_search")
    args = parser.parse_args()

    cfg = load_config(args.config)
    base = resolve(cfg.paths.experiments) / "auto_search"
    origin = resolve(args.origin)

    grid = pd.read_csv(base / "screen.csv")
    grid = grid[(~grid["metric"].isin(INDEX_ONLY_METRICS)) & (grid["nan_rate"] < 0.999)].copy()
    grid["strength"] = (grid["user_auc"] - 0.5).abs()
    cutoff = grid.groupby("entity")["strength"].apply(lambda s: s.nlargest(20).min())

    flat = [_spec(item) for item in json.load(open(origin / "agent_pool_quota.json"))["specs"]]
    flat_frame = screen_specs(cfg, flat, sample_users=1000, save=False)
    flat_frame["kind"] = "flat_agent"

    pool = json.load(open(origin / "agent_pool_expr_qwen.json"))["expressions"]
    exprs = [expression_from_mapping(item["expr"]) for item in pool]
    screener = ExpressionScreener(cfg, sample_users=1000)
    try:
        expr_frame = screener.score(exprs)
    finally:
        screener.close()
    expr_frame["kind"] = "expr_agent"

    frame = pd.concat([flat_frame, expr_frame], ignore_index=True)
    frame["strength"] = (frame["user_auc"] - 0.5).abs()
    frame["in_quota"] = frame["strength"] >= frame["entity"].map(cutoff)
    summary = (
        frame.groupby("kind")["in_quota"]
        .agg(["sum", "count", "mean"])
        .reset_index()
        .rename(columns={"sum": "n_in_quota", "count": "n_features", "mean": "share_in_quota"})
    )
    summary["share_in_quota"] = summary["share_in_quota"].round(3)
    tables = ensure_dir(resolve("reports") / "tables")
    summary.to_csv(tables / "kion_survival.csv", index=False)
    frame[["name", "kind", "entity", "user_auc", "in_quota"]].to_csv(
        tables / "kion_survival_features.csv", index=False
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
