"""CLI: обучить baseline LightGBM на ручных признаках и оценить."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys_auto.config import load_config, resolve
from recsys_auto.eval.metrics import evaluate
from recsys_auto.features.pipeline import feature_columns, load_dataset
from recsys_auto.models.boosting import (
    add_scores,
    feature_importance,
    predict,
    save_model,
    train_lgbm,
)
from recsys_auto.utils.io import ensure_dir, get_logger, save_json

log = get_logger("train_baseline")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--train-snapshots", default="train_w0,train_w1,train_w2")
    parser.add_argument("--val", default="val")
    parser.add_argument("--test", default="test")
    parser.add_argument("--name", default="baseline_manual")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_names = [s.strip() for s in args.train_snapshots.split(",") if s.strip()]
    train = pd.concat([load_dataset(cfg, n) for n in train_names], ignore_index=True)
    valid = load_dataset(cfg, args.val)
    cols = [c for c in feature_columns(train) if c in valid.columns]
    log.info("train rows=%d, valid rows=%d, features=%d", len(train), len(valid), len(cols))

    model = train_lgbm(
        train,
        valid,
        feature_cols=cols,
        params=dict(cfg.model.params),
        num_boost_rounds=int(cfg.model.num_boost_rounds),
        early_stopping_rounds=int(cfg.model.early_stopping_rounds),
    )

    out_dir = ensure_dir(resolve(cfg.paths.experiments) / args.name)
    save_model(model, out_dir)
    feature_importance(model).to_csv(out_dir / "feature_importance.csv", index=False)

    results: dict[str, dict] = {}
    ks = tuple(int(k) for k in cfg.eval.ks)
    for name, frame in ((args.val, valid), (args.test, load_dataset(cfg, args.test))):
        scored = add_scores(frame, predict(model, frame))
        metrics = evaluate(scored, ks=ks)
        results[name] = metrics
        log.info("%s metrics: %s", name, metrics)
        scored[["uid", "item_id", "label", "score", "snapshot"]].to_parquet(
            out_dir / f"scores_{name}.parquet", index=False
        )
    save_json(
        {
            "train_snapshots": train_names,
            "n_features": len(cols),
            "best_iteration": model.best_iteration,
            "metrics": results,
        },
        out_dir / "metrics.json",
    )


if __name__ == "__main__":
    main()
