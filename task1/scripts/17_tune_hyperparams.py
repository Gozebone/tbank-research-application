"""CLI: подбор гиперпараметров LightGBM на валидации.

Сетка небольшая и обучается на подвыборке пользователей (быстро), лучшая
конфигурация затем переобучается на полных train-снимках. Тест используется
один раз — для итоговой конфигурации.

    python scripts/17_tune_hyperparams.py --tag quota
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys_auto.auto_features.derived import derive_relations
from recsys_auto.auto_features.search import load_candidates
from recsys_auto.config import load_config, resolve
from recsys_auto.eval.metrics import evaluate
from recsys_auto.features.dsl import INDEX_ONLY_METRICS
from recsys_auto.features.pipeline import feature_columns
from recsys_auto.models.boosting import (
    add_scores,
    feature_importance,
    predict,
    save_model,
    train_lgbm,
)
from recsys_auto.utils.io import ensure_dir, get_logger, load_json, save_json

log = get_logger("tune")

TRAIN_SNAPSHOTS = ["train_w0", "train_w1", "train_w2", "train_w3"]

GRID: list[dict] = [
    {"name": "base(lr.05,63,100,ff.8)", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 100, "feature_fraction": 0.8},
    {"name": "lr.03", "learning_rate": 0.03, "num_leaves": 63, "min_data_in_leaf": 100, "feature_fraction": 0.8},
    {"name": "lr.08,leaves127", "learning_rate": 0.08, "num_leaves": 127, "min_data_in_leaf": 100, "feature_fraction": 0.8},
    {"name": "leaves31,min50,ff.6", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 50, "feature_fraction": 0.6},
    {"name": "leaves127,min200,ff.6", "learning_rate": 0.05, "num_leaves": 127, "min_data_in_leaf": 200, "feature_fraction": 0.6},
    {"name": "min200,ff.9", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 200, "feature_fraction": 0.9},
]


def _drop_index_only(frame: pd.DataFrame) -> pd.DataFrame:
    raw = [
        c
        for c in frame.columns
        if (parts := c.split("__")) and len(parts) == 4 and parts[1] in INDEX_ONLY_METRICS
    ]
    return frame.drop(columns=raw)


def _auto(path: Path, needed: set[str], uids: np.ndarray | None) -> pd.DataFrame:
    filters = [("uid", "in", uids.tolist())] if uids is not None else None
    frame = pd.read_parquet(path, filters=filters)
    cutoff = int(frame["cutoff"].iloc[0])
    derived = derive_relations(frame, cutoff)
    frame = _drop_index_only(frame)
    keys = ["uid", "item_id", "snapshot"]
    base = [c for c in needed if c in frame.columns]
    rel = [c for c in needed if c in derived.columns]
    out = frame[keys + base].copy()
    if rel:
        out = pd.concat([out, derived[rel]], axis=1)
    return out


def _manual(snapshot: str, cfg, uids: np.ndarray | None) -> pd.DataFrame:
    frame = load_candidates(cfg, snapshot)
    if uids is not None:
        frame = frame[frame["uid"].isin(uids)].reset_index(drop=True)
    return frame


def build(cfg, snapshot: str, feature_cols: list[str], uids: np.ndarray | None) -> pd.DataFrame:
    manual = _manual(snapshot, cfg, uids)
    manual_cols = [c for c in feature_columns(manual) if c in feature_cols]
    needed = set(feature_cols) - set(manual.columns)
    auto = _auto(
        resolve(cfg.paths.experiments) / "auto_search" / "features_quota" / f"{snapshot}.parquet",
        needed,
        uids,
    )
    keys = ["uid", "item_id", "snapshot"]
    frame = manual[keys + ["label"] + manual_cols].merge(
        auto, on=keys, how="left", validate="one_to_one"
    )
    missing = [c for c in feature_cols if c not in frame.columns]
    if missing:
        raise RuntimeError(f"{snapshot}: missing {len(missing)} features: {missing[:3]}")
    frame[feature_cols] = frame[feature_cols].astype("float32")
    return frame[keys + ["label"] + feature_cols]


def params_with(cfg, entry: dict) -> dict:
    params = dict(cfg.model.params)
    for key in ("learning_rate", "num_leaves", "min_data_in_leaf", "feature_fraction"):
        params[key] = entry[key]
    return params


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--tag", default="quota")
    parser.add_argument("--train-users", type=int, default=2500)
    parser.add_argument("--val-users", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-final", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    auto = resolve(cfg.paths.experiments) / "auto_search"
    model_dir = auto / args.tag / "models" / "manual_plus_auto"
    feature_cols = load_json(model_dir / "model.meta.json")["feature_cols"]
    log.info("tuning on %d features from %s", len(feature_cols), args.tag)

    val_meta = load_candidates(cfg, "val", columns=["uid"])
    all_uids = val_meta["uid"].unique()
    rng = np.random.default_rng(args.seed)
    val_uids = rng.choice(all_uids, size=min(args.val_users, len(all_uids)), replace=False)
    train_uids = rng.choice(
        np.setdiff1d(all_uids, val_uids),
        size=min(args.train_users, len(all_uids) - len(val_uids)),
        replace=False,
    )
    log.info("tuning subsample: %d train users, %d val users", len(train_uids), len(val_uids))

    tune_train = pd.concat(
        [build(cfg, s, feature_cols, train_uids) for s in TRAIN_SNAPSHOTS], ignore_index=True
    )
    tune_val = build(cfg, "val", feature_cols, val_uids)
    log.info("tuning frames: train=%d, val=%d", len(tune_train), len(tune_val))

    rows = []
    best = None
    for entry in GRID:
        started = time.perf_counter()
        model = train_lgbm(
            tune_train,
            tune_val,
            feature_cols=feature_cols,
            params=params_with(cfg, entry),
            num_boost_rounds=int(cfg.model.num_boost_rounds),
            early_stopping_rounds=int(cfg.model.early_stopping_rounds),
        )
        metrics = evaluate(add_scores(tune_val, predict(model, tune_val)), ks=tuple(cfg.eval.ks))
        seconds = time.perf_counter() - started
        rows.append(
            {
                "config": entry["name"],
                "learning_rate": entry["learning_rate"],
                "num_leaves": entry["num_leaves"],
                "min_data_in_leaf": entry["min_data_in_leaf"],
                "feature_fraction": entry["feature_fraction"],
                "best_iteration": model.best_iteration,
                "val_ndcg@10": round(metrics["ndcg@10"], 5),
                "val_roc_auc": round(metrics["roc_auc"], 5),
                "seconds": round(seconds, 1),
            }
        )
        log.info("%s: val ndcg@10=%.5f (%.0fs)", entry["name"], metrics["ndcg@10"], seconds)
        if best is None or metrics["ndcg@10"] > best[1]["val_ndcg@10"]:
            best = (entry, rows[-1])
    table = pd.DataFrame(rows).sort_values("val_ndcg@10", ascending=False)
    tables = ensure_dir(resolve(cfg.paths.reports) / "tables")
    table.to_csv(tables / "hp_tuning.csv", index=False)
    print(table.to_string(index=False))
    del tune_train, tune_val

    payload = {
        "tag": args.tag,
        "grid": rows,
        "best": best[0],
        "n_train_users": len(train_uids),
        "n_val_users": len(val_uids),
    }
    if args.skip_final:
        save_json(payload, auto / "hp_tuning.json")
        return

    log.info("retraining best config on full data: %s", best[0]["name"])
    train = pd.concat([build(cfg, s, feature_cols, None) for s in TRAIN_SNAPSHOTS], ignore_index=True)
    valid = build(cfg, "val", feature_cols, None)
    started = time.perf_counter()
    model = train_lgbm(
        train,
        valid,
        feature_cols=feature_cols,
        params=params_with(cfg, best[0]),
        num_boost_rounds=int(cfg.model.num_boost_rounds),
        early_stopping_rounds=int(cfg.model.early_stopping_rounds),
    )
    seconds = time.perf_counter() - started
    out_dir = ensure_dir(auto / args.tag / "models" / "manual_plus_auto_tuned")
    save_model(model, out_dir)
    feature_importance(model).to_csv(out_dir / "feature_importance.csv", index=False)
    val_metrics = evaluate(add_scores(valid, predict(model, valid)), ks=tuple(cfg.eval.ks))
    del train, valid
    test = build(cfg, "test", feature_cols, None)
    test_metrics = evaluate(add_scores(test, predict(model, test)), ks=tuple(cfg.eval.ks))
    log.info("tuned val: %s", {k: round(v, 4) for k, v in val_metrics.items() if k in ("ndcg@10", "roc_auc")})
    log.info("tuned test: %s", {k: round(v, 4) for k, v in test_metrics.items() if k in ("ndcg@10", "roc_auc")})

    payload.update(
        {
            "final": {
                "params": best[0],
                "n_features": len(feature_cols),
                "train_seconds": seconds,
                "metrics": {"val": val_metrics, "test": test_metrics},
            }
        }
    )
    save_json(payload, auto / "hp_tuning.json")
    log.info("saved %s", auto / "hp_tuning.json")


if __name__ == "__main__":
    main()
