"""CLI: оценить сохранённую модель auto_all на отложенном снимке.

Модель ``auto_all`` обучается на всём материализованном пуле (187 признаков)
без предотбора. Раньше её test-метрика не считалась, хотя именно на неё
опирается утверждение «автоматика догоняет ручной baseline». Скрипт
пересобирает тестовый фрейм из ``features_quota`` (базовые агрегаты +
derived-отношения) и считает те же метрики, что и ``04_train_baseline``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys_auto.auto_features.derived import derive_relations
from recsys_auto.config import load_config, resolve
from recsys_auto.eval.metrics import evaluate
from recsys_auto.features.dsl import INDEX_ONLY_METRICS
from recsys_auto.models.boosting import TrainedModel, add_scores, predict
from recsys_auto.utils.io import load_json, save_json


def drop_index_only(frame: pd.DataFrame) -> pd.DataFrame:
    raw = [
        c
        for c in frame.columns
        if (parts := c.split("__")) and len(parts) == 4 and parts[1] in INDEX_ONLY_METRICS
    ]
    return frame.drop(columns=raw)


def build_frame(cfg, snapshot: str, feature_cols: list[str]) -> pd.DataFrame:
    auto = resolve(cfg.paths.experiments) / "auto_search"
    frame = pd.read_parquet(auto / "features_quota" / f"{snapshot}.parquet")
    cutoff = int(frame["cutoff"].iloc[0])
    derived = derive_relations(frame, cutoff)
    frame = drop_index_only(frame)
    missing = [c for c in feature_cols if c not in frame.columns]
    for col in missing:
        if col in derived.columns:
            frame[col] = derived[col]
    still = [c for c in feature_cols if c not in frame.columns]
    if still:
        raise RuntimeError(f"missing {len(still)} features, e.g. {still[:3]}")
    return frame[["uid", "item_id", "label", "snapshot"] + feature_cols]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--model-dir", default="experiments/auto_search/quota/models/auto_all")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--out", default="reports/tables/auto_all_test.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_dir = resolve(args.model_dir)
    meta = load_json(model_dir / "model.meta.json")
    booster = lgb.Booster(model_file=str(model_dir / "model.txt"))
    model = TrainedModel(booster, meta["feature_cols"], meta["best_iteration"])
    frame = build_frame(cfg, args.split, model.feature_cols)
    scored = add_scores(frame, predict(model, frame))
    scored[["uid", "item_id", "label", "score", "snapshot"]].to_parquet(
        model_dir / f"scores_{args.split}.parquet", index=False
    )
    metrics = evaluate(scored, ks=tuple(cfg.eval.ks))
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_json(
        {"model_dir": str(model_dir), "split": args.split, "n_features": len(model.feature_cols), "metrics": metrics},
        out,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
