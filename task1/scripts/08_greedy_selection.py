"""CLI: жадный forward selection признаков по приросту NDCG@10.

Идея: на подвыборке пользователей последовательно добавляем признак, который
даёт максимальный прирост метрики ранжирования. Это учитывает
комплементарность признаков, чего не делает одномерный AUC-скрининг.

Кандидаты берутся из top-gain модели ``auto_all``. Финальная модель с
отобранным набором обучается на полных данных и сравнивается с остальными
вариантами на валидации; тест — только если она лучшая.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys_auto.auto_features.derived import derive_relations
from recsys_auto.config import load_config, resolve
from recsys_auto.eval.metrics import evaluate
from recsys_auto.features.dsl import INDEX_ONLY_METRICS
from recsys_auto.features.pipeline import META_COLUMNS
from recsys_auto.models.boosting import add_scores, feature_importance, predict, save_model, train_lgbm
from recsys_auto.utils.io import ensure_dir, get_logger, load_json, save_json

log = get_logger("greedy")

TRAIN_SNAPSHOTS = ["train_w0", "train_w1", "train_w2", "train_w3"]


def _paths(cfg, tag: str) -> dict[str, Path]:
    base = resolve(cfg.paths.experiments) / "auto_search"
    return {
        "base": base,
        "features": base / f"features_{tag}",
        "derived": base / f"derived_selection_{tag}.json",
        "metrics": base / f"metrics_{tag}.json",
        "models": ensure_dir(base / tag / "models"),
        "greedy": base / f"greedy_{tag}.json",
    }


def _raw_ts_columns(frame: pd.DataFrame) -> list[str]:
    return [
        c
        for c in frame.columns
        if (parts := c.split("__")) and len(parts) == 4 and parts[1] in INDEX_ONLY_METRICS
    ]


def load_frame(
    paths: dict,
    snapshot: str,
    base_cols: list[str],
    derived_cols: list[str],
    uids: np.ndarray | None = None,
) -> pd.DataFrame:
    """Загрузить снимок, посчитать производные, вернуть нужные колонки."""
    path = paths["features"] / f"{snapshot}.parquet"
    read_kwargs = {}
    if uids is not None:
        read_kwargs["filters"] = [("uid", "in", uids.tolist())]
    frame = pd.read_parquet(path, **read_kwargs)
    cutoff = int(frame["cutoff"].iloc[0])
    derived = derive_relations(frame, cutoff)
    frame = frame.drop(columns=_raw_ts_columns(frame))
    meta = [c for c in META_COLUMNS if c in frame.columns]
    out = frame[meta + [c for c in base_cols if c in frame.columns]].copy()
    keep = [c for c in derived_cols if c in derived.columns]
    if keep:
        out = pd.concat([out, derived[keep]], axis=1)
    return out


def fast_ndcg10(frame: pd.DataFrame, scores: np.ndarray) -> float:
    data = frame[["uid", "label"]].copy()
    data["score"] = scores
    # случайный порядок для корректного разрешения равных скоров
    data = data.sample(frac=1.0, random_state=42).reset_index(drop=True)
    data = data.sort_values(["uid", "score"], ascending=[True, False], kind="mergesort")
    values: list[float] = []
    for _, group in data.groupby("uid", sort=False):
        labels = group["label"].to_numpy()[:10]
        if labels.sum() == 0:
            values.append(0.0)
            continue
        dcg = float((labels / np.log2(np.arange(2, len(labels) + 2))).sum())
        ideal_len = min(int(group["label"].sum()), 10)
        idcg = float((1.0 / np.log2(np.arange(2, ideal_len + 2))).sum())
        values.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(values))


def fast_params(cfg: dict, threads: int = 12) -> dict:
    params = dict(cfg.model.params)
    params.update(
        {
            "learning_rate": 0.1,
            "num_leaves": 15,
            "min_data_in_leaf": 50,
            "feature_fraction": 1.0,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "num_threads": threads,
            "verbosity": -1,
            "metric": "None",
        }
    )
    return params


def train_fast(train: pd.DataFrame, valid: pd.DataFrame, cols: list[str], params: dict) -> float:
    dataset = lgb.Dataset(train[cols], label=train["label"].to_numpy(), free_raw_data=False)
    booster = lgb.train(params, dataset, num_boost_round=60)
    scores = booster.predict(valid[cols])
    return fast_ndcg10(valid, scores)


def greedy_select(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    candidates: list[str],
    params: dict,
    max_features: int,
    tolerance: float = 1e-4,
) -> tuple[list[str], list[dict]]:
    selected: list[str] = []
    remaining = list(candidates)
    history: list[dict] = []
    current = 0.0
    started = time.perf_counter()
    for step in range(max_features):
        best_feature, best_score = None, current
        for feature in remaining:
            score = train_fast(train, valid, selected + [feature], params)
            if score > best_score:
                best_feature, best_score = feature, score
        if best_feature is None or best_score - current <= tolerance:
            log.info("stop at step %d: no improvement", step)
            break
        selected.append(best_feature)
        remaining.remove(best_feature)
        history.append(
            {
                "step": step + 1,
                "feature": best_feature,
                "ndcg@10": best_score,
                "gain": best_score - current,
                "elapsed_s": time.perf_counter() - started,
            }
        )
        current = best_score
        log.info(
            "step %d: +%s -> ndcg@10=%.5f (+%.5f), %.0fs",
            step + 1,
            best_feature,
            best_score,
            history[-1]["gain"],
            time.perf_counter() - started,
        )
    return selected, history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="quota")
    parser.add_argument("--pool-size", type=int, default=50)
    parser.add_argument("--max-features", type=int, default=12)
    parser.add_argument("--train-users", type=int, default=500)
    parser.add_argument("--val-users", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config()
    paths = _paths(cfg, args.tag)
    rng = np.random.default_rng(args.seed)

    importance_path = paths["models"] / "auto_all" / "feature_importance.csv"
    if not importance_path.exists():
        raise FileNotFoundError(
            f"{importance_path} not found; run --stage all-features first"
        )
    importance = pd.read_csv(importance_path)
    candidates = importance.query("gain > 0").head(args.pool_size)["feature"].tolist()
    log.info("greedy pool: %d features from auto_all gain", len(candidates))

    derived_payload = load_json(paths["derived"])
    derived_cols = derived_payload["selected"]

    # все колонки, нужные для производных, — их и читаем
    sample = pd.read_parquet(paths["features"] / "val.parquet", columns=["uid", "cutoff"])
    val_uids_all = sample["uid"].unique()
    val_uids = rng.choice(val_uids_all, size=min(args.val_users, len(val_uids_all)), replace=False)
    train_uids = rng.choice(
        np.setdiff1d(val_uids_all, val_uids),
        size=min(args.train_users, len(val_uids_all) - len(val_uids)),
        replace=False,
    )
    log.info("subsample: %d train users, %d val users", len(train_uids), len(val_uids))

    all_cols = [
        c
        for c in pq.ParquetFile(paths["features"] / "train_w0.parquet").schema_arrow.names
        if (parts := c.split("__")) and len(parts) == 4 and parts[1] not in INDEX_ONLY_METRICS
    ]
    train_parts = []
    for snapshot in TRAIN_SNAPSHOTS:
        part = load_frame(paths, snapshot, all_cols, derived_cols, uids=train_uids)
        train_parts.append(part)
        log.info("loaded %s: %d rows", snapshot, len(part))
    train = pd.concat(train_parts, ignore_index=True)
    valid = load_frame(paths, "val", all_cols, derived_cols, uids=val_uids)
    log.info("greedy train=%d rows, valid=%d rows", len(train), len(valid))

    # кандидаты должны существовать в обоих фреймах
    candidates = [c for c in candidates if c in train.columns and c in valid.columns]
    selected, history = greedy_select(
        train,
        valid,
        candidates,
        fast_params(cfg),
        max_features=args.max_features,
    )
    save_json(
        {
            "tag": args.tag,
            "pool_size": len(candidates),
            "n_train_users": len(train_uids),
            "n_val_users": len(val_uids),
            "n_train_rows": len(train),
            "n_val_rows": len(valid),
            "selected": selected,
            "history": history,
        },
        paths["greedy"],
    )
    log.info("selected %d features: %s", len(selected), selected)
    del train, valid
    if not selected:
        raise RuntimeError("greedy selection returned no features")

    # финальная модель на полных данных
    started_full = time.perf_counter()
    train_full_parts = []
    for snapshot in TRAIN_SNAPSHOTS:
        train_full_parts.append(load_frame(paths, snapshot, all_cols, derived_cols))
    train_full = pd.concat(train_full_parts, ignore_index=True)
    del train_full_parts
    val_full = load_frame(paths, "val", all_cols, derived_cols)
    log.info("full train=%d rows, valid=%d rows", len(train_full), len(val_full))

    model = train_lgbm(
        train_full,
        val_full,
        feature_cols=selected,
        params=dict(cfg.model.params),
        num_boost_rounds=int(cfg.model.num_boost_rounds),
        early_stopping_rounds=int(cfg.model.early_stopping_rounds),
    )
    train_seconds = time.perf_counter() - started_full
    out_dir = ensure_dir(paths["models"] / "auto_greedy")
    save_model(model, out_dir)
    feature_importance(model).to_csv(out_dir / "feature_importance.csv", index=False)
    scored_val = add_scores(val_full, predict(model, val_full))
    val_metrics = evaluate(scored_val, ks=tuple(int(k) for k in cfg.eval.ks))
    log.info("auto_greedy val: %s", val_metrics)
    del train_full, val_full

    payload = load_json(paths["metrics"])
    payload.setdefault("results", {})
    payload["results"]["auto_greedy"] = {
        "metrics": {"val": val_metrics},
        "train_seconds": train_seconds,
        "n_features": len(selected),
    }
    best = max(
        payload["results"],
        key=lambda k: payload["results"][k]["metrics"]["val"]["ndcg@10"],
    )
    payload["best_variant"] = best
    if best == "auto_greedy" and "test" not in payload["results"][best]["metrics"]:
        test = load_frame(paths, "test", all_cols, derived_cols)
        payload["results"]["auto_greedy"]["metrics"]["test"] = evaluate(
            add_scores(test, predict(model, test)),
            ks=tuple(int(k) for k in cfg.eval.ks),
        )
        log.info("auto_greedy test: %s", payload["results"]["auto_greedy"]["metrics"]["test"])
        del test
    save_json(payload, paths["metrics"])
    log.info("updated %s; best=%s", paths["metrics"], best)


if __name__ == "__main__":
    main()
