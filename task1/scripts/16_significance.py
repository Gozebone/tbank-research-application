"""CLI: значимость различий и сегментный анализ.

Пересчитывает test-скоры сохранёнными моделями (baseline, quota union,
expr union), затем делает paired bootstrap по пользователям: доверительный
интервал для ΔNDCG@10 и долю ресэмплов, где разница не положительна.

Дополнительно — NDCG по квартилям активности пользователя: видно, на каких
пользователях объединение признаков даёт выигрыш.

    python scripts/16_significance.py --bootstrap 2000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys_auto.auto_features.derived import derive_relations
from recsys_auto.auto_features.search import load_candidates
from recsys_auto.config import load_config, resolve
from recsys_auto.features.dsl import INDEX_ONLY_METRICS
from recsys_auto.features.pipeline import feature_columns
from recsys_auto.models.boosting import TrainedModel, add_scores, predict
from recsys_auto.utils.io import ensure_dir, get_logger, load_json, save_json

log = get_logger("significance")

KS = 10


def _drop_index_only(frame: pd.DataFrame) -> pd.DataFrame:
    raw = [
        c
        for c in frame.columns
        if (parts := c.split("__")) and len(parts) == 4 and parts[1] in INDEX_ONLY_METRICS
    ]
    return frame.drop(columns=raw)


def _auto_columns(path: Path, needed: set[str]) -> pd.DataFrame:
    """Нужные auto-признаки (базовые + производные) из материализованного файла."""
    frame = pd.read_parquet(path)
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


def _plain_columns(path: Path, needed: set[str]) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    keys = ["uid", "item_id", "snapshot"]
    keep = [c for c in needed if c in frame.columns]
    return frame[keys + keep].copy()


def _load_model(path: Path) -> TrainedModel:
    import lightgbm as lgb

    meta = load_json(path / "model.meta.json")
    booster = lgb.Booster(model_file=str(path / "model.txt"))
    return TrainedModel(
        booster=booster,
        feature_cols=meta["feature_cols"],
        best_iteration=meta["best_iteration"],
    )


def _scores(
    model_dir: Path, manual: pd.DataFrame, extra_frames: list[pd.DataFrame]
) -> pd.DataFrame:
    model = _load_model(model_dir)
    needed = set(model.feature_cols) - set(manual.columns)
    frame = manual.copy()
    for extra in extra_frames:
        keep = [c for c in extra.columns if c in needed or c in ("uid", "item_id", "snapshot")]
        frame = frame.merge(extra[keep], on=["uid", "item_id", "snapshot"], how="left")
    missing = [c for c in model.feature_cols if c not in frame.columns]
    if missing:
        raise RuntimeError(f"{model_dir}: missing {len(missing)} features, e.g. {missing[:3]}")
    frame["score"] = predict(model, frame)
    return frame[["uid", "item_id", "label", "score"]]


def _per_user_ndcg(df: pd.DataFrame, k: int = KS) -> pd.Series:
    ordered = df.sort_values(["uid", "score"], ascending=[True, False], kind="mergesort")
    out: dict[int, float] = {}
    for uid, group in ordered.groupby("uid", sort=False):
        labels = group["label"].to_numpy()[:k]
        if labels.sum() == 0:
            out[uid] = 0.0
            continue
        dcg = float((labels / np.log2(np.arange(2, len(labels) + 2))).sum())
        ideal = min(int(group["label"].sum()), k)
        idcg = float((1.0 / np.log2(np.arange(2, ideal + 2))).sum())
        out[uid] = dcg / idcg if idcg > 0 else 0.0
    return pd.Series(out, name="ndcg@10")


def _bootstrap(a: pd.Series, b: pd.Series, n: int, seed: int = 42) -> dict:
    """Paired bootstrap: b - a по пользователям."""
    rng = np.random.default_rng(seed)
    index = np.arange(len(a))
    deltas = np.empty(n)
    for i in range(n):
        sample = rng.choice(index, size=len(index), replace=True)
        deltas[i] = b.iloc[sample].mean() - a.iloc[sample].mean()
    return {
        "delta": float(b.mean() - a.mean()),
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
        "p_not_positive": float((deltas <= 0).mean()),
        "n_users": int(len(a)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config()
    base = resolve(cfg.paths.experiments)
    auto = base / "auto_search"
    manual_test = load_candidates(cfg, "test")
    manual_cols = set(feature_columns(manual_test))
    manual_test = manual_test[["uid", "item_id", "snapshot", "label"] + sorted(manual_cols)]

    scores: dict[str, pd.DataFrame] = {}

    baseline_path = base / "baseline_manual" / "scores_test.parquet"
    baseline = pd.read_parquet(baseline_path, columns=["uid", "item_id", "label", "score"])
    scores["baseline"] = baseline

    quota_model = auto / "quota" / "models" / "manual_plus_auto"
    needed = set(_load_model(quota_model).feature_cols) - manual_cols
    quota_auto = _auto_columns(auto / "features_quota" / "test.parquet", needed)
    scores["quota_union"] = _scores(quota_model, manual_test, [quota_auto])
    del quota_auto

    expr_model = auto / "expr_add" / "models" / "manual_plus_expr"
    needed = set(_load_model(expr_model).feature_cols) - manual_cols
    expr_auto = _auto_columns(auto / "features_quota" / "test.parquet", needed)
    expr_extra = _plain_columns(auto / "expr_features_expr_add" / "test.parquet", needed)
    scores["expr_union"] = _scores(expr_model, manual_test, [expr_auto, expr_extra])
    del expr_auto, expr_extra

    tuned_model = auto / "quota" / "models" / "manual_plus_auto_tuned"
    if tuned_model.exists():
        needed = set(_load_model(tuned_model).feature_cols) - manual_cols
        tuned_auto = _auto_columns(auto / "features_quota" / "test.parquet", needed)
        scores["tuned_union"] = _scores(tuned_model, manual_test, [tuned_auto])
        del tuned_auto
    del manual_test

    per_user = {name: _per_user_ndcg(df) for name, df in scores.items()}
    common = set.intersection(*(set(s.index) for s in per_user.values()))
    per_user = {name: s.loc[sorted(common)] for name, s in per_user.items()}
    for name, series in per_user.items():
        log.info("%s: NDCG@10=%.4f, users=%d", name, series.mean(), len(series))

    comparisons = [
        ("baseline", "quota_union"),
        ("baseline", "expr_union"),
        ("quota_union", "expr_union"),
    ]
    if "tuned_union" in per_user:
        comparisons.extend(
            [("baseline", "tuned_union"), ("quota_union", "tuned_union")]
        )
    rows = []
    for left, right in comparisons:
        stats = _bootstrap(per_user[left], per_user[right], args.bootstrap, args.seed)
        rows.append({"comparison": f"{right} - {left}", **stats})
    table = pd.DataFrame(rows)
    tables = ensure_dir(resolve(cfg.paths.reports) / "tables")
    table.to_csv(tables / "significance.csv", index=False)
    save_json(
        {
            "k": KS,
            "bootstrap": args.bootstrap,
            "seed": args.seed,
            "ndcg": {name: round(float(s.mean()), 4) for name, s in per_user.items()},
            "comparisons": table.to_dict(orient="records"),
        },
        tables / "significance.json",
    )
    print(table.to_string(index=False))

    # сегментный анализ по активности пользователя
    activity = (
        pd.read_parquet(
            resolve(cfg.paths.processed) / "datasets" / "test.parquet",
            columns=["uid", "user__count__listen__all"],
        )
        .drop_duplicates("uid")
        .set_index("uid")["user__count__listen__all"]
    )
    activity = activity.loc[activity.index.isin(common)]
    segments = pd.qcut(activity.rank(method="first"), 4, labels=["q1_low", "q2", "q3", "q4_high"])
    rows = []
    for segment in segments.cat.categories:
        uids = segments[segments == segment].index
        row = {"segment": str(segment), "n_users": len(uids)}
        for name, series in per_user.items():
            row[name] = round(float(series.loc[uids].mean()), 4)
        row["delta_union"] = round(row["quota_union"] - row["baseline"], 4)
        row["delta_expr"] = round(row["expr_union"] - row["baseline"], 4)
        rows.append(row)
    segment_table = pd.DataFrame(rows)
    segment_table.to_csv(tables / "segment_analysis.csv", index=False)
    print(segment_table.to_string(index=False))


if __name__ == "__main__":
    main()
