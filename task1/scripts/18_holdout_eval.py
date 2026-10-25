"""CLI: проверка обобщения на втором (дизъюнктном) сэмпле пользователей.

Признаки, гиперпараметры и пороги **не переподбираются**: берутся ровно те
наборы, что победили на первом сэмпле (manual + quota + expr), и обучаются
заново на новом сэмпле. Это проверяет, переносится ли вывод, а не подгоняется
ли он под конкретных пользователей.

    python scripts/18_holdout_eval.py --config configs/holdout.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys_auto.auto_features.derived import derive_relations
from recsys_auto.auto_features.expr import expression_from_mapping
from recsys_auto.auto_features.expr_search import (
    ExpressionScreener,
    materialize_expressions,
)
from recsys_auto.auto_features.search import load_candidates, materialize_specs
from recsys_auto.config import load_config, resolve
from recsys_auto.eval.metrics import evaluate
from recsys_auto.features.dsl import INDEX_ONLY_METRICS, FeatureSpec
from recsys_auto.features.pipeline import feature_columns
from recsys_auto.models.boosting import (
    add_scores,
    feature_importance,
    predict,
    save_model,
    train_lgbm,
)
from recsys_auto.utils.io import ensure_dir, get_logger, load_json, save_json

log = get_logger("holdout")

TRAIN = ["train_w0", "train_w1", "train_w2", "train_w3"]
ALL = TRAIN + ["val", "test"]


def _spec_from_dict(data: dict) -> FeatureSpec:
    return FeatureSpec(
        entity=data["entity"],
        metric=data["metric"],
        window_days=data.get("window_days"),
        event_type=data.get("event_type"),
        condition=data.get("condition"),
        decay_days=data.get("decay_days"),
    )


def _drop_index_only(frame: pd.DataFrame) -> pd.DataFrame:
    raw = [
        c
        for c in frame.columns
        if (parts := c.split("__")) and len(parts) == 4 and parts[1] in INDEX_ONLY_METRICS
    ]
    return frame.drop(columns=raw)


def _build(cfg, snapshot: str, feature_cols: list[str], flat_dir: Path, expr_dir: Path) -> pd.DataFrame:
    manual = load_candidates(cfg, snapshot)
    manual_cols = set(feature_columns(manual))
    needed = set(feature_cols) - manual_cols

    flat = pd.read_parquet(flat_dir / f"{snapshot}.parquet")
    cutoff = int(flat["cutoff"].iloc[0])
    derived = derive_relations(flat, cutoff)
    flat = _drop_index_only(flat)
    keys = ["uid", "item_id", "snapshot"]
    from_flat = [c for c in needed if c in flat.columns]
    from_rel = [c for c in needed if c in derived.columns]
    extra = flat[keys + from_flat].copy()
    if from_rel:
        extra = pd.concat([extra, derived[from_rel]], axis=1)

    expr = pd.read_parquet(expr_dir / f"{snapshot}.parquet")
    from_expr = [c for c in needed if c in expr.columns]
    if from_expr:
        extra = extra.merge(expr[keys + from_expr], on=keys, how="left")

    frame = manual[keys + ["label"] + [c for c in manual_cols if c in feature_cols]].merge(
        extra, on=keys, how="left", validate="one_to_one"
    )
    missing = [c for c in feature_cols if c not in frame.columns]
    if missing:
        raise RuntimeError(f"{snapshot}: missing {len(missing)} features: {missing[:3]}")
    frame[feature_cols] = frame[feature_cols].astype("float32")
    return frame[keys + ["label"] + feature_cols]


def _train(cfg, name, train, valid, cols, out_dir):
    model = train_lgbm(
        train,
        valid,
        feature_cols=cols,
        params=dict(cfg.model.params),
        num_boost_rounds=int(cfg.model.num_boost_rounds),
        early_stopping_rounds=int(cfg.model.early_stopping_rounds),
    )
    path = ensure_dir(out_dir / name)
    save_model(model, path)
    feature_importance(model).to_csv(path / "feature_importance.csv", index=False)
    val = evaluate(add_scores(valid, predict(model, valid)), ks=tuple(cfg.eval.ks))
    log.info("%s val ndcg@10=%.4f", name, val["ndcg@10"])
    return model, val


def _per_user_ndcg(df: pd.DataFrame, k: int = 10) -> pd.Series:
    ordered = df.sort_values(["uid", "score"], ascending=[True, False], kind="mergesort")
    out = {}
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


def _bootstrap(a: pd.Series, b: pd.Series, n: int = 2000, seed: int = 42) -> dict:
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


def _survival(cfg, origin: Path, flat_specs: list[FeatureSpec], exprs: list, sample_users: int):
    """Насколько признаки агента конкурентны на новом сэмпле."""
    from recsys_auto.auto_features.search import screen_specs

    grid_base = [
        _spec_from_dict(s) for s in load_json(origin / "selected_specs_quota.json")["base_specs"]
    ]
    flat_pool = [s for s in flat_specs if s.name not in {g.name for g in grid_base}]
    frame = screen_specs(cfg, grid_base + flat_pool, sample_users=sample_users, save=False)
    frame["kind"] = np.where(frame["name"].isin({g.name for g in grid_base}), "grid", "flat_agent")

    screener = ExpressionScreener(cfg, sample_users=sample_users)
    try:
        expr_frame = screener.score(exprs)
    finally:
        screener.close()
    expr_frame["kind"] = "expr_agent"
    combined = pd.concat([frame, expr_frame], ignore_index=True)
    combined["strength"] = (combined["user_auc"] - 0.5).abs()
    grid = combined[combined["kind"] == "grid"]
    cutoff = grid.groupby("entity")["strength"].apply(lambda s: s.nlargest(20).min())
    agent = combined[combined["kind"] != "grid"].copy()
    agent["in_quota"] = agent["strength"] >= agent["entity"].map(cutoff)
    return agent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/holdout.yaml")
    parser.add_argument("--origin", default="experiments/auto_search")
    parser.add_argument("--sample-users", type=int, default=1000)
    parser.add_argument("--out-reports", default="reports")
    parser.add_argument("--prefix", default="holdout", help="префикс имён артефактов")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    auto = resolve(cfg.paths.experiments) / "auto_search"
    origin = resolve(args.origin)
    flat_dir = ensure_dir(auto / "features_quota")
    expr_dir = ensure_dir(auto / "expr_features_expr_add")

    # 1) фиксированные наборы признаков с первого сэмпла
    flat_selection = load_json(origin / "selected_specs_quota.json")
    flat_specs = [_spec_from_dict(s) for s in flat_selection["base_specs"] + flat_selection["companion_specs"]]
    expr_pool = {item["name"]: item for item in load_json(origin / "agent_pool_expr_qwen.json")["expressions"]}
    chosen = load_json(origin / "expr_selection_expr_add.json")["chosen"]
    exprs = [expression_from_mapping(expr_pool[name]["expr"]) for name in chosen]
    all_exprs = [expression_from_mapping(item["expr"]) for item in expr_pool.values()]
    log.info(
        "transferred: %d flat specs, %d expressions (%d in pool)",
        len(flat_specs),
        len(exprs),
        len(all_exprs),
    )

    # 2) материализация на новом сэмпле
    for snapshot in ALL:
        out = flat_dir / f"{snapshot}.parquet"
        if args.force or not out.exists():
            materialize_specs(cfg, flat_specs, snapshot, out)
        out = expr_dir / f"{snapshot}.parquet"
        if args.force or not out.exists():
            materialize_expressions(cfg, exprs, snapshot, out)

    # 3) точные наборы фичей победивших моделей
    union_meta = load_json(origin / "quota" / "models" / "manual_plus_auto" / "model.meta.json")
    expr_meta = load_json(origin / "expr_add" / "models" / "manual_plus_expr" / "model.meta.json")
    manual_cols = feature_columns(load_candidates(cfg, "val"))
    union_cols = union_meta["feature_cols"]
    expr_cols = expr_meta["feature_cols"]

    train_union = pd.concat([_build(cfg, s, union_cols, flat_dir, expr_dir) for s in TRAIN], ignore_index=True)
    val_union = _build(cfg, "val", union_cols, flat_dir, expr_dir)
    log.info("union frames: train=%d, val=%d", len(train_union), len(val_union))

    out_dir = ensure_dir(auto / "models")
    results = {}

    # baseline
    train_manual = pd.concat([load_candidates(cfg, s) for s in TRAIN], ignore_index=True)
    val_manual = load_candidates(cfg, "val")
    model, val = _train(cfg, "baseline", train_manual, val_manual, manual_cols, out_dir)
    results["baseline"] = {"model": model, "val": val, "cols": manual_cols}
    del train_manual, val_manual

    # quota union
    model, val = _train(cfg, "quota_union", train_union, val_union, union_cols, out_dir)
    results["quota_union"] = {"model": model, "val": val, "cols": union_cols}
    del train_union, val_union

    # expr union
    train_expr = pd.concat([_build(cfg, s, expr_cols, flat_dir, expr_dir) for s in TRAIN], ignore_index=True)
    val_expr = _build(cfg, "val", expr_cols, flat_dir, expr_dir)
    model, val = _train(cfg, "expr_union", train_expr, val_expr, expr_cols, out_dir)
    results["expr_union"] = {"model": model, "val": val, "cols": expr_cols}
    del train_expr, val_expr

    # 4) test и bootstrap
    test_manual = load_candidates(cfg, "test")
    per_user = {}
    rows = []
    for name, payload in results.items():
        if name == "baseline":
            frame = test_manual
        else:
            frame = _build(cfg, "test", payload["cols"], flat_dir, expr_dir)
        scored = add_scores(frame, predict(payload["model"], frame))
        metrics = evaluate(scored, ks=tuple(cfg.eval.ks))
        payload["test"] = metrics
        per_user[name] = _per_user_ndcg(scored)
        rows.append(
            {
                "approach": name,
                "n_features": len(payload["cols"]),
                "val_ndcg@10": round(payload["val"]["ndcg@10"], 4),
                "test_ndcg@10": round(metrics["ndcg@10"], 4),
                "test_roc_auc": round(metrics["roc_auc"], 4),
                "test_recall@10": round(metrics["recall@10"], 4),
                "test_map@10": round(metrics["map@10"], 4),
            }
        )
        log.info("%s test ndcg@10=%.4f", name, metrics["ndcg@10"])
        del frame, scored

    common = set.intersection(*(set(s.index) for s in per_user.values()))
    per_user = {k: v.loc[sorted(common)] for k, v in per_user.items()}
    significance = []
    for left, right in [("baseline", "quota_union"), ("baseline", "expr_union"), ("quota_union", "expr_union")]:
        stats = _bootstrap(per_user[left], per_user[right])
        significance.append({"comparison": f"{right} - {left}", **stats})

    # 5) конкурентоспособность признаков агента на новом сэмпле
    flat_agent_specs = [
        _spec_from_dict(item)
        for item in load_json(origin / "agent_pool_quota.json")["specs"]
    ]
    survival = _survival(cfg, origin, flat_agent_specs, all_exprs, args.sample_users)
    survival_rows = (
        survival.groupby("kind")["in_quota"]
        .agg(["sum", "count", "mean"])
        .reset_index()
        .rename(columns={"sum": "n_in_quota", "count": "n_features", "mean": "share_in_quota"})
    )
    survival_rows["share_in_quota"] = survival_rows["share_in_quota"].round(3)

    # 6) артефакты
    out_reports = ensure_dir(resolve(args.out_reports))
    tables = ensure_dir(out_reports / "tables")
    metrics_table = pd.DataFrame(rows)
    metrics_table.to_csv(tables / f"{args.prefix}_metrics.csv", index=False)
    sig_table = pd.DataFrame(significance)
    sig_table.to_csv(tables / f"{args.prefix}_significance.csv", index=False)
    survival_rows.to_csv(tables / f"{args.prefix}_feature_survival.csv", index=False)

    payload = {
        "config": args.config,
        "n_users_total": int(cfg.data.yambda.n_users),
        "n_users_test": int(len(common)),
        "n_train_rows": None,
        "metrics": {name: {"val": r["val"], "test": r.get("test")} for name, r in results.items()},
        "significance": significance,
        "survival": survival_rows.to_dict(orient="records"),
    }
    save_json(payload, auto / f"metrics_{args.prefix}.json")
    save_json(payload, tables / f"{args.prefix}_summary.json")
    log.info("saved holdout artifacts to %s", tables)
    print(metrics_table.to_string(index=False))
    print(sig_table.to_string(index=False))
    print(survival_rows.to_string(index=False))


if __name__ == "__main__":
    main()
