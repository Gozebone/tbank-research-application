"""CLI: обучение на пуле расширенного DSL (условия, затухание, выражения).

Берём готовый сеточный отбор (тег ``quota``), добавляем top-k выражений агента
и обучаем те же варианты, что в ``06_auto_search``: base, base+relations,
union с ручным набором. Сравнение идёт на одном протоколе.

    python scripts/13_train_expr_pool.py --pool expr_kimi --top-k 20 --tag expr_add
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
from recsys_auto.auto_features.expr import expression_from_mapping
from recsys_auto.auto_features.expr_search import ExpressionScreener, materialize_expressions
from recsys_auto.auto_features.search import load_candidates
from recsys_auto.auto_features.selectors import correlation_prune, univariate_auc
from recsys_auto.config import load_config, resolve
from recsys_auto.eval.metrics import evaluate
from recsys_auto.features.dsl import INDEX_ONLY_METRICS
from recsys_auto.features.pipeline import META_COLUMNS, feature_columns
from recsys_auto.models.boosting import (
    add_scores,
    feature_importance,
    predict,
    save_model,
    train_lgbm,
)
from recsys_auto.utils.io import ensure_dir, get_logger, load_json, save_json

log = get_logger("expr_train")

TRAIN_SNAPSHOTS = ["train_w0", "train_w1", "train_w2", "train_w3"]
ALL_SNAPSHOTS = TRAIN_SNAPSHOTS + ["val", "test"]


def _auto_base(cfg) -> Path:
    return resolve(cfg.paths.experiments) / "auto_search"


def _load_grid_frame(cfg, grid_tag: str, snapshot: str) -> pd.DataFrame:
    """Полный материализованный фрейм сеточного запуска (включая companions).

    Companions нужны производным признакам: recency/tenure строятся из
    ``last_ts``/``first_ts``, share — из ``user__count__listen__*``.
    """
    path = _auto_base(cfg) / f"features_{grid_tag}" / f"{snapshot}.parquet"
    return pd.read_parquet(path)


def _rel_columns(frame: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    return derive_relations(frame, cutoff)


def _select_derived(sample: pd.DataFrame, cutoff: int, top_k: int) -> list[str]:
    derived = _rel_columns(sample, cutoff)
    if not len(derived.columns):
        return []
    auc = univariate_auc(
        pd.concat([sample[["label"]], derived], axis=1), list(derived.columns)
    )
    auc = auc[(auc - 0.5).abs() > 0.01]
    ordered = list(auc.reindex((auc - 0.5).abs().sort_values(ascending=False).index).index)
    ordered = correlation_prune(derived, ordered, threshold=0.98)
    return ordered[:top_k]


def _train(cfg, name, train, valid, cols, out_dir):
    started = time.perf_counter()
    model = train_lgbm(
        train,
        valid,
        feature_cols=cols,
        params=dict(cfg.model.params),
        num_boost_rounds=int(cfg.model.num_boost_rounds),
        early_stopping_rounds=int(cfg.model.early_stopping_rounds),
    )
    seconds = time.perf_counter() - started
    save_model(model, out_dir / name)
    importance = feature_importance(model)
    importance.to_csv(out_dir / name / "feature_importance.csv", index=False)
    metrics = {"val": evaluate(add_scores(valid, predict(model, valid)), ks=tuple(cfg.eval.ks))}
    log.info("%s val ndcg@10=%.4f (%.0fs)", name, metrics["val"]["ndcg@10"], seconds)
    return {"model": model, "metrics": metrics, "train_seconds": seconds, "importance": importance}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--pool", default="expr_kimi", help="тег пула агента")
    parser.add_argument("--grid-tag", default="quota")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--derived-top-k", type=int, default=50)
    parser.add_argument("--tag", default="expr_add")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    base = _auto_base(cfg)
    pool = load_json(base / f"agent_pool_{args.pool}.json")
    if not pool.get("expressions"):
        raise ValueError(f"pool {args.pool} is not an expression pool")

    # 1) отбор выражений на полном скрининге (1000 пользователей)
    cand = [expression_from_mapping(item["expr"]) for item in pool["expressions"]]
    screener = ExpressionScreener(cfg, sample_users=1000)
    try:
        scores = screener.score(cand)
    finally:
        screener.close()
    chosen_names = scores.head(args.top_k)["name"].tolist()
    chosen = [expr for expr in cand if expr.name in set(chosen_names)]
    save_json(
        {
            "pool": args.pool,
            "top_k": args.top_k,
            "scores": scores.to_dict(orient="records"),
            "chosen": [expr.name for expr in chosen],
        },
        base / f"expr_selection_{args.tag}.json",
    )
    log.info("selected %d expressions out of %d", len(chosen), len(cand))

    # 2) материализация выражений для всех снимков
    feat_dir = ensure_dir(base / f"expr_features_{args.tag}")
    for snapshot in ALL_SNAPSHOTS:
        out = feat_dir / f"{snapshot}.parquet"
        if out.exists() and not args.force:
            log.info("skip %s (exists)", snapshot)
            continue
        materialize_expressions(cfg, chosen, snapshot, out)

    # 3) сеточные базовые признаки и их прунинг (как в 06)
    selection = load_json(base / f"selected_specs_{args.grid_tag}.json")
    base_names = [s["name"] for s in selection["base_specs"]]
    val_full = _load_grid_frame(cfg, args.grid_tag, "val")
    sample_idx = np.random.default_rng(42).choice(
        len(val_full), size=min(400_000, len(val_full)), replace=False
    )
    sample = val_full.iloc[sample_idx].reset_index(drop=True)
    base_pruned = correlation_prune(
        sample, [c for c in base_names if c in sample.columns], threshold=0.98
    )
    cutoff = int(val_full["cutoff"].iloc[0])
    derived_cols = _select_derived(sample, cutoff, args.derived_top_k)
    del val_full

    expr_names = [expr.name for expr in chosen]
    base_cols = list(base_pruned)
    log.info(
        "grid base after pruning: %d; expressions: %d; derived: %d",
        len(base_cols),
        len(expr_names),
        len(derived_cols),
    )

    # 4) сборка фреймов
    def build(snapshot: str) -> pd.DataFrame:
        grid = _load_grid_frame(cfg, args.grid_tag, snapshot)
        expr_frame = pd.read_parquet(feat_dir / f"{snapshot}.parquet")
        cols = [c for c in expr_names if c in expr_frame.columns]
        merged = grid.merge(
            expr_frame[["uid", "item_id", "snapshot"] + cols],
            on=["uid", "item_id", "snapshot"],
            how="left",
            validate="one_to_one",
        )
        cut = int(merged["cutoff"].iloc[0])
        derived = _rel_columns(merged, cut)
        keep = [c for c in derived_cols if c in derived.columns]
        if keep:
            merged = pd.concat([merged, derived[keep]], axis=1)
        wanted = [
            c
            for c in list(META_COLUMNS) + base_cols + cols + derived_cols
            if c in merged.columns
        ]
        return merged[wanted]

    train = pd.concat([build(s) for s in TRAIN_SNAPSHOTS], ignore_index=True)
    valid = build("val")
    log.info("train=%d rows, valid=%d rows", len(train), len(valid))

    expr_cols = [c for c in expr_names if c in train.columns]
    rel_cols = base_cols + expr_cols
    out_dir = ensure_dir(base / args.tag / "models")

    results: dict[str, dict] = {}
    results["auto_base_expr"] = _train(
        cfg, "auto_base_expr", train, valid, rel_cols, out_dir
    )
    results["auto_rel_expr"] = _train(
        cfg,
        "auto_rel_expr",
        train,
        valid,
        rel_cols + [c for c in derived_cols if c in train.columns],
        out_dir,
    )

    manual_train = pd.concat([load_candidates(cfg, n) for n in TRAIN_SNAPSHOTS], ignore_index=True)
    manual_val = load_candidates(cfg, "val")
    manual_cols = feature_columns(manual_train)
    join_cols = ["uid", "item_id", "snapshot"]
    rel_all = rel_cols + [c for c in derived_cols if c in train.columns]
    union_train = manual_train.merge(train[join_cols + rel_all], on=join_cols, how="left")
    union_val = manual_val.merge(valid[join_cols + rel_all], on=join_cols, how="left")
    results["manual_plus_expr"] = _train(
        cfg,
        "manual_plus_expr",
        union_train,
        union_val,
        manual_cols + rel_all,
        out_dir,
    )
    del train, valid, union_train, union_val, manual_train, manual_val

    best = max(results, key=lambda k: results[k]["metrics"]["val"]["ndcg@10"])
    test = build("test")
    if best == "manual_plus_expr":
        manual_test = load_candidates(cfg, "test")
        test_frame = manual_test.merge(test[join_cols + rel_all], on=join_cols, how="left")
        test_cols = manual_cols + rel_all
    else:
        test_frame = test
        test_cols = rel_cols if best == "auto_base_expr" else rel_all
    results[best]["metrics"]["test"] = evaluate(
        add_scores(test_frame, predict(results[best]["model"], test_frame[test_cols])),
        ks=tuple(cfg.eval.ks),
    )
    log.info("best=%s test: %s", best, results[best]["metrics"]["test"])

    payload = {
        "pool": args.pool,
        "grid_tag": args.grid_tag,
        "n_grid_base": len(base_pruned),
        "n_expressions": len(expr_cols),
        "n_derived": len(derived_cols),
        "n_manual": len(manual_cols),
        "best_variant": best,
        "chosen": [expr.name for expr in chosen],
        "results": {
            name: {
                "metrics": res["metrics"],
                "train_seconds": res["train_seconds"],
                "n_features": len(res["importance"]),
            }
            for name, res in results.items()
        },
    }
    save_json(payload, base / f"metrics_{args.tag}.json")
    log.info("saved %s", base / f"metrics_{args.tag}.json")


if __name__ == "__main__":
    main()
