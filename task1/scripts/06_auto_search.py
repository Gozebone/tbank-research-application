"""CLI: автоматический поиск и отбор признаков.

Стадии можно запускать отдельно (результаты кешируются):

    python scripts/06_auto_search.py --stage screen
    python scripts/06_auto_search.py --stage materialize
    python scripts/06_auto_search.py --stage train
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys_auto.auto_features.derived import REL_PREFIX, derive_relations
from recsys_auto.auto_features.generator import generate_specs
from recsys_auto.auto_features.search import (
    load_candidates,
    materialize_specs,
    screen_specs,
    select_specs,
)
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

log = get_logger("auto_search")

TRAIN_SNAPSHOTS = ["train_w0", "train_w1", "train_w2", "train_w3"]
EVAL_SNAPSHOTS = ["val", "test"]


def _paths(cfg, tag: str = "topk") -> dict[str, Path]:
    base = ensure_dir(resolve(cfg.paths.experiments) / "auto_search")
    return {
        "base": base,
        "tag": tag,
        "screen": base / "screen.csv",
        "screen_tag": base / f"screen_{tag}.csv",
        "selection": base / f"selected_specs_{tag}.json",
        "features": ensure_dir(base / f"features_{tag}"),
        "derived": base / f"derived_selection_{tag}.json",
        "metrics": base / f"metrics_{tag}.json",
        "models": ensure_dir(base / tag / "models"),
    }


def _load_pool_specs(path: str | Path) -> list:
    """Прочитать пул признаков, предложенных LLM-агентом."""
    from recsys_auto.features.dsl import FeatureSpec

    payload = load_json(path)
    specs: list[FeatureSpec] = []
    for item in payload.get("specs", []):
        try:
            specs.append(
                FeatureSpec(
                    entity=item["entity"],
                    metric=item["metric"],
                    window_days=item.get("window_days"),
                    event_type=item.get("event_type"),
                )
            )
        except ValueError as exc:
            log.warning("skip invalid pool spec %s: %s", item.get("name"), exc)
    return specs


def stage_screen(
    cfg,
    top_k: int,
    sample_users: int,
    tag: str,
    per_entity: int | None,
    extra_pool: str | None = None,
) -> None:
    paths = _paths(cfg, tag)
    specs = generate_specs()
    log.info("generated %d candidate specs", len(specs))

    extra_specs: list = []
    if extra_pool:
        if not paths["screen"].exists():
            raise FileNotFoundError(
                f"{paths['screen']} not found; run --stage screen without --extra-pool first"
            )
        grid_screen = pd.read_csv(paths["screen"])
        extra_specs = _load_pool_specs(extra_pool)
        known = set(grid_screen["name"])
        fresh = [s for s in extra_specs if s.name not in known]
        log.info(
            "agent pool: %d specs, %d new (not in grid screen)", len(extra_specs), len(fresh)
        )
        extra_screen = (
            screen_specs(cfg, fresh, sample_users=sample_users, save=False)
            if fresh
            else pd.DataFrame(columns=grid_screen.columns)
        )
        screen = pd.concat([grid_screen, extra_screen], ignore_index=True)
        screen = screen.sort_values(
            ["strength", "auc"], ascending=[False, False]
        ).reset_index(drop=True)
        screen.to_csv(paths["screen_tag"], index=False)
        log.info("combined screen saved to %s (%d rows)", paths["screen_tag"], len(screen))
        all_specs = specs + extra_specs
    else:
        if paths["screen"].exists():
            log.info("reusing existing screen: %s", paths["screen"])
            screen = pd.read_csv(paths["screen"])
        else:
            screen = screen_specs(cfg, specs, sample_users=sample_users)
        all_specs = specs

    selected, companions = select_specs(
        screen, all_specs, top_k=top_k, per_entity=per_entity
    )
    extra_names = {s.name for s in extra_specs}
    n_from_extra = sum(1 for s in selected if s.name in extra_names)
    save_json(
        {
            "top_k": top_k,
            "per_entity": per_entity,
            "extra_pool": str(extra_pool) if extra_pool else None,
            "n_extra_specs": len(extra_specs),
            "n_selected_from_extra": n_from_extra,
            "screen_meta": load_json(paths["base"] / "screen_meta.json"),
            "base_specs": [s.describe() for s in selected],
            "companion_specs": [s.describe() for s in companions],
            "screen_top20": screen.head(20).to_dict(orient="records"),
        },
        paths["selection"],
    )
    log.info(
        "selected %d base + %d companion specs (%d from agent pool)",
        len(selected),
        len(companions),
        n_from_extra,
    )


def _selected_specs(cfg, paths):
    payload = load_json(paths["selection"])
    from recsys_auto.features.dsl import FeatureSpec

    base = [FeatureSpec(**{k: v for k, v in s.items() if k != "name"}) for s in payload["base_specs"]]
    companions = [
        FeatureSpec(**{k: v for k, v in s.items() if k != "name"})
        for s in payload["companion_specs"]
    ]
    return base, companions


def stage_materialize(cfg, force: bool, tag: str) -> None:
    paths = _paths(cfg, tag)
    base, companions = _selected_specs(cfg, paths)
    specs = base + companions
    log.info("materializing %d specs", len(specs))
    for snapshot in TRAIN_SNAPSHOTS + EVAL_SNAPSHOTS:
        out = paths["features"] / f"{snapshot}.parquet"
        if out.exists() and not force:
            log.info("skip %s (exists)", snapshot)
            continue
        materialize_specs(cfg, specs, snapshot, out)


def _load_auto_frame(
    cfg,
    paths,
    snapshot: str,
    base_cols: list[str],
    derived_cols: list[str] | None,
) -> pd.DataFrame:
    frame = pd.read_parquet(paths["features"] / f"{snapshot}.parquet")
    cutoff = int(frame["cutoff"].iloc[0])
    derived = derive_relations(frame, cutoff)
    raw_ts = [
        c
        for c in frame.columns
        if (parts := c.split("__")) and len(parts) == 4 and parts[1] in INDEX_ONLY_METRICS
    ]
    frame = frame.drop(columns=raw_ts)
    meta = [c for c in META_COLUMNS if c in frame.columns]
    out = frame[meta + [c for c in base_cols if c in frame.columns]].copy()
    if derived_cols:
        keep = [c for c in derived_cols if c in derived.columns]
        out = pd.concat([out, derived[keep]], axis=1)
    return out


def _build_variant_frames(cfg, paths, base_cols, derived_cols, snapshots):
    return {
        name: _load_auto_frame(cfg, paths, name, base_cols, derived_cols)
        for name in snapshots
    }


def _train_and_eval(
    cfg,
    name: str,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    cols: list[str],
    out_base: Path,
) -> dict:
    started = time.perf_counter()
    model = train_lgbm(
        train,
        valid,
        feature_cols=cols,
        params=dict(cfg.model.params),
        num_boost_rounds=int(cfg.model.num_boost_rounds),
        early_stopping_rounds=int(cfg.model.early_stopping_rounds),
    )
    train_seconds = time.perf_counter() - started
    out_dir = ensure_dir(out_base / name)
    save_model(model, out_dir)
    importance = feature_importance(model)
    importance.to_csv(out_dir / "feature_importance.csv", index=False)

    ks = tuple(int(k) for k in cfg.eval.ks)
    scored_val = add_scores(valid, predict(model, valid))
    metrics = {"val": evaluate(scored_val, ks=ks)}
    log.info("%s val: %s", name, metrics["val"])
    log.info("%s trained in %.1fs", name, train_seconds)
    return {
        "model": model,
        "metrics": metrics,
        "train_seconds": train_seconds,
        "importance": importance,
        "out_dir": out_dir,
    }


def _is_feature_column(name: str) -> bool:
    parts = name.split("__")
    return len(parts) == 4 and parts[1] not in INDEX_ONLY_METRICS


def _select_derived(raw, raw_sample, cutoff, derived_top_k, paths) -> list[str]:
    derived_sample = derive_relations(raw_sample, cutoff)
    derived_auc = univariate_auc(
        pd.concat([raw_sample[["label"]], derived_sample], axis=1),
        list(derived_sample.columns),
    )
    derived_auc = derived_auc[(derived_auc - 0.5).abs() > 0.01]
    ordered = list(
        derived_auc.reindex((derived_auc - 0.5).abs().sort_values(ascending=False).index).index
    )
    ordered = correlation_prune(derived_sample, ordered, threshold=0.98)
    derived_cols = ordered[:derived_top_k]
    save_json(
        {"n_generated": int(derived_sample.shape[1]), "selected": derived_cols},
        paths["derived"],
    )
    log.info("selected %d derived features out of %d", len(derived_cols), derived_sample.shape[1])
    return derived_cols


def stage_train(cfg, top_k: int, derived_top_k: int, tag: str, ablate: bool = True) -> None:
    paths = _paths(cfg, tag)
    base, companions = _selected_specs(cfg, paths)
    base_names = [s.name for s in base]

    # производные признаки строим на валидации; скрининг — на подвыборке
    raw = pd.read_parquet(paths["features"] / "val.parquet")
    cutoff = int(raw["cutoff"].iloc[0])
    sample_idx = np.random.default_rng(42).choice(
        len(raw), size=min(400_000, len(raw)), replace=False
    )
    raw_sample = raw.iloc[sample_idx].reset_index(drop=True)
    # убираем почти дублирующие базовые признаки (any/listen, соседние окна)
    base_available = [c for c in base_names if c in raw_sample.columns]
    base_pruned = correlation_prune(raw_sample, base_available, threshold=0.98)
    log.info("base features after correlation pruning: %d -> %d", len(base_available), len(base_pruned))

    derived_cols = _select_derived(raw, raw_sample, cutoff, derived_top_k, paths)
    del raw, raw_sample

    train_frames = _build_variant_frames(
        cfg, paths, base_pruned, derived_cols, TRAIN_SNAPSHOTS
    )
    val_frame = _load_auto_frame(cfg, paths, "val", base_pruned, derived_cols)
    train = pd.concat(train_frames.values(), ignore_index=True)
    del train_frames

    results: dict[str, dict] = {}
    out_base = paths["models"]

    # 1) только базовые агрегаты
    res = _train_and_eval(cfg, "auto_base", train, val_frame, base_pruned, out_base)
    results["auto_base"] = res

    # 2) абляции по gain
    if ablate:
        top_gain = res["importance"].head(max(50, top_k)).query("gain > 0")["feature"].tolist()
        for k in (25, 50):
            cols = [c for c in top_gain if c in train.columns][:k]
            if len(cols) != k:
                continue
            sub = _train_and_eval(cfg, f"auto_base_top{k}", train, val_frame, cols, out_base)
            results[f"auto_base_top{k}"] = sub

    # 3) базовые + производные
    rel_cols = [c for c in base_pruned + derived_cols if c in train.columns]
    res_rel = _train_and_eval(cfg, "auto_rel", train, val_frame, rel_cols, out_base)
    results["auto_rel"] = res_rel

    # 4) объединение с ручным набором
    manual_train = pd.concat(
        [load_candidates(cfg, n) for n in TRAIN_SNAPSHOTS], ignore_index=True
    )
    manual_val = load_candidates(cfg, "val")
    manual_cols = feature_columns(manual_train)
    union_train = manual_train.merge(
        train[["uid", "item_id", "snapshot"] + rel_cols],
        on=["uid", "item_id", "snapshot"],
        how="left",
        validate="one_to_one",
    )
    union_val = manual_val.merge(
        val_frame[["uid", "item_id", "snapshot"] + rel_cols],
        on=["uid", "item_id", "snapshot"],
        how="left",
        validate="one_to_one",
    )
    union_cols = manual_cols + rel_cols
    res_union = _train_and_eval(cfg, "manual_plus_auto", union_train, union_val, union_cols, out_base)
    results["manual_plus_auto"] = res_union
    del train, val_frame, union_train, union_val, manual_train, manual_val

    # 5) тест — только для лучшего по валидации варианта
    best_name = max(
        results,
        key=lambda k: results[k]["metrics"]["val"]["ndcg@10"],
    )
    best = results[best_name]
    if best_name == "manual_plus_auto":
        test_frame = load_candidates(cfg, "test")
        test_frame = test_frame.merge(
            _load_auto_frame(cfg, paths, "test", base_pruned, derived_cols)[
                ["uid", "item_id", "snapshot"] + rel_cols
            ],
            on=["uid", "item_id", "snapshot"],
            how="left",
            validate="one_to_one",
        )
    else:
        test_frame = _load_auto_frame(cfg, paths, "test", base_pruned, derived_cols)
    test_scored = add_scores(test_frame, predict(best["model"], test_frame))
    best["metrics"]["test"] = evaluate(test_scored, ks=tuple(int(k) for k in cfg.eval.ks))
    log.info("best=%s test: %s", best_name, best["metrics"]["test"])

    payload = {
        "top_k": top_k,
        "derived_top_k": derived_top_k,
        "n_base_features_selected": len(base_names),
        "n_base_features_pruned": len(base_pruned),
        "n_derived_features": len(derived_cols),
        "n_manual_features": len(manual_cols),
        "best_variant": best_name,
        "results": {
            name: {
                "metrics": res["metrics"],
                "train_seconds": res["train_seconds"],
                "n_features": len(res["importance"]),
            }
            for name, res in results.items()
        },
        "materialization": {
            s: load_json(paths["features"] / f"{s}.meta.json")["seconds"]
            for s in TRAIN_SNAPSHOTS + EVAL_SNAPSHOTS
        },
    }
    save_json(payload, paths["metrics"])
    log.info("saved metrics to %s", paths["metrics"])


def stage_train_all(cfg, tag: str, derived_top_k: int, gain_top_k: int = 40) -> None:
    """Обучить модель на всём материализованном пуле и отобрать по gain.

    В отличие от top-k/quota, здесь нет предварительного одномерного отбора:
    бустинг сам оценивает вклад признаков, после чего обучается компактная
    модель на top-gain. Это проверяет гипотезу, что проблема наивного отбора —
    отсутствие комплементарности.
    """
    paths = _paths(cfg, tag)
    raw = pd.read_parquet(paths["features"] / "val.parquet")
    all_cols = [c for c in raw.columns if _is_feature_column(c)]
    cutoff = int(raw["cutoff"].iloc[0])
    if paths["derived"].exists():
        derived_cols = load_json(paths["derived"])["selected"]
    else:
        idx = np.random.default_rng(42).choice(len(raw), size=min(400_000, len(raw)), replace=False)
        sample = raw.iloc[idx].reset_index(drop=True)
        derived_cols = _select_derived(raw, sample, cutoff, derived_top_k, paths)
        del sample
    del raw
    log.info("all-features pool: %d base + %d derived", len(all_cols), len(derived_cols))

    train_frames = _build_variant_frames(cfg, paths, all_cols, derived_cols, TRAIN_SNAPSHOTS)
    train = pd.concat(train_frames.values(), ignore_index=True)
    del train_frames
    val = _load_auto_frame(cfg, paths, "val", all_cols, derived_cols)
    cols = [c for c in all_cols + derived_cols if c in train.columns]

    res = _train_and_eval(cfg, "auto_all", train, val, cols, paths["models"])
    top_gain = res["importance"].head(gain_top_k).query("gain > 0")["feature"].tolist()
    res_top = _train_and_eval(
        cfg, f"auto_gain_top{gain_top_k}", train, val, top_gain, paths["models"]
    )
    del train, val

    metrics_path = paths["metrics"]
    payload = load_json(metrics_path) if metrics_path.exists() else {}
    payload.setdefault("results", {})
    for name, result in (("auto_all", res), (f"auto_gain_top{gain_top_k}", res_top)):
        payload["results"][name] = {
            "metrics": result["metrics"],
            "train_seconds": result["train_seconds"],
            "n_features": len(result["importance"]),
        }
    best = max(
        payload["results"],
        key=lambda k: payload["results"][k]["metrics"]["val"]["ndcg@10"],
    )
    payload["best_variant"] = best
    new_names = {"auto_all", f"auto_gain_top{gain_top_k}"}
    if best in new_names and "test" not in payload["results"][best]["metrics"]:
        model = {("auto_all"): res, f"auto_gain_top{gain_top_k}": res_top}[best]["model"]
        test = _load_auto_frame(cfg, paths, "test", all_cols, derived_cols)
        payload["results"][best]["metrics"]["test"] = evaluate(
            add_scores(test, predict(model, test)),
            ks=tuple(int(k) for k in cfg.eval.ks),
        )
        log.info("%s test: %s", best, payload["results"][best]["metrics"]["test"])
    payload["all_features"] = {"n_base": len(all_cols), "n_derived": len(derived_cols)}
    save_json(payload, metrics_path)
    log.info("updated metrics: %s", metrics_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--stage", default="all", choices=["screen", "materialize", "train", "all-features", "all"])
    parser.add_argument("--top-k", type=int, default=80)
    parser.add_argument("--per-entity", type=int, default=None, help="квота признаков на сущность")
    parser.add_argument("--derived-top-k", type=int, default=60)
    parser.add_argument("--sample-users", type=int, default=1_000)
    parser.add_argument("--tag", default="topk", help="имя эксперимента (вариант отбора)")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-ablate", action="store_true", help="не обучать top-k абляции")
    parser.add_argument(
        "--extra-pool",
        default=None,
        help="json с признаками LLM-агента: добавить их к сеточным кандидатам",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.stage in ("screen", "all"):
        stage_screen(
            cfg,
            args.top_k,
            args.sample_users,
            args.tag,
            args.per_entity,
            extra_pool=args.extra_pool,
        )
    if args.stage in ("materialize", "all"):
        stage_materialize(cfg, args.force, args.tag)
    if args.stage in ("train", "all"):
        stage_train(cfg, args.top_k, args.derived_top_k, args.tag, ablate=not args.no_ablate)
    if args.stage in ("all-features", "all"):
        stage_train_all(cfg, args.tag, args.derived_top_k)


if __name__ == "__main__":
    main()
