"""CLI: сравнение пулов, предложенных разными LLM.

Ключевой вопрос: если агент проигрывает сетке, это свойство модели или
протокола отбора? Один и тот же харнесс прогоняется несколькими моделями, а
все пулы скрининятся вместе на полной выборке (1000 пользователей). Считается,
сколько признаков каждой модели попало бы в квоту 20 на сущность — это и есть
«выживаемость» без переобучения модели.

    python scripts/12_compare_agents.py --pools quota,ab_deepseek,ab_kimi
    python scripts/12_compare_agents.py --pools expr_qwen,expr_deepseek,expr_kimi \\
        --out agent_model_ab_expr.csv
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys_auto.auto_features.expr import expression_from_mapping, spec_from_mapping
from recsys_auto.auto_features.expr_search import ExpressionScreener
from recsys_auto.auto_features.search import screen_specs
from recsys_auto.config import load_config, resolve
from recsys_auto.features.dsl import INDEX_ONLY_METRICS, FeatureSpec
from recsys_auto.utils.io import ensure_dir, get_logger, load_json, save_json

log = get_logger("agent_ab")


def _load_pool(path: Path) -> tuple[list, str]:
    payload = load_json(path)
    if payload.get("expressions"):
        exprs = [expression_from_mapping(item["expr"]) for item in payload["expressions"]]
        return exprs, "expr"
    specs = []
    for item in payload.get("specs", []):
        specs.append(
            spec_from_mapping(
                {
                    "entity": item["entity"],
                    "metric": item["metric"],
                    "event_type": item.get("event_type"),
                    "window_days": item.get("window_days"),
                    "condition": item.get("condition"),
                    "decay_days": item.get("decay_days"),
                }
            )
        )
    return specs, "flat"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pools", default="quota")
    parser.add_argument("--sample-users", type=int, default=1000)
    parser.add_argument("--quota", type=int, default=20)
    parser.add_argument("--out", default="agent_model_ab.csv")
    parser.add_argument("--force", action="store_true", help="перескрининить заново")
    args = parser.parse_args()

    cfg = load_config()
    base = resolve(cfg.paths.experiments) / "auto_search"
    tags = [t.strip() for t in args.pools.split(",") if t.strip()]

    pools: dict[str, set[str]] = {}
    all_exprs: dict[str, object] = {}
    kinds: set[str] = set()
    for tag in tags:
        path = base / f"agent_pool_{tag}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        exprs, kind = _load_pool(path)
        pools[tag] = {expr.name for expr in exprs}
        kinds.add(kind)
        for expr in exprs:
            all_exprs.setdefault(expr.name, expr)

    log.info(
        "screening %d unique features from %d pools (%s) on %d users",
        len(all_exprs),
        len(tags),
        "/".join(sorted(kinds)),
        args.sample_users,
    )
    cache_path = base / f"agent_ab_screen_{'_'.join(sorted(kinds))}.csv"
    frame = None
    if cache_path.exists() and not args.force:
        cached = pd.read_csv(cache_path)
        if set(all_exprs) <= set(cached["name"]):
            log.info("reusing cached screen: %s", cache_path)
            frame = cached[cached["name"].isin(all_exprs)].copy()
    if frame is None:
        if kinds == {"expr"}:
            screener = ExpressionScreener(cfg, sample_users=args.sample_users)
            try:
                frame = screener.score(list(all_exprs.values()))
            finally:
                screener.close()
        elif kinds == {"flat"}:
            frame = screen_specs(
                cfg, list(all_exprs.values()), sample_users=args.sample_users, save=False
            )
        else:
            raise ValueError("mix of flat and expression pools is not supported")
        frame.drop(columns=[c for c in ("strength", "rank_entity") if c in frame.columns]).to_csv(
            cache_path, index=False
        )
    frame["strength"] = (frame["user_auc"] - 0.5).abs()

    # Сравниваем каждый пул с одной и той же сеткой, а не с другими пулами.
    grid_path = base / "screen.csv"
    if not grid_path.exists():
        raise FileNotFoundError(f"{grid_path} not found; run stage screen first")
    grid = pd.read_csv(grid_path)
    grid = grid[
        ~grid["metric"].isin(INDEX_ONLY_METRICS) & (grid["nan_rate"] < 0.999)
    ].copy()
    grid["strength"] = (grid["user_auc"] - 0.5).abs()
    cutoff = grid.groupby("entity")["strength"].apply(
        lambda values: values.nlargest(args.quota).min()
    )

    rows = []
    for tag in tags:
        names = pools[tag]
        sub = frame[frame["name"].isin(names)].copy()
        if not len(sub):
            continue
        combined = pd.concat(
            [grid[["name", "entity", "strength"]], sub[["name", "entity", "strength"]]],
            ignore_index=True,
        ).drop_duplicates("name", keep="last")
        combined["rank_entity"] = (
            combined.groupby("entity")["strength"]
            .rank(ascending=False, method="min")
            .astype(int)
        )
        sub = sub.merge(combined[["name", "rank_entity"]], on="name", how="left")
        in_quota = sub["strength"] >= sub["entity"].map(cutoff)
        transcript_path = base / f"agent_transcript_{tag}.json"
        stats = (
            load_json(transcript_path).get("stats", {}) if transcript_path.exists() else {}
        )
        rows.append(
            {
                "pool": tag,
                "kind": "expr" if kinds == {"expr"} else "flat",
                "n_specs": len(sub),
                "n_in_quota": int(in_quota.sum()),
                "share_in_quota": round(float(in_quota.mean()), 3),
                "median_user_auc": round(float(sub["user_auc"].median()), 4),
                "max_user_auc": round(float(sub["user_auc"].max()), 4),
                "best_entity_rank": int(sub["rank_entity"].min()),
                "n_entities": int(sub["entity"].nunique()),
                "n_ops": int(sub["op"].nunique()) if "op" in sub else 1,
                "cost_usd": stats.get("cost_usd"),
                "seconds": stats.get("seconds"),
            }
        )
    table = pd.DataFrame(rows)
    tables_dir = ensure_dir(resolve(cfg.paths.reports) / "tables")
    table.to_csv(tables_dir / args.out, index=False)

    overlap = {}
    for left, right in combinations(tags, 2):
        inter = pools[left] & pools[right]
        union = pools[left] | pools[right]
        overlap[f"{left}|{right}"] = round(len(inter) / len(union), 3) if union else 0.0

    save_json(
        {
            "sample_users": args.sample_users,
            "quota_per_entity": args.quota,
            "table": table.to_dict(orient="records"),
            "jaccard": overlap,
        },
        tables_dir / args.out.replace(".csv", ".json"),
    )
    log.info("wrote %s", tables_dir / args.out)
    print(table.to_string(index=False))
    print("\nJaccard overlap between pools:", overlap)


if __name__ == "__main__":
    main()
