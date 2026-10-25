"""CLI: сводка по KION — перенос признаков и повторный поиск.

Сравниваются четыре варианта на одном test:
1. baseline (99 ручных признаков, обучен на KION);
2. перенос сеточного набора, отобранного на Yambda (203 признака);
3. перенос сетки + выражений (223);
4. повторный автоматический поиск на KION (146 признаков).

Для каждого считается NDCG@10 и paired bootstrap к baseline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys_auto.auto_features.derived import derive_relations
from recsys_auto.auto_features.search import load_candidates
from recsys_auto.config import load_config, resolve
from recsys_auto.eval.metrics import evaluate
from recsys_auto.features.dsl import INDEX_ONLY_METRICS
from recsys_auto.features.pipeline import feature_columns
from recsys_auto.models.boosting import TrainedModel, add_scores, predict
from recsys_auto.utils.io import ensure_dir, get_logger, load_json, save_json

log = get_logger("kion_report")
KEYS = ["uid", "item_id", "snapshot"]


def _model(path: Path) -> TrainedModel:
    import lightgbm as lgb

    meta = load_json(path / "model.meta.json")
    return TrainedModel(
        booster=lgb.Booster(model_file=str(path / "model.txt")),
        feature_cols=meta["feature_cols"],
        best_iteration=meta["best_iteration"],
    )


def _drop_index_only(frame: pd.DataFrame) -> pd.DataFrame:
    raw = [
        c
        for c in frame.columns
        if (parts := c.split("__")) and len(parts) == 4 and parts[1] in INDEX_ONLY_METRICS
    ]
    return frame.drop(columns=raw)


def _frame(snapshot: str, cfg, feature_cols: list[str], mode: str) -> pd.DataFrame:
    manual = load_candidates(cfg, snapshot)
    manual_cols = set(feature_columns(manual))
    needed = set(feature_cols) - manual_cols
    base = resolve(cfg.paths.experiments) / "auto_search"
    keys = KEYS
    frame = manual[keys + ["label"] + [c for c in manual_cols if c in feature_cols]].copy()

    if mode == "local":
        raw = pd.read_parquet(base / "features_kion" / f"{snapshot}.parquet")
    else:
        raw = pd.read_parquet(base / "features_quota" / f"{snapshot}.parquet")
    cutoff = int(raw["cutoff"].iloc[0])
    derived = derive_relations(raw, cutoff)
    raw = _drop_index_only(raw)
    cols = [c for c in needed if c in raw.columns]
    frame = frame.merge(raw[keys + cols], on=keys, how="left", validate="one_to_one")
    rel = [c for c in needed if c in derived.columns]
    if rel:
        frame = pd.concat([frame, derived[rel]], axis=1)
    if mode == "transfer_expr":
        expr = pd.read_parquet(base / "expr_features_expr_add" / f"{snapshot}.parquet")
        cols = [c for c in needed if c in expr.columns]
        frame = frame.merge(expr[keys + cols], on=keys, how="left", validate="one_to_one")
    missing = [c for c in feature_cols if c not in frame.columns]
    if missing:
        raise RuntimeError(f"{snapshot}/{mode}: missing {len(missing)}: {missing[:3]}")
    frame[feature_cols] = frame[feature_cols].astype("float32")
    return frame


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
    return pd.Series(out)


def _bootstrap(a: pd.Series, b: pd.Series, n: int = 2000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(a))
    deltas = np.empty(n)
    for i in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        deltas[i] = b.iloc[s].mean() - a.iloc[s].mean()
    return {
        "delta": float(b.mean() - a.mean()),
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
        "p_not_positive": float((deltas <= 0).mean()),
        "n_users": int(len(a)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/kion.yaml")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()

    cfg = load_config(args.config)
    base = resolve(cfg.paths.experiments) / "auto_search"
    specs = {
        "baseline": (base / "models" / "baseline", "local"),
        "transfer_grid": (base / "models" / "quota_union", "transfer"),
        "transfer_expr": (base / "models" / "expr_union", "transfer_expr"),
    }
    # kion_search снят с публикации: его features/model были обучены на старом
    # наборе кандидатов. Перенос сетки и выражений пересчитаны на выровненном
    # наборе; локальный поиск на KION требует отдельного переобучения.
    per_user: dict[str, pd.Series] = {}
    rows = []
    for name, (path, mode) in specs.items():
        if not path.exists():
            log.warning("skip %s: %s", name, path)
            continue
        model = _model(path)
        frame = _frame("test", cfg, model.feature_cols, mode)
        scored = add_scores(frame, predict(model, frame))
        metrics = evaluate(scored, ks=tuple(cfg.eval.ks))
        per_user[name] = _per_user_ndcg(scored)
        rows.append(
            {
                "approach": name,
                "n_features": len(model.feature_cols),
                "test_ndcg@10": round(metrics["ndcg@10"], 4),
                "test_roc_auc": round(metrics["roc_auc"], 4),
                "test_recall@10": round(metrics["recall@10"], 4),
            }
        )
        log.info("%s: test ndcg@10=%.4f", name, metrics["ndcg@10"])
        del frame, scored

    common = set.intersection(*(set(s.index) for s in per_user.values()))
    per_user = {k: v.loc[sorted(common)] for k, v in per_user.items()}
    significance = []
    for name in per_user:
        if name == "baseline":
            continue
        significance.append({"comparison": f"{name} - baseline", **_bootstrap(per_user["baseline"], per_user[name], args.bootstrap)})

    tables = ensure_dir(resolve("reports") / "tables")
    pd.DataFrame(rows).to_csv(tables / "kion_comparison.csv", index=False)
    pd.DataFrame(significance).to_csv(tables / "kion_bootstrap.csv", index=False)
    save_json(
        {"metrics": rows, "significance": significance, "n_users": len(common)},
        tables / "kion_summary.json",
    )

    order = [r["approach"] for r in rows]
    values = [r["test_ndcg@10"] for r in rows]
    plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10.5, "axes.labelsize": 9.5})
    fig, axes = plt.subplots(2, 1, figsize=(6.2, 5.6))
    ax = axes[0]
    xs = np.arange(len(order))
    ax.bar(xs, values, 0.55, color=["#9aa5b1", "#2f6fd0", "#2f6fd0", "#e08a1e"])
    for x, v in zip(xs, values):
        ax.text(x, v + 0.0015, f"{v:.4f}", ha="center", fontsize=8)
    short = {
        "baseline": "baseline",
        "transfer_grid": "transfer\ngrid",
        "transfer_expr": "transfer\ngrid+expr",
        "kion_search": "KION\nsearch",
    }
    ax.set_xticks(xs)
    ax.set_xticklabels([short.get(name, name) for name in order], fontsize=8)
    ax.set_ylabel("NDCG@10 (test)")
    ax.set_ylim(min(values) - 0.006, max(values) + 0.006)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    labels, deltas, lows, highs = [], [], [], []
    for row in significance:
        labels.append(row["comparison"].replace(" - baseline", ""))
        deltas.append(row["delta"])
        lows.append(row["delta"] - row["ci_low"])
        highs.append(row["ci_high"] - row["delta"])
    xs = np.arange(len(labels))
    ax.bar(xs, deltas, 0.5, color="#2f6fd0")
    ax.errorbar(xs, deltas, yerr=[lows, highs], fmt="none", ecolor="#333", capsize=4)
    for x, v in zip(xs, deltas):
        ax.text(x, v + 0.0004, f"{v:+.4f}", ha="center", fontsize=8)
    ax.axhline(0, color="#333", linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("ΔNDCG@10 к baseline")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    figures = ensure_dir(resolve("reports") / "figures")
    fig.savefig(figures / "kion_effect.png", dpi=150)
    plt.close(fig)
    log.info("wrote kion artifacts")
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(significance).to_string(index=False))


if __name__ == "__main__":
    main()
