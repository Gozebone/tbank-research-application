"""CLI: сводная таблица и график «первый сэмпл vs второй сэмпл»."""

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

from recsys_auto.config import load_config, resolve
from recsys_auto.utils.io import ensure_dir, get_logger, load_json

log = get_logger("holdout_report")

APPROACHES = {
    "baseline": "manual features",
    "quota_union": "manual + grid",
    "expr_union": "manual + grid + expressions",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="reports")
    args = parser.parse_args()

    cfg = load_config()
    auto = resolve(cfg.paths.experiments) / "auto_search"
    out = ensure_dir(resolve(args.out_dir))
    tables = ensure_dir(out / "tables")
    figures = ensure_dir(out / "figures")

    baseline = load_json(resolve(cfg.paths.experiments) / "baseline_manual" / "metrics.json")["metrics"]
    quota = load_json(auto / "metrics_quota.json")["results"]["manual_plus_auto"]["metrics"]
    expr = load_json(auto / "metrics_expr_add.json")["results"]["manual_plus_expr"]["metrics"]
    original = {
        "baseline": {"val": baseline["val"]["ndcg@10"], "test": baseline["test"]["ndcg@10"]},
        "quota_union": {"val": quota["val"]["ndcg@10"], "test": quota["test"]["ndcg@10"]},
        "expr_union": {"val": expr["val"]["ndcg@10"], "test": expr["test"]["ndcg@10"]},
    }

    holdout_metrics = pd.read_csv(tables / "holdout_metrics.csv").set_index("approach")
    rows = []
    for name, title in APPROACHES.items():
        rows.append(
            {
                "approach": name,
                "description": title,
                "n_features": int(holdout_metrics.loc[name, "n_features"]),
                "original_val": round(original[name]["val"], 4),
                "original_test": round(original[name]["test"], 4),
                "holdout_val": round(float(holdout_metrics.loc[name, "val_ndcg@10"]), 4),
                "holdout_test": round(float(holdout_metrics.loc[name, "test_ndcg@10"]), 4),
            }
        )
    table = pd.DataFrame(rows)
    table["delta_holdout_vs_baseline"] = (
        table["holdout_test"] - table.loc[table["approach"] == "baseline", "holdout_test"].iloc[0]
    ).round(4)
    table.to_csv(tables / "holdout_comparison.csv", index=False)
    print(table.to_string(index=False))

    # дельты с CI на двух сэмплах
    original_sig = pd.read_csv(tables / "significance.csv").set_index("comparison")
    holdout_sig = pd.read_csv(tables / "holdout_significance.csv").set_index("comparison")
    pairs = [
        ("quota_union - baseline", "manual + grid"),
        ("expr_union - baseline", "manual + grid + expr"),
    ]
    plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10.5, "axes.labelsize": 9.5})
    fig, axes = plt.subplots(2, 1, figsize=(6.2, 5.8))

    ax = axes[0]
    xs = np.arange(len(table))
    width = 0.36
    ax.bar(xs - width / 2, table["original_test"], width, label="sample 1 (10k)", color="#9aa5b1")
    ax.bar(xs + width / 2, table["holdout_test"], width, label="sample 2 (3k)", color="#2f6fd0")
    for x, value in zip(xs - width / 2, table["original_test"]):
        ax.text(x, value + 0.0008, f"{value:.4f}", ha="center", va="bottom", fontsize=7, rotation=90)
    for x, value in zip(xs + width / 2, table["holdout_test"]):
        ax.text(x, value + 0.0008, f"{value:.4f}", ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_xticks(xs)
    ax.set_xticklabels(["baseline", "grid", "grid+expr"], fontsize=9)
    ax.set_ylabel("NDCG@10 (test)")
    low = min(table["original_test"].min(), table["holdout_test"].min())
    high = max(table["original_test"].max(), table["holdout_test"].max())
    ax.set_ylim(low - 0.006, high + 0.004)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    labels = [label for _, label in pairs]
    xs = np.arange(len(pairs))
    for index, (source, name, color) in enumerate(
        [(original_sig, "sample 1 (10k)", "#9aa5b1"), (holdout_sig, "sample 2 (3k)", "#2f6fd0")]
    ):
        values, lows, highs = [], [], []
        for key, _ in pairs:
            row = source.loc[key]
            values.append(float(row["delta"]))
            lows.append(float(row["delta"]) - float(row["ci_low"]))
            highs.append(float(row["ci_high"]) - float(row["delta"]))
        offsets = xs - 0.16 + index * 0.32
        ax.bar(offsets, values, 0.3, label=name, color=color)
        ax.errorbar(offsets, values, yerr=[lows, highs], fmt="none", ecolor="#333", capsize=3)
        for offset, value in zip(offsets, values):
            ax.text(offset, value + 0.0008, f"{value:+.4f}", ha="center", fontsize=7)
    ax.axhline(0, color="#333", linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("ΔNDCG@10 к baseline")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(figures / "holdout_effect.png", dpi=150)
    plt.close(fig)
    log.info("wrote %s", figures / "holdout_effect.png")


if __name__ == "__main__":
    main()
