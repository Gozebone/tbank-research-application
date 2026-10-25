"""CLI: таблица и график «плоский DSL vs расширенный DSL».

Сравнивает метрики сеточного запуска (тег ``quota``) и запуска с выражениями
агента (тег ``expr_add``) на одинаковых train/val/test.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys_auto.config import load_config, resolve
from recsys_auto.utils.io import ensure_dir, get_logger, load_json

log = get_logger("expr_report")

PAIRS = {
    "auto_base": "auto_base_expr",
    "auto_rel": "auto_rel_expr",
    "manual_plus_auto": "manual_plus_expr",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flat-tag", default="quota")
    parser.add_argument("--expr-tag", default="expr_add")
    args = parser.parse_args()

    cfg = load_config()
    base = resolve(cfg.paths.experiments) / "auto_search"
    flat = load_json(base / f"metrics_{args.flat_tag}.json")["results"]
    expr = load_json(base / f"metrics_{args.expr_tag}.json")["results"]

    rows = []
    for flat_name, expr_name in PAIRS.items():
        if flat_name not in flat or expr_name not in expr:
            continue
        f_val = flat[flat_name]["metrics"]["val"]
        e_val = expr[expr_name]["metrics"]["val"]
        f_test = flat[flat_name]["metrics"].get("test", {})
        e_test = expr[expr_name]["metrics"].get("test", {})
        rows.append(
            {
                "variant": flat_name,
                "n_features_flat": flat[flat_name]["n_features"],
                "n_features_expr": expr[expr_name]["n_features"],
                "val_flat": round(f_val["ndcg@10"], 4),
                "val_expr": round(e_val["ndcg@10"], 4),
                "delta_val": round(e_val["ndcg@10"] - f_val["ndcg@10"], 4),
                "test_flat": round(f_test["ndcg@10"], 4) if f_test else None,
                "test_expr": round(e_test["ndcg@10"], 4) if e_test else None,
                "delta_test": (
                    round(e_test["ndcg@10"] - f_test["ndcg@10"], 4) if f_test and e_test else None
                ),
            }
        )
    table = pd.DataFrame(rows)
    tables = ensure_dir(resolve(cfg.paths.reports) / "tables")
    table.to_csv(tables / "expr_variants.csv", index=False)
    log.info("wrote %s", tables / "expr_variants.csv")
    print(table.to_string(index=False))

    if len(table) < 2:
        log.warning("not enough variants for the figure")
        return
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    import numpy as np

    xs = np.arange(len(table))
    width = 0.36
    ax.bar(xs - width / 2, table["val_flat"], width, label="plain DSL", color="#9aa5b1")
    ax.bar(xs + width / 2, table["val_expr"], width, label="extended DSL", color="#2f6fd0")
    for x, value in zip(xs - width / 2, table["val_flat"]):
        ax.text(x, value + 0.001, f"{value:.4f}", ha="center", va="bottom", fontsize=7, rotation=90)
    for x, value in zip(xs + width / 2, table["val_expr"]):
        ax.text(x, value + 0.001, f"{value:.4f}", ha="center", va="bottom", fontsize=7, rotation=90)
    low, high = table[["val_flat", "val_expr"]].min().min(), table[["val_flat", "val_expr"]].max().max()
    ax.set_ylim(low - 0.01, high + 0.018)
    ax.set_xticks(xs)
    ax.set_xticklabels(table["variant"])
    ax.set_ylabel("NDCG@10 (val)")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    figures = ensure_dir(resolve(cfg.paths.reports) / "figures")
    fig.savefig(figures / "expr_effect.png", dpi=150)
    plt.close(fig)
    log.info("wrote %s", figures / "expr_effect.png")


if __name__ == "__main__":
    main()
