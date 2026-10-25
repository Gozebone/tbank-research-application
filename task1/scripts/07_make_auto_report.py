"""CLI: таблицы и графики по результатам автоматического поиска."""

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
from recsys_auto.features.dsl import INDEX_ONLY_METRICS
from recsys_auto.features.manual import manual_specs
from recsys_auto.utils.io import ensure_dir, get_logger, load_json, save_json

log = get_logger("auto_report")


def build_tables(cfg, base: Path, tag: str) -> pd.DataFrame:
    metrics = load_json(base / f"metrics_{tag}.json")
    selection = load_json(base / f"selected_specs_{tag}.json")

    rows = []
    for name, payload in metrics["results"].items():
        row = {"variant": name, **payload["metrics"]["val"]}
        row["n_features"] = payload["n_features"]
        row["train_seconds"] = round(payload["train_seconds"], 1)
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("ndcg@10", ascending=False)
    report_dir = ensure_dir(resolve(cfg.paths.reports) / "tables")
    table.to_csv(report_dir / f"auto_variants_{tag}.csv", index=False)

    manual_names = {
        s.name for s in manual_specs() if s.metric not in INDEX_ONLY_METRICS
    }
    auto_names = {s["name"] for s in selection["base_specs"]}
    overlap = sorted(auto_names & manual_names)

    screen = pd.read_csv(base / "screen.csv")
    screen["rank"] = np.arange(1, len(screen) + 1)
    manual_screen = screen[screen["name"].isin(manual_names)].copy()
    manual_screen["percentile"] = 1 - manual_screen["rank"] / len(screen)
    save_json(
        {
            "tag": tag,
            "n_manual_base": len(manual_names),
            "n_auto": len(auto_names),
            "n_overlap": len(overlap),
            "jaccard": len(overlap) / len(auto_names | manual_names),
            "auto_selected_from_manual": len(overlap) / len(auto_names),
            "manual_mean_percentile_in_screen": float(manual_screen["percentile"].mean()),
            "manual_median_rank": float(manual_screen["rank"].median()),
            "n_screen": len(screen),
            "only_auto": sorted(auto_names - manual_names),
            "overlap": overlap,
            "manual_specs_in_screen": manual_screen.sort_values("rank")[
                ["name", "rank", "user_auc", "auc"]
            ].to_dict(orient="records"),
        },
        base / f"feature_overlap_{tag}.json",
    )
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="topk")
    args = parser.parse_args()

    cfg = load_config()
    base = resolve(cfg.paths.experiments) / "auto_search"
    tag = args.tag
    table = build_tables(cfg, base, tag)

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.barh(table["variant"], table["ndcg@10"], color="#4c72b0")
    ax.set(xlabel="NDCG@10")
    fig.tight_layout()
    figures = ensure_dir(resolve(cfg.paths.reports) / "figures")
    fig.savefig(figures / f"auto_variants_{tag}.png", dpi=140)
    plt.close(fig)

    importance_path = base / tag / "models" / "auto_rel" / "feature_importance.csv"
    if importance_path.exists():
        imp = pd.read_csv(importance_path).head(25).iloc[::-1]
        fig, ax = plt.subplots(figsize=(9, 8))
        ax.barh(imp["feature"], imp["gain"], color="#55a868")
        ax.set(title=f"auto_rel ({tag}): top-25 признаков по gain", xlabel="gain")
        fig.tight_layout()
        fig.savefig(figures / f"auto_importance_{tag}.png", dpi=140)
        plt.close(fig)

    log.info("auto report written; best variant: %s", table.iloc[0]["variant"])


if __name__ == "__main__":
    main()
