"""CLI: базовые графики для отчёта."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys_auto.config import load_config, resolve
from recsys_auto.utils.io import ensure_dir, get_logger

log = get_logger("figures")


def main() -> None:
    cfg = load_config()
    fig_dir = ensure_dir(resolve(cfg.paths.reports) / "figures")
    events = pl.read_parquet(resolve(cfg.prepare.events_file))
    users = pd.read_parquet(resolve(cfg.prepare.users_file))
    items = pd.read_parquet(resolve(cfg.prepare.items_file))

    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(users["n_listens"].clip(lower=1), bins=60, log=True)
    axes[0].set(title="Прослушивания на пользователя", xlabel="n_listens", ylabel="users (log)")
    axes[1].hist(items["n_listens"].clip(lower=1), bins=60, log=True)
    axes[1].set(title="Прослушивания на трек", xlabel="n_listens", ylabel="items (log)")
    played = events.filter(pl.col("event_type") == "listen")["played_ratio"].to_numpy()
    axes[2].hist(played, bins=50)
    axes[2].set(title="Распределение played_ratio", xlabel="played ratio, %", ylabel="events")
    fig.tight_layout()
    fig.savefig(fig_dir / "data_distributions.png", dpi=140)
    plt.close(fig)

    importance_path = resolve(cfg.paths.experiments) / "baseline_manual" / "feature_importance.csv"
    if importance_path.exists():
        imp = pd.read_csv(importance_path).head(25).iloc[::-1]
        fig, ax = plt.subplots(figsize=(9, 8))
        ax.barh(imp["feature"], imp["gain"], color="#4c72b0")
        ax.set(title="Baseline: top-25 признаков по gain", xlabel="gain")
        fig.tight_layout()
        fig.savefig(fig_dir / "baseline_importance.png", dpi=140)
        plt.close(fig)
        log.info("saved feature importance figure")

    log.info("figures saved to %s", fig_dir)


if __name__ == "__main__":
    main()
