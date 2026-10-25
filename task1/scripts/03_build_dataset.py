"""CLI: построить датасеты (кандидаты + признаки) для снимков."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys_auto.config import load_config, resolve
from recsys_auto.data.split import make_time_split
from recsys_auto.features.pipeline import build_snapshot_dataset, save_dataset
from recsys_auto.utils.io import ensure_dir, get_logger, save_json

log = get_logger("build_dataset")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="all", choices=["all", "train", "val", "test"])
    parser.add_argument("--snapshot", default=None, help="конкретный снимок, например train_w0")
    args = parser.parse_args()

    cfg = load_config(args.config)
    events = pl.scan_parquet(resolve(cfg.prepare.events_file))
    split = make_time_split(
        events,
        test_days=int(cfg.split.test_days),
        val_days=int(cfg.split.val_days),
        gap_days=int(cfg.split.gap_days),
    )
    out_dir = ensure_dir(resolve(cfg.paths.processed))
    save_json(split.__dict__, out_dir / "split.json")
    snapshots = split.snapshots(
        train_weeks=int(cfg.split.train_snapshot_weeks),
        horizon_days=int(cfg.split.label_horizon_days),
    )
    selected = [
        s
        for s in snapshots
        if (args.snapshot is None or s.name == args.snapshot)
        and (args.split == "all" or s.split == args.split)
    ]
    if not selected:
        raise SystemExit("no snapshots selected")

    log.info("time split: %s", split)
    for snapshot in selected:
        log.info("=== snapshot %s: cutoff=%d labels=[%d, %d)", snapshot.name, snapshot.cutoff, snapshot.label_start, snapshot.label_end)
        bundle = build_snapshot_dataset(cfg, snapshot)
        save_dataset(cfg, bundle, snapshot)


if __name__ == "__main__":
    main()
