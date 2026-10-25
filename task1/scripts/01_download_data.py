"""CLI: скачать все датасеты."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys_auto.config import load_config
from recsys_auto.data.download import download_kion, download_yambda, extract_kion, free_space_gb
from recsys_auto.utils.io import get_logger

log = get_logger("download")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--skip-kion", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    log.info("free space: %.1f GB", free_space_gb(cfg.paths.raw))

    yambda = download_yambda(cfg)
    log.info("yambda files: %d", len(yambda))
    if not args.skip_kion:
        kion = download_kion(cfg)
        extract_kion(cfg)
        log.info("kion files: %d", len(kion))


if __name__ == "__main__":
    main()
