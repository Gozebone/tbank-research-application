"""CLI: нормализовать данные и сэмплировать пользователей."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys_auto.config import load_config
from recsys_auto.data.prepare import prepare_dataset
from recsys_auto.utils.io import get_logger

log = get_logger("prepare")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = prepare_dataset(cfg, force=args.force)
    for name, path in out.items():
        log.info("%s -> %s", name, path)


if __name__ == "__main__":
    main()
