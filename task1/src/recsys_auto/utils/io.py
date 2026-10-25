"""Мелкие утилиты ввода-вывода."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd


def get_logger(name: str = "recsys_auto") -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(payload: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)


def load_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def save_table(df: pd.DataFrame, path: str | Path, index: bool = False) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=index)
    elif path.suffix == ".csv":
        df.to_csv(path, index=index)
    else:  # pragma: no cover
        raise ValueError(f"unsupported table format: {path.suffix}")


def load_table(path: str | Path, **kwargs) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path, **kwargs)
    if path.suffix == ".csv":
        return pd.read_csv(path, **kwargs)
    raise ValueError(f"unsupported table format: {path.suffix}")
