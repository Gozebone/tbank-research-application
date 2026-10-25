"""Отпечаток подготовленных данных.

Нужен, чтобы нельзя было обучить модель на датасете, собранном из другой
выборки событий (например, после повторного запуска подготовки).
"""

from __future__ import annotations

from ..config import Config, resolve
from .io import load_json


def dataset_fingerprint(cfg: Config) -> str:
    stats = load_json(resolve(cfg.paths.interim) / "stats.json")
    return "_".join(
        str(stats.get(key, ""))
        for key in ("n_events", "n_users", "ts_max", "uid_hash")
    )
