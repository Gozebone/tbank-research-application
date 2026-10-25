"""Конфигурация проекта.

Читает YAML и отдаёт вложенные словари с доступом по атрибутам, а также
разрешает пути относительно корня проекта.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "base.yaml"


class Config(dict):
    """dict с доступом через атрибуты."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:  # pragma: no cover - защита от опечаток
            raise AttributeError(item) from exc
        return Config(value) if isinstance(value, dict) else value


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path is not None else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    cfg = Config(raw)
    cfg["_path"] = str(path)
    return cfg


def resolve(path: str | Path) -> Path:
    """Путь относительно корня проекта (абсолютные пути не трогаем)."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path
