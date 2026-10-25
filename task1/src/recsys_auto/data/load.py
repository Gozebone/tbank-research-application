"""Загрузка и нормализация событий."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from ..utils.time import YAMBDA_TS_UNIT_SECONDS

EVENT_COLUMNS = [
    "uid",
    "item_id",
    "ts",
    "event_type",
    "is_organic",
    "played_ratio",
    "track_length_seconds",
]


def scan_yambda(path: str | Path) -> pl.LazyFrame:
    """Прочитать Yambda multi_event и привести к внутренней схеме.

    Внутренняя схема: uid:uint32, item_id:uint32, ts:int64 (секунды),
    event_type:str, is_organic:uint8, played_ratio:uint16, track_length_seconds:uint32.
    """
    lf = pl.scan_parquet(path)
    return lf.select(
        pl.col("uid").cast(pl.UInt32),
        pl.col("item_id").cast(pl.UInt32),
        (pl.col("timestamp").cast(pl.Int64) * YAMBDA_TS_UNIT_SECONDS).alias("ts"),
        pl.col("event_type").cast(pl.Utf8),
        pl.col("is_organic").cast(pl.UInt8),
        pl.col("played_ratio_pct").cast(pl.UInt16).alias("played_ratio"),
        pl.col("track_length_seconds").cast(pl.UInt32),
    )


def sample_users(
    lf: pl.LazyFrame,
    n_users: int,
    min_events: int,
    seed: int,
    exclude: np.ndarray | None = None,
) -> tuple[pl.LazyFrame, pl.DataFrame]:
    """Сэмплировать пользователей и сохранить всю их историю.

    Возвращает (отфильтрованные события, таблицу выбранных пользователей).
    Пользователи с числом событий < min_events отбрасываются.
    ``exclude`` позволяет получить дизъюнктную выборку (второй сэмпл).
    """
    stats = (
        lf.group_by("uid")
        .agg(pl.len().alias("n_events"))
        .filter(pl.col("n_events") >= min_events)
        .sort("uid")  # детерминированный порядок до сэмплирования
        .collect(engine="streaming")
    )
    if exclude is not None and len(exclude):
        stats = stats.filter(~pl.col("uid").is_in(exclude.tolist()))
    if stats.height <= n_users:
        chosen = stats
    else:
        chosen = stats.sample(n=n_users, seed=seed, shuffle=True)
    events = lf.filter(pl.col("uid").is_in(chosen.get_column("uid").implode())).sort(
        ["uid", "ts", "item_id"]
    )
    return events, chosen.sort("uid")


def normalize_events(events: pl.DataFrame) -> pl.DataFrame:
    """Привести типы и порядок колонок."""
    return events.select(EVENT_COLUMNS)
