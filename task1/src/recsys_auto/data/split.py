"""Разбиение по времени и снимки (snapshots) для обучения.

Схема (все границы в секундах от начала данных):

    train snapshots ... | train_end | gap | val_start | val | gap | test_start | test

Каждый снимок описывается cutoff: признаки считаются по событиям с ts < cutoff,
а целевые события — в окне [cutoff, cutoff + horizon).
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from ..utils.time import days_to_seconds


@dataclass(frozen=True)
class Snapshot:
    name: str
    split: str  # train | val | test
    cutoff: int
    label_start: int
    label_end: int

    @property
    def horizon_seconds(self) -> int:
        return self.label_end - self.label_start


@dataclass(frozen=True)
class TimeSplit:
    t_min: int
    t_max: int
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    test_start: int
    test_end: int

    def snapshots(self, train_weeks: int, horizon_days: int = 7) -> list[Snapshot]:
        horizon = days_to_seconds(horizon_days)
        out: list[Snapshot] = []
        # обучающие снимки: последние train_weeks недель train-периода
        for k in range(train_weeks):
            label_end = self.train_end - k * horizon
            label_start = label_end - horizon
            if label_start <= self.train_start:
                break
            out.append(
                Snapshot(
                    name=f"train_w{k}",
                    split="train",
                    cutoff=label_start,
                    label_start=label_start,
                    label_end=label_end,
                )
            )
        out.append(
            Snapshot(
                name="val",
                split="val",
                cutoff=self.val_start,
                label_start=self.val_start,
                label_end=self.val_end,
            )
        )
        out.append(
            Snapshot(
                name="test",
                split="test",
                cutoff=self.test_start,
                label_start=self.test_start,
                label_end=self.test_end,
            )
        )
        return out


def make_time_split(
    events: pl.DataFrame | pl.LazyFrame,
    test_days: int = 7,
    val_days: int = 7,
    gap_days: int = 7,
) -> TimeSplit:
    stats = events.select(pl.col("ts").min().alias("t_min"), pl.col("ts").max().alias("t_max"))
    if isinstance(stats, pl.LazyFrame):
        stats = stats.collect()
    row = stats.row(0)
    t_min, t_max = int(row[0]), int(row[1])
    test_end = t_max + 1
    test_start = test_end - days_to_seconds(test_days)
    val_end = test_start - days_to_seconds(gap_days)
    val_start = val_end - days_to_seconds(val_days)
    train_end = val_start - days_to_seconds(gap_days)
    if not (t_min < train_end < val_start < val_end < test_start < test_end):
        raise ValueError("time split is degenerate; not enough history")
    return TimeSplit(
        t_min=t_min,
        t_max=t_max,
        train_start=t_min,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        test_start=test_start,
        test_end=test_end,
    )
