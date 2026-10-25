"""Вычисление агрегатов по спецификациям признаков (DuckDB).

Ключевая идея эффективности: для одной группы ``(entity, metric, event_type)``
все временные окна считаются одним запросом через агрегаты с ``FILTER``.
Разные сущности возвращаются отдельными таблицами и джойнятся к кандидатам
по своим ключам.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import duckdb
import pandas as pd

from .dsl import (
    CONDITIONS,
    ENTITY_KEYS,
    ENTITY_TABLE,
    KEY_TABLE_NAMES,
    METRIC_CONDITIONS,
    FeatureSpec,
    METRICS,
)


def create_connection(
    events_path: str | Path,
    items_path: str | Path | None = None,
    threads: int = 16,
    memory_limit: str = "48GB",
) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET threads={int(threads)}")
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(
        f"CREATE OR REPLACE VIEW events AS SELECT * FROM read_parquet('{events_path}')"
    )
    if items_path is not None:
        con.execute(
            "CREATE OR REPLACE VIEW items AS "
            f"SELECT item_id, primary_artist_id, primary_album_id FROM read_parquet('{items_path}')"
        )
        con.execute(
            "CREATE OR REPLACE VIEW events_enriched AS "
            "SELECT e.*, i.primary_artist_id, i.primary_album_id "
            "FROM events e LEFT JOIN items i USING (item_id)"
        )
    else:
        con.execute(
            "CREATE OR REPLACE VIEW events_enriched AS "
            "SELECT *, CAST(NULL AS UINTEGER) AS primary_artist_id, "
            "CAST(NULL AS UINTEGER) AS primary_album_id FROM events"
        )
    return con


def register_key_tables(
    con: duckdb.DuckDBPyConnection,
    candidates: pd.DataFrame,
) -> set[str]:
    """Зарегистрировать таблицы ключей кандидатов.

    Позволяет считать агрегаты только для пар, которые реально встречаются
    среди кандидатов, вместо всех пар в истории. Возвращает набор сущностей,
    для которых ключи доступны.
    """
    registered: set[str] = set()
    for entity, keys in ENTITY_KEYS.items():
        if all(k in candidates.columns for k in keys):
            frame = (
                candidates[list(keys)]
                .dropna()
                .astype({k: "int64" for k in keys})
                .drop_duplicates()
            )
            con.register(KEY_TABLE_NAMES[entity], frame)
            registered.add(entity)
    return registered


def _group_specs(specs: list[FeatureSpec]) -> dict[tuple, list[FeatureSpec]]:
    groups: dict[tuple, list[FeatureSpec]] = defaultdict(list)
    for spec in specs:
        groups[(spec.entity, spec.metric, spec.event_type)].append(spec)
    return groups


def _window_expr(spec: FeatureSpec, cutoff: int, alias: str) -> str:
    if spec.metric == "decay_count":
        tau_seconds = float(spec.decay_days) * 86_400.0
        base = (
            f"sum(exp(-ln(2) * ({int(cutoff)} - t.ts) / {tau_seconds}))"
        )
    else:
        base = METRICS[spec.metric]
    conditions: list[str] = []
    if spec.window_days is not None:
        start = cutoff - spec.window_days * 86_400
        conditions.append(f"ts >= {start}")
    metric_condition = METRIC_CONDITIONS.get(spec.metric)
    if metric_condition:
        conditions.append(metric_condition)
    if spec.condition:
        conditions.append(CONDITIONS[spec.condition])
    if not conditions:
        return f'{base} AS "{alias}"'
    return f'{base} FILTER (WHERE {" AND ".join(conditions)}) AS "{alias}"'


def compute_specs(
    con: duckdb.DuckDBPyConnection,
    specs: list[FeatureSpec],
    cutoff: int,
    key_entities: set[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Посчитать признаки на момент cutoff.

    Если передан ``key_entities`` (результат ``register_key_tables``),
    агрегация ограничивается ключами кандидатов — это резко уменьшает объём
    результата при поиске по сотням признаков.

    Возвращает ``{entity: DataFrame}``; в каждой таблице ключи сущности и
    по колонке на спецификацию (имя колонки = ``spec.name``).
    """
    if not specs:
        return {}
    by_entity: dict[str, pd.DataFrame] = {}
    for (entity, metric, event_type), group in _group_specs(specs).items():
        keys = ENTITY_KEYS[entity]
        table = ENTITY_TABLE[entity]
        exprs = [_window_expr(spec, cutoff, spec.name) for spec in group]
        where = [f"t.ts < {cutoff}"]
        if event_type is not None:
            where.append(f"t.event_type = '{event_type}'")

        if key_entities and entity in key_entities:
            key_table = KEY_TABLE_NAMES[entity]
            join = " AND ".join(f"t.{k} = k.{k}" for k in keys)
            keys_expr = ", ".join(f"k.{k}" for k in keys)
            from_clause = f"{table} t JOIN {key_table} k ON {join}"
        else:
            keys_expr = ", ".join(f"t.{k}" for k in keys)
            from_clause = f"{table} t"

        sql = (
            f"SELECT {keys_expr}, {', '.join(exprs)} "
            f"FROM {from_clause} WHERE {' AND '.join(where)} "
            f"GROUP BY {keys_expr}"
        )
        frame = con.execute(sql).fetch_df()
        if entity in by_entity:
            by_entity[entity] = by_entity[entity].merge(frame, on=list(keys), how="outer")
        else:
            by_entity[entity] = frame
    return by_entity


def fill_missing(df: pd.DataFrame, specs: list[FeatureSpec]) -> pd.DataFrame:
    """Заполнить пропуски: счётчики -> 0, остальные агрегаты оставить NaN."""
    for spec in specs:
        if spec.name not in df.columns:
            df[spec.name] = 0.0 if spec.fill_zero else pd.NA
        elif spec.fill_zero:
            df[spec.name] = df[spec.name].fillna(0)
    return df


def attach_features(
    candidates: pd.DataFrame,
    features: dict[str, pd.DataFrame],
    specs: list[FeatureSpec],
    items: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Приджойнить агрегаты к таблице кандидатов."""
    need_meta = any(s.entity in ("user_artist", "user_album") for s in specs)
    cand = candidates
    if need_meta:
        if items is None:
            raise ValueError("items frame is required for artist/album features")
        meta = items[["item_id", "primary_artist_id", "primary_album_id"]].drop_duplicates(
            "item_id"
        )
        missing = [c for c in ("primary_artist_id", "primary_album_id") if c not in cand.columns]
        if missing:
            cand = cand.merge(meta, on="item_id", how="left")

    for entity, frame in features.items():
        keys = list(ENTITY_KEYS[entity])
        cols = keys + [s.name for s in specs if s.entity == entity and s.name in frame.columns]
        cand = cand.merge(frame[cols], on=keys, how="left")
    return fill_missing(cand, specs)
