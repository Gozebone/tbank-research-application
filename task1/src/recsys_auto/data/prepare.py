"""Подготовка данных: нормализация, сэмплирование пользователей, метаданные."""

from __future__ import annotations

import hashlib

import numpy as np
import polars as pl

from ..config import Config, resolve
from ..utils.io import ensure_dir, get_logger, save_json
from .load import normalize_events, sample_users, scan_yambda

log = get_logger(__name__)


def _uid_hash(uids: pl.Series) -> str:
    array = np.sort(uids.unique().to_numpy())
    return hashlib.sha1(array.tobytes()).hexdigest()[:12]


def _user_stats(events: pl.DataFrame) -> pl.DataFrame:
    return (
        events.group_by("uid")
        .agg(
            pl.len().alias("n_events"),
            (pl.col("event_type") == "listen").sum().alias("n_listens"),
            (pl.col("event_type") == "like").sum().alias("n_likes"),
            (pl.col("event_type") == "dislike").sum().alias("n_dislikes"),
            pl.col("item_id").n_unique().alias("n_items"),
            pl.col("ts").min().alias("first_ts"),
            pl.col("ts").max().alias("last_ts"),
            (pl.col("ts") // 86_400).n_unique().alias("n_active_days"),
        )
        .with_columns(
            ((pl.col("last_ts") - pl.col("first_ts")) / 86_400).alias("tenure_days"),
        )
        .sort("uid")
    )


def _item_stats(events: pl.DataFrame) -> pl.DataFrame:
    listens = events.filter(pl.col("event_type") == "listen")
    agg = listens.group_by("item_id").agg(
        pl.len().alias("n_listens"),
        pl.col("uid").n_unique().alias("n_users"),
        pl.col("played_ratio").mean().alias("mean_played_ratio"),
        pl.col("played_ratio").max().alias("max_played_ratio"),
        pl.col("track_length_seconds").mean().alias("mean_track_length"),
        pl.col("ts").min().alias("first_ts"),
        pl.col("ts").max().alias("last_ts"),
    )
    likes = (
        events.filter(pl.col("event_type") == "like")
        .group_by("item_id")
        .agg(pl.len().alias("n_likes"))
    )
    return agg.join(likes, on="item_id", how="left").with_columns(
        pl.col("n_likes").fill_null(0)
    )


def _item_mappings(raw_map: str, item_ids: pl.Series, key: str) -> pl.DataFrame:
    df = (
        pl.scan_parquet(raw_map)
        .filter(pl.col("item_id").is_in(item_ids.implode()))
        .unique()
        .collect(engine="streaming")
    )
    primary = (
        df.group_by("item_id")
        .agg(
            pl.col(key).min().alias(f"primary_{key}"),
            pl.len().alias(f"n_{key}s"),
        )
    )
    return df.sort(["item_id", key]), primary


def prepare_kion(cfg: Config, force: bool = False) -> dict[str, str]:
    """Нормализовать KION в ту же схему событий, что и Yambda.

    KION хранит агрегат по паре (user, item): дата последнего просмотра,
    суммарная длительность и доля просмотра. Поэтому одно событие = один
    просмотр; ``event_type`` всегда ``listen`` (аналог просмотра), а
    ``played_ratio`` берётся из ``watched_pct``.

    Категориальные связи: ``genres`` играет роль исполнителя (косвенная связь
    «пользователь × жанр»), ``studios`` — роль альбома.
    """
    interim = resolve(cfg.paths.interim)
    out = {
        "events": resolve(cfg.prepare.events_file),
        "users": resolve(cfg.prepare.users_file),
        "items": resolve(cfg.prepare.items_file),
        "artists": resolve(cfg.prepare.artists_file),
        "albums": interim / "item_albums.parquet",
    }
    if not force and all(p.exists() for p in out.values()):
        log.info("prepared data already exists; use --force to rebuild")
        return {k: str(v) for k, v in out.items()}
    for p in out.values():
        ensure_dir(p.parent)

    raw = resolve(cfg.data.kion.raw_dir)
    interactions = raw / "data_en" / "interactions.csv"
    items_csv = raw / "data_en" / "items_en.csv"
    if not interactions.exists():
        raise FileNotFoundError(interactions)

    lf = pl.scan_csv(interactions).with_columns(
        pl.col("user_id").cast(pl.Int64).alias("uid"),
        pl.col("item_id").cast(pl.Int64),
        (pl.col("last_watch_dt").str.to_date().dt.epoch("d") * 86_400).cast(pl.Int64).alias("ts"),
        pl.lit("listen").alias("event_type"),
        pl.lit(1, dtype=pl.Int8).alias("is_organic"),
        (pl.col("watched_pct").fill_null(0.0) / 100.0).cast(pl.Float32).alias("played_ratio"),
        pl.col("total_dur").cast(pl.Float32).alias("track_length_seconds"),
    )

    min_events = int(cfg.data.kion.min_user_events)
    n_users = int(cfg.data.kion.n_users)
    stats = (
        lf.group_by("uid")
        .agg(pl.len().alias("n_events"))
        .filter(pl.col("n_events") >= min_events)
        .sort("uid")
        .collect(engine="streaming")
    )
    chosen = (
        stats
        if stats.height <= n_users
        else stats.sample(n=n_users, seed=int(cfg.project.seed), shuffle=True)
    )
    events = (
        lf.filter(pl.col("uid").is_in(chosen.get_column("uid").implode()))
        .collect(engine="streaming")
        .select(["uid", "item_id", "ts", "event_type", "is_organic", "played_ratio", "track_length_seconds"])
        .sort(["uid", "ts", "item_id"])
    )
    log.info("kion events: %d rows, %d users", events.height, events["uid"].n_unique())
    events.write_parquet(out["events"], compression="zstd", statistics=True)

    users = _user_stats(events)
    items = _item_stats(events)

    meta = (
        pl.read_csv(items_csv)
        .select(["item_id", "content_type", "genres", "studios", "release_year"])
        .with_columns(
            pl.col("item_id").cast(pl.Int64),
            pl.col("genres").fill_null("unknown").alias("genre_key"),
            pl.col("studios").fill_null("unknown").alias("studio_key"),
        )
    )
    genres = (
        meta.select("genre_key")
        .unique()
        .sort("genre_key")
        .with_row_index("primary_artist_id")
        .with_columns(pl.col("primary_artist_id").cast(pl.UInt32))
    )
    studios = (
        meta.select("studio_key")
        .unique()
        .sort("studio_key")
        .with_row_index("primary_album_id")
        .with_columns(pl.col("primary_album_id").cast(pl.UInt32))
    )
    meta = (
        meta.join(genres, on="genre_key", how="left")
        .join(studios, on="studio_key", how="left")
        .drop(["genre_key", "studio_key"])
    )
    items = items.join(meta, on="item_id", how="left")
    users.write_parquet(out["users"], compression="zstd")
    items.write_parquet(out["items"], compression="zstd")
    items.select(["item_id", "primary_artist_id"]).write_parquet(out["artists"], compression="zstd")
    items.select(["item_id", "primary_album_id"]).write_parquet(out["albums"], compression="zstd")

    stats_payload = {
        "dataset": "kion",
        "n_events": int(events.height),
        "n_users": int(users.height),
        "n_items": int(items.height),
        "ts_min": int(events["ts"].min()),
        "ts_max": int(events["ts"].max()),
        "span_days": float((events["ts"].max() - events["ts"].min()) / 86_400),
        "uid_hash": _uid_hash(events.get_column("uid")),
        "event_counts": {row[0]: int(row[1]) for row in events.group_by("event_type").agg(pl.len()).iter_rows()},
    }
    save_json(stats_payload, interim / "stats.json")
    log.info("kion stats: %s", stats_payload)
    return {k: str(v) for k, v in out.items()}


def prepare_yambda(cfg: Config, force: bool = False) -> dict[str, str]:
    paths = cfg.data.yambda
    out = {
        "events": resolve(cfg.prepare.events_file),
        "users": resolve(cfg.prepare.users_file),
        "items": resolve(cfg.prepare.items_file),
        "artists": resolve(cfg.prepare.artists_file),
        "albums": resolve(cfg.paths.interim) / "item_albums.parquet",
    }
    if not force and all(p.exists() for p in out.values()):
        log.info("prepared data already exists; use --force to rebuild")
        return {k: str(v) for k, v in out.items()}

    for p in out.values():
        ensure_dir(p.parent)

    lf = scan_yambda(resolve(paths.raw_file))
    exclude = None
    exclude_file = paths.get("exclude_users_file") if hasattr(paths, "get") else None
    if exclude_file:
        exclude = pl.read_parquet(resolve(exclude_file)).get_column("uid").to_numpy()
        log.info("excluding %d users from the first sample", len(exclude))
    events_lf, chosen_users = sample_users(
        lf,
        n_users=int(paths.n_users),
        min_events=int(paths.min_user_events),
        seed=int(cfg.project.seed),
        exclude=exclude,
    )
    log.info("collecting events for %d users ...", chosen_users.height)
    events = normalize_events(events_lf.collect(engine="streaming"))
    log.info("events: %d rows, %.1f MB", events.height, events.estimated_size() / 1e6)
    events.write_parquet(out["events"], compression="zstd", statistics=True)

    users = _user_stats(events)
    items = _item_stats(events)
    users.write_parquet(out["users"], compression="zstd")
    items.write_parquet(out["items"], compression="zstd")

    item_ids = items.get_column("item_id")
    artists, artist_primary = _item_mappings(resolve(paths.artist_map), item_ids, "artist_id")
    albums, album_primary = _item_mappings(resolve(paths.album_map), item_ids, "album_id")
    artists.write_parquet(out["artists"], compression="zstd")
    albums.write_parquet(out["albums"], compression="zstd")
    items = (
        items.join(artist_primary, on="item_id", how="left")
        .join(album_primary, on="item_id", how="left")
    )
    items.write_parquet(out["items"], compression="zstd")

    stats = {
        "n_events": int(events.height),
        "n_users": int(users.height),
        "n_items": int(items.height),
        "ts_min": int(events["ts"].min()),
        "ts_max": int(events["ts"].max()),
        "span_days": float((events["ts"].max() - events["ts"].min()) / 86_400),
        "uid_hash": _uid_hash(events.get_column("uid")),
        "event_counts": {
            row[0]: int(row[1])
            for row in events.group_by("event_type").agg(pl.len()).iter_rows()
        },
    }
    save_json(stats, resolve(cfg.paths.interim) / "stats.json")
    log.info("prepared stats: %s", stats)
    return {k: str(v) for k, v in out.items()}


def prepare_dataset(cfg: Config, force: bool = False) -> dict[str, str]:
    """Диспетчер по датасету из конфига."""
    dataset = str(cfg.data.dataset).lower()
    if dataset == "yambda":
        return prepare_yambda(cfg, force=force)
    if dataset == "kion":
        return prepare_kion(cfg, force=force)
    raise ValueError(f"unknown dataset: {dataset}")
