"""Автоматический поиск признаков: скрининг и материализация.

Схема:

1. ``run_screen`` — считает все сгенерированные спецификации на подвыборке
   кандидатов валидации и оценивает одномерный AUC.
2. ``run_materialize`` — для отобранных спецификаций (плюс «компаньоны»,
   нужные производным признакам) считает значения для всех снимков и
   сохраняет их.
3. Обучение и сравнение вариантов — в ``scripts/06_auto_search.py``.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from ..config import Config, resolve
from ..features.aggregates import (
    attach_features,
    compute_specs,
    create_connection,
    register_key_tables,
)
from ..features.dsl import FeatureSpec
from ..features.dsl import INDEX_ONLY_METRICS
from ..utils.io import ensure_dir, get_logger, load_json, save_json
from ..utils.fingerprint import dataset_fingerprint
from .selectors import per_user_auc, univariate_auc

log = get_logger(__name__)

META_COLUMNS = [
    "uid",
    "item_id",
    "label",
    "snapshot",
    "split",
    "cutoff",
    "label_start",
    "label_end",
    "primary_artist_id",
    "primary_album_id",
]
KEY_COLUMNS = ["uid", "item_id", "primary_artist_id", "primary_album_id"]


def load_candidates(cfg: Config, name: str, columns: list[str] | None = None) -> pd.DataFrame:
    path = resolve(cfg.paths.processed) / "datasets" / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    meta_path = path.with_suffix(".meta.json")
    if meta_path.exists():
        expected = dataset_fingerprint(cfg)
        actual = load_json(meta_path).get("data_fingerprint")
        if actual and actual != expected:
            raise RuntimeError(
                f"dataset {name} was built from different events "
                f"({actual} != {expected}); rebuild it"
            )
    return pd.read_parquet(path, columns=columns)


def chunks(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _items_frame(cfg: Config) -> pd.DataFrame:
    return pd.read_parquet(resolve(cfg.prepare.items_file))[
        ["item_id", "primary_artist_id", "primary_album_id"]
    ]


def screen_specs(
    cfg: Config,
    specs: list[FeatureSpec],
    snapshot: str = "val",
    sample_users: int = 1_000,
    chunk_size: int = 60,
    seed: int = 42,
    save: bool = True,
) -> pd.DataFrame:
    """Скрининг всех спецификаций на подвыборке пользователей валидации.

    Считаются два скора: глобальный AUC и средний AUC внутри пользователя.
    Для отбора используется второй — он соответствует ranking-задаче.

    ``save=False`` не пишет ``screen.csv``/``screen_meta.json`` — используется
    LLM-агентом, который скрининит свои кандидаты в памяти.
    """
    started = time.perf_counter()
    sample = load_candidates(cfg, snapshot, columns=META_COLUMNS)
    if sample_users and sample["uid"].nunique() > sample_users:
        rng = np.random.default_rng(seed)
        uids = rng.choice(sample["uid"].unique(), size=sample_users, replace=False)
        sample = sample[sample["uid"].isin(uids)].reset_index(drop=True)
    sample["_row_id"] = np.arange(len(sample))
    items = _items_frame(cfg)

    con = create_connection(
        resolve(cfg.prepare.events_file),
        resolve(cfg.prepare.items_file),
        threads=int(cfg.model.params.get("num_threads", 16)),
    )
    key_entities = register_key_tables(con, sample)

    rows: list[dict] = []
    feature_counter = 0
    try:
        for chunk in chunks(specs, chunk_size):
            # cutoff берём из снимка: в таблице кандидатов он одинаковый
            cutoff = int(sample["cutoff"].iloc[0])
            aggregates = compute_specs(con, chunk, cutoff=cutoff, key_entities=key_entities)
            featured = attach_features(
                sample[["_row_id"] + KEY_COLUMNS], aggregates, chunk, items=items
            ).set_index("_row_id").sort_index()
            columns = [s.name for s in chunk if s.name in featured.columns]
            joined = pd.concat([sample[["uid", "label"]], featured[columns]], axis=1)
            auc = univariate_auc(joined, columns)
            user_auc = per_user_auc(joined, columns)
            for spec in chunk:
                if spec.name not in auc.index:
                    continue
                rows.append(
                    {
                        **spec.describe(),
                        "auc": float(auc[spec.name]),
                        "user_auc": float(user_auc[spec.name]),
                        "nan_rate": float(featured[spec.name].isna().mean()),
                    }
                )
            feature_counter += len(columns)
            log.info(
                "screened %d/%d specs (%.1fs)",
                feature_counter,
                len(specs),
                time.perf_counter() - started,
            )
    finally:
        con.close()

    result = pd.DataFrame(rows)
    result["strength"] = (result["user_auc"] - 0.5).abs().fillna(
        (result["auc"] - 0.5).abs()
    )
    result = result.sort_values(
        ["strength", "auc"], ascending=[False, False]
    ).reset_index(drop=True)
    if save:
        out_dir = ensure_dir(resolve(cfg.paths.experiments) / "auto_search")
        result.to_csv(out_dir / "screen.csv", index=False)
        save_json(
            {
                "n_specs": len(specs),
                "n_evaluated": len(result),
                "sample_users": int(sample["uid"].nunique()),
                "sample_rows": len(sample),
                "screen_seconds": time.perf_counter() - started,
            },
            out_dir / "screen_meta.json",
        )
    log.info("screen done in %.1fs: %d specs", time.perf_counter() - started, len(result))
    return result


def _spec_index(specs: list[FeatureSpec]) -> dict[str, FeatureSpec]:
    return {s.name: s for s in specs}


def select_specs(
    screen: pd.DataFrame,
    all_specs: list[FeatureSpec],
    top_k: int = 80,
    per_entity: int | None = None,
) -> tuple[list[FeatureSpec], list[FeatureSpec]]:
    """Выбрать базовые спецификации.

    ``per_entity`` задаёт квоту на сущность — это защищает от того, что
    топ-80 целиком заполнится коррелирующими признаками одной сущности
    (например, только user_artist) и потеряет популярность объектов.
    """
    index = _spec_index(all_specs)
    usable = screen[(screen["nan_rate"] < 0.999)].copy()
    if "metric" in usable.columns:
        usable = usable[~usable["metric"].isin(INDEX_ONLY_METRICS)]
    if per_entity is not None:
        chosen = (
            usable.groupby("entity", group_keys=False, sort=False)
            .head(per_entity)
            .sort_values("strength", ascending=False)
        )
    else:
        chosen = usable.head(top_k)
    selected_names = [n for n in chosen["name"] if n in index]
    selected = [index[n] for n in selected_names]

    companions: dict[str, FeatureSpec] = {}
    for spec in selected:
        entity, window = spec.entity, spec.window_days
        candidates = [
            ("count", window, "like"),
            ("count", window, "listen"),
            ("count", None, "listen"),
            ("organic_count", window, "listen"),
            ("last_ts", None, None),
            ("first_ts", None, None),
        ]
        if entity != "user":
            candidates.append(("count", window, "listen"))
        for metric, win, event in candidates:
            try:
                extra = FeatureSpec(entity, metric, win, event)
            except ValueError:
                continue
            if extra.name not in {s.name for s in selected}:
                companions.setdefault(extra.name, extra)
        # знаменатель для share-признаков
        if entity != "user":
            user_col = FeatureSpec("user", "count", window, "listen")
            if user_col.name not in {s.name for s in selected}:
                companions.setdefault(user_col.name, user_col)
    companion_specs = [s for s in companions.values() if s.name in index]
    return selected, companion_specs


def materialize_specs(
    cfg: Config,
    specs: list[FeatureSpec],
    snapshot: str,
    out_path: Path,
    chunk_size: int = 60,
) -> Path:
    """Посчитать признаки для всех кандидатов снимка и сохранить."""
    started = time.perf_counter()
    candidates = load_candidates(cfg, snapshot, columns=META_COLUMNS)
    candidates["_row_id"] = np.arange(len(candidates))
    cutoff = int(candidates["cutoff"].iloc[0])
    items = _items_frame(cfg)

    con = create_connection(
        resolve(cfg.prepare.events_file),
        resolve(cfg.prepare.items_file),
        threads=int(cfg.model.params.get("num_threads", 16)),
    )
    key_entities = register_key_tables(con, candidates)
    key_frame = candidates[["_row_id"] + KEY_COLUMNS]
    parts: list[pd.DataFrame] = []
    try:
        for chunk in chunks(specs, chunk_size):
            aggregates = compute_specs(con, chunk, cutoff=cutoff, key_entities=key_entities)
            featured = attach_features(key_frame, aggregates, chunk, items=items)
            featured = featured.set_index("_row_id").sort_index()
            columns = [s.name for s in chunk if s.name in featured.columns]
            parts.append(featured[columns])
    finally:
        con.close()

    features = pd.concat(parts, axis=1)
    out = pd.concat([candidates.set_index("_row_id"), features], axis=1).reset_index(drop=True)
    ensure_dir(out_path.parent)
    out.to_parquet(out_path, index=False)
    log.info(
        "materialized %s: %d rows x %d features in %.1fs",
        snapshot,
        len(out),
        len(specs),
        time.perf_counter() - started,
    )
    save_json(
        {
            "snapshot": snapshot,
            "data_fingerprint": dataset_fingerprint(cfg),
            "n_rows": len(out),
            "n_specs": len(specs),
            "seconds": time.perf_counter() - started,
        },
        out_path.with_suffix(".meta.json"),
    )
    return out_path
