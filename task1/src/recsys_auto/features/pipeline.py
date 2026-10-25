"""Сборка датасета: кандидаты + признаки для одного снимка."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import polars as pl

from ..config import Config, resolve
from ..data.candidates import build_candidates
from ..data.split import Snapshot
from ..utils.io import ensure_dir, get_logger, load_json, save_json
from ..utils.fingerprint import dataset_fingerprint
from .aggregates import attach_features, compute_specs, create_connection
from .dsl import INDEX_ONLY_METRICS, FeatureSpec
from .manual import add_derived, manual_specs

log = get_logger(__name__)

META_COLUMNS = {
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
}


@dataclass
class DatasetBundle:
    frame: pd.DataFrame
    specs: list[FeatureSpec]
    feature_cols: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.feature_cols = [c for c in self.frame.columns if c not in META_COLUMNS]


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in META_COLUMNS]


def build_snapshot_dataset(
    cfg: Config,
    snapshot: Snapshot,
    specs: list[FeatureSpec] | None = None,
    seed: int | None = None,
) -> DatasetBundle:
    seed = int(cfg.project.seed) if seed is None else seed
    events_path = resolve(cfg.prepare.events_file)
    items_path = resolve(cfg.prepare.items_file)
    events = pl.read_parquet(events_path)
    items = pd.read_parquet(items_path)

    is_train = snapshot.split == "train"
    negatives = (
        cfg.candidates.negatives_per_positive_train
        if is_train
        else cfg.candidates.negatives_per_positive_eval
    )
    log.info("building candidates for %s (negatives/pos=%d)", snapshot.name, negatives)
    candidates = build_candidates(
        events,
        snapshot,
        max_positives_per_user=int(cfg.candidates.max_positives_per_user),
        negatives_per_positive=int(negatives),
        pool_size=int(cfg.candidates.negative_pool_size),
        min_popularity=int(cfg.candidates.min_item_popularity),
        seed=seed,
    )
    log.info("candidates: %d rows, positives: %.1f%%", len(candidates), 100 * candidates.label.mean())

    specs = specs or manual_specs()
    # дедупликация по имени (в ручном наборе возможно повторение)
    unique_specs: dict[str, FeatureSpec] = {}
    for spec in specs:
        unique_specs.setdefault(spec.name, spec)
    specs = list(unique_specs.values())

    log.info("computing %d feature specs at cutoff=%d", len(specs), snapshot.cutoff)
    con = create_connection(events_path, items_path, threads=int(cfg.model.params.get("num_threads", 16)))
    try:
        aggregates = compute_specs(con, specs, snapshot.cutoff)
    finally:
        con.close()

    frame = attach_features(candidates, aggregates, specs, items=items)
    frame = add_derived(frame, snapshot.cutoff)
    # сырые timestamp-агрегаты нужны были только для recency; в модель не идут
    index_only = [s.name for s in specs if s.metric in INDEX_ONLY_METRICS]
    frame = frame.drop(columns=[c for c in index_only if c in frame.columns])
    frame = frame.sort_values(["uid", "label"], ascending=[True, False]).reset_index(drop=True)
    return DatasetBundle(frame=frame, specs=specs)


def save_dataset(cfg: Config, bundle: DatasetBundle, snapshot: Snapshot) -> None:
    out_dir = ensure_dir(resolve(cfg.paths.processed) / "datasets")
    path = out_dir / f"{snapshot.name}.parquet"
    bundle.frame.to_parquet(path, index=False)
    save_json(
        {
            "snapshot": snapshot.__dict__,
            "data_fingerprint": dataset_fingerprint(cfg),
            "n_rows": len(bundle.frame),
            "n_users": int(bundle.frame.uid.nunique()),
            "positive_rate": float(bundle.frame.label.mean()),
            "feature_cols": bundle.feature_cols,
            "specs": [s.describe() for s in bundle.specs],
        },
        out_dir / f"{snapshot.name}.meta.json",
    )
    log.info("saved %s (%d rows, %d features)", path, len(bundle.frame), len(bundle.feature_cols))


def load_dataset(cfg: Config, name: str, check_fingerprint: bool = True) -> pd.DataFrame:
    out_dir = resolve(cfg.paths.processed) / "datasets"
    path = out_dir / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"dataset not built: {path}")
    if check_fingerprint:
        meta_path = out_dir / f"{name}.meta.json"
        if meta_path.exists():
            expected = dataset_fingerprint(cfg)
            actual = load_json(meta_path).get("data_fingerprint")
            if actual and actual != expected:
                raise RuntimeError(
                    f"dataset {name} was built from different events "
                    f"({actual} != {expected}); rebuild it with "
                    f"scripts/03_build_dataset.py --split all"
                )
    return pd.read_parquet(path)
