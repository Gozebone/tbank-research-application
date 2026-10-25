"""Скрининг и материализация расширенных выражений.

Листья DSL считаются тем же DuckDB-кодом, что и плоские признаки; бинарные
операции вычисляются в pandas поверх листьев. Это сохраняет единый протокол
проверки: те же снимки, те же ключи кандидатов, тот же within-user AUC.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config, resolve
from ..features.aggregates import (
    attach_features,
    compute_specs,
    create_connection,
    register_key_tables,
)
from ..features.dsl import FeatureSpec
from ..utils.fingerprint import dataset_fingerprint
from ..utils.io import ensure_dir, get_logger, save_json
from .expr import Expr
from .search import KEY_COLUMNS, META_COLUMNS, _items_frame, chunks, load_candidates
from .selectors import per_user_auc, univariate_auc

log = get_logger(__name__)


def unique_leaves(exprs: list[FeatureSpec | Expr]) -> list[FeatureSpec]:
    seen: dict[str, FeatureSpec] = {}
    for expr in exprs:
        for leaf in expr.leaves() if isinstance(expr, Expr) else [expr]:
            seen.setdefault(leaf.name, leaf)
    return list(seen.values())


def _evaluate_all(
    exprs: list[FeatureSpec | Expr], values: pd.DataFrame
) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    for expr in exprs:
        if isinstance(expr, Expr):
            out[expr.name] = expr.evaluate(values)
    return out


class ExpressionScreener:
    """Держит подвыборку кандидатов в памяти и досчитывает листья по запросу."""

    def __init__(
        self,
        cfg: Config,
        sample_users: int = 200,
        seed: int = 42,
        chunk_size: int = 40,
    ) -> None:
        self.cfg = cfg
        self.sample_users = sample_users
        self.seed = seed
        self.chunk_size = chunk_size

        sample = load_candidates(cfg, "val", columns=META_COLUMNS)
        if sample_users and sample["uid"].nunique() > sample_users:
            rng = np.random.default_rng(seed)
            uids = rng.choice(sample["uid"].unique(), size=sample_users, replace=False)
            sample = sample[sample["uid"].isin(uids)].reset_index(drop=True)
        self.sample = sample
        self.sample["_row_id"] = np.arange(len(sample))
        self.labels = sample[["uid", "label"]].reset_index(drop=True)
        self.values = pd.DataFrame(index=sample.index)
        self._items = _items_frame(cfg)
        self.con = create_connection(
            resolve(cfg.prepare.events_file),
            resolve(cfg.prepare.items_file),
            threads=int(cfg.model.params.get("num_threads", 16)),
        )
        self.key_entities = register_key_tables(self.con, sample)
        self.cutoff = int(sample["cutoff"].iloc[0])
        log.info(
            "expression screener: %d users, %d rows, cutoff=%d",
            sample["uid"].nunique(),
            len(sample),
            self.cutoff,
        )

    def ensure(self, leaves: list[FeatureSpec]) -> None:
        missing = [leaf for leaf in leaves if leaf.name not in self.values.columns]
        if not missing:
            return
        started = time.perf_counter()
        key_frame = self.sample[["_row_id"] + KEY_COLUMNS]
        parts: list[pd.DataFrame] = []
        for chunk in chunks(missing, self.chunk_size):
            aggregates = compute_specs(
                self.con, chunk, cutoff=self.cutoff, key_entities=self.key_entities
            )
            featured = attach_features(key_frame, aggregates, chunk, items=self._items)
            featured = featured.set_index("_row_id").sort_index()
            columns = [spec.name for spec in chunk if spec.name in featured.columns]
            parts.append(featured[columns])
        new_values = pd.concat(parts, axis=1)
        self.values = pd.concat([self.values, new_values], axis=1)
        log.info(
            "computed %d new leaves in %.1fs", len(missing), time.perf_counter() - started
        )

    def score(self, exprs: list[FeatureSpec | Expr]) -> pd.DataFrame:
        self.ensure(unique_leaves(exprs))
        columns: dict[str, pd.Series] = {}
        for expr in exprs:
            if isinstance(expr, Expr):
                columns[expr.name] = expr.evaluate(self.values)
            else:
                columns[expr.name] = self.values[expr.name].astype("float64")
        frame = pd.concat([self.labels, pd.DataFrame(columns)], axis=1)
        names = list(columns)
        auc = univariate_auc(frame, names)
        user_auc = per_user_auc(frame, names)
        rows = []
        for expr in exprs:
            if expr.name not in auc.index:
                continue
            rows.append(
                {
                    "name": expr.name,
                    "entity": expr.entity if isinstance(expr, Expr) else expr.entity,
                    "op": expr.op if isinstance(expr, Expr) else "agg",
                    "auc": float(auc[expr.name]),
                    "user_auc": float(user_auc[expr.name]),
                    "nan_rate": float(frame[expr.name].isna().mean()),
                }
            )
        result = pd.DataFrame(rows)
        if len(result):
            result["strength"] = (result["user_auc"] - 0.5).abs()
            result = result.sort_values("strength", ascending=False).reset_index(drop=True)
        return result

    def close(self) -> None:
        try:
            self.con.close()
        except Exception:  # pragma: no cover
            pass


def materialize_expressions(
    cfg: Config,
    exprs: list[FeatureSpec | Expr],
    snapshot: str,
    out_path: Path,
    chunk_size: int = 40,
) -> Path:
    """Посчитать листья и выражения для всех кандидатов снимка."""
    started = time.perf_counter()
    leaves = unique_leaves(exprs)
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
        for chunk in chunks(leaves, chunk_size):
            aggregates = compute_specs(con, chunk, cutoff=cutoff, key_entities=key_entities)
            featured = attach_features(key_frame, aggregates, chunk, items=items)
            featured = featured.set_index("_row_id").sort_index()
            columns = [spec.name for spec in chunk if spec.name in featured.columns]
            parts.append(featured[columns])
    finally:
        con.close()

    values = pd.concat(parts, axis=1)
    derived = _evaluate_all(exprs, values)
    out = pd.concat(
        [candidates.set_index("_row_id"), values, pd.DataFrame(derived, index=values.index)],
        axis=1,
    ).reset_index(drop=True)
    ensure_dir(out_path.parent)
    out.to_parquet(out_path, index=False)
    log.info(
        "materialized %s: %d rows, %d leaves + %d expressions in %.1fs",
        snapshot,
        len(out),
        len(leaves),
        len(derived),
        time.perf_counter() - started,
    )
    save_json(
        {
            "snapshot": snapshot,
            "data_fingerprint": dataset_fingerprint(cfg),
            "n_rows": len(out),
            "n_leaves": len(leaves),
            "n_expressions": len(derived),
            "seconds": time.perf_counter() - started,
        },
        out_path.with_suffix(".meta.json"),
    )
    return out_path
