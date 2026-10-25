"""Обучение градиентного бустинга и инференс."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..utils.io import ensure_dir, get_logger, save_json

log = get_logger(__name__)


@dataclass
class TrainedModel:
    booster: lgb.Booster
    feature_cols: list[str]
    best_iteration: int


def train_lgbm(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_cols: list[str],
    params: dict,
    num_boost_rounds: int = 2000,
    early_stopping_rounds: int = 100,
) -> TrainedModel:
    train_set = lgb.Dataset(
        train[feature_cols],
        label=train["label"].to_numpy(),
        free_raw_data=False,
    )
    valid_set = lgb.Dataset(
        valid[feature_cols],
        label=valid["label"].to_numpy(),
        reference=train_set,
        free_raw_data=False,
    )
    log.info("training LightGBM: %d features, %d rows", len(feature_cols), len(train))
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=num_boost_rounds,
        valid_sets=[valid_set],
        valid_names=["valid"],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )
    return TrainedModel(
        booster=booster,
        feature_cols=feature_cols,
        best_iteration=booster.best_iteration or num_boost_rounds,
    )


def predict(model: TrainedModel, df: pd.DataFrame) -> np.ndarray:
    return model.booster.predict(df[model.feature_cols], num_iteration=model.best_iteration)


def add_scores(df: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    out["score"] = scores
    return out


def save_model(model: TrainedModel, out_dir: str | Path) -> Path:
    out_dir = ensure_dir(out_dir)
    path = out_dir / "model.txt"
    model.booster.save_model(str(path))
    save_json(
        {"best_iteration": model.best_iteration, "feature_cols": model.feature_cols},
        out_dir / "model.meta.json",
    )
    return path


def feature_importance(model: TrainedModel) -> pd.DataFrame:
    importance = model.booster.feature_importance(importance_type="gain")
    return (
        pd.DataFrame({"feature": model.feature_cols, "gain": importance})
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )
