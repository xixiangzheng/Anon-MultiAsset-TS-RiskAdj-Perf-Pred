from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

STRATEGY_DIR = Path(__file__).resolve().parent
if str(STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(STRATEGY_DIR))

from data_utils import add_group_history_features, top_correlated_features


def select_history_features(
    frame: pd.DataFrame,
    raw_features: list[str],
    *,
    top_k: int = 48,
    sample_rows: int = 200_000,
    seed: int = 2026,
) -> list[str]:
    if top_k <= 0:
        return []
    ranked = top_correlated_features(
        frame,
        raw_features,
        top_k=min(top_k, len(raw_features)),
        sample_rows=sample_rows,
        seed=seed,
    )
    return ranked[: min(top_k, len(ranked))]


def prepare_model_frame(
    frame: pd.DataFrame,
    *,
    raw_features: list[str],
    history_features: list[str],
    rolling_windows: tuple[int, ...] = (5,),
) -> tuple[pd.DataFrame, list[str]]:
    needed = ["row_id", "time_id", "asset_id", "weight", "target", *raw_features]
    out = frame.loc[:, [col for col in needed if col in frame.columns]].copy()
    out, engineered = add_group_history_features(
        out,
        history_features,
        rolling_windows=rolling_windows,
    )
    model_cols = ["asset_id", *raw_features, *engineered]
    out[model_cols] = out[model_cols].astype("float32")
    out["asset_id"] = out["asset_id"].astype("int8")
    return out, model_cols
