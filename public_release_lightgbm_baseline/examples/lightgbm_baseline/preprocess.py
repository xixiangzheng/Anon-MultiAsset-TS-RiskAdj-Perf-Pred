from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PreprocessSpec:
    raw_features: tuple[str, ...]


def sanitize_feature_frame(frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = frame.copy()
    if not feature_cols:
        return out
    values = out.loc[:, feature_cols].to_numpy(dtype=np.float32, copy=True)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    out.loc[:, feature_cols] = values
    return out


def fit_feature_schema(
    frame: pd.DataFrame,
    feature_cols: list[str],
    *,
    min_finite_ratio: float = 0.01,
) -> PreprocessSpec:
    """One-shot health check on early train; freeze surviving raw feature columns."""
    if not feature_cols:
        raise ValueError("feature_cols must be non-empty")
    kept: list[str] = []
    n_rows = max(len(frame), 1)
    for col in feature_cols:
        series = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        finite = series.notna()
        if float(finite.mean()) < min_finite_ratio:
            continue
        values = series.fillna(0.0).to_numpy(dtype=np.float64)
        if np.nanmax(values) == np.nanmin(values):
            continue
        kept.append(col)
    if not kept:
        raise ValueError("no usable feature columns after health check")
    return PreprocessSpec(raw_features=tuple(kept))


def apply_preprocess(frame: pd.DataFrame, spec: PreprocessSpec) -> pd.DataFrame:
    cols = list(spec.raw_features)
    missing = [col for col in cols if col not in frame.columns]
    if missing:
        raise ValueError(f"frame missing frozen features: {missing[:5]}")
    return sanitize_feature_frame(frame, cols)
