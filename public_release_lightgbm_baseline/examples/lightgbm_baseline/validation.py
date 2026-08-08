from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TimeFold:
    fold_id: int
    train_time_ids: np.ndarray
    valid_time_ids: np.ndarray


@dataclass(frozen=True)
class ValidationPlan:
    folds: tuple[TimeFold, ...]
    development_time_ids: np.ndarray
    holdout_time_ids: np.ndarray
    purge_steps: int
    cv_scheme: str = "purged_kfold"


def make_validation_plan(
    time_ids,
    *,
    n_splits: int = 5,
    holdout_fraction: float = 0.15,
    purge_steps: int = 30,
) -> ValidationPlan:
    """Equal-block purged K-fold on development times; trailing holdout untouched.

    For each fold, valid is one block; train is the other blocks with a symmetric
    purge/embargo of ``purge_steps`` time_ids on both sides of the valid block.
    """
    unique = np.unique(np.asarray(time_ids, dtype=np.int64))
    if len(unique) < n_splits + 2:
        raise ValueError("not enough unique time_id values for validation")
    if not 0.0 < holdout_fraction < 0.5:
        raise ValueError("holdout_fraction must be between 0 and 0.5")
    if purge_steps < 0:
        raise ValueError("purge_steps must be non-negative")
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")

    holdout_count = max(1, int(np.ceil(len(unique) * holdout_fraction)))
    development, holdout = unique[:-holdout_count], unique[-holdout_count:]
    if len(development) < n_splits:
        raise ValueError("development time_ids are fewer than n_splits")

    blocks = tuple(np.asarray(block, dtype=np.int64) for block in np.array_split(development, n_splits))
    folds: list[TimeFold] = []
    for fold_id in range(n_splits):
        valid = blocks[fold_id]
        if valid.size == 0:
            raise ValueError("validation plan produced an empty valid block")
        train_parts = [blocks[idx] for idx in range(n_splits) if idx != fold_id]
        candidate_train = np.concatenate(train_parts)
        v_min = int(valid.min())
        v_max = int(valid.max())
        keep = (candidate_train <= v_min - purge_steps - 1) | (candidate_train >= v_max + purge_steps + 1)
        train = candidate_train[keep]
        if train.size == 0:
            raise ValueError("validation plan produced an empty train fold after purge")
        folds.append(TimeFold(fold_id, np.sort(train), valid))
    return ValidationPlan(tuple(folds), development, holdout, int(purge_steps), cv_scheme="purged_kfold")


def weighted_zero_mean_r2(y_true, y_pred, weight) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_pred, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    denominator = float(np.sum(w * y * y))
    if denominator <= 0.0 or not np.isfinite(denominator):
        return 0.0
    return float(1.0 - np.sum(w * (y - p) ** 2) / denominator)


def fit_prediction_scale(y_true, prediction, weight) -> float:
    """Closed-form amplitude diagnostic; must not be applied as a calibrator."""
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    denominator = float(np.sum(w * p * p))
    if denominator <= 0.0 or not np.isfinite(denominator):
        return 1.0
    scale = float(np.sum(w * y * p) / denominator)
    return scale if np.isfinite(scale) else 1.0


def evaluate_gates(
    *,
    oof_raw_score: float,
    holdout_raw_score: float,
    fitted_oof_scale: float,
    scale_low: float = 0.75,
    scale_high: float = 1.25,
) -> dict:
    checks = {
        "oof_raw_positive": bool(oof_raw_score > 0.0),
        "holdout_raw_positive": bool(holdout_raw_score > 0.0),
        "scale_in_range": bool(scale_low <= fitted_oof_scale <= scale_high),
    }
    return {
        **checks,
        "gates_passed": bool(all(checks.values())),
        "scale_low": scale_low,
        "scale_high": scale_high,
        "prediction_scale_applied": 1.0,
        "fitted_oof_scale": float(fitted_oof_scale),
    }
