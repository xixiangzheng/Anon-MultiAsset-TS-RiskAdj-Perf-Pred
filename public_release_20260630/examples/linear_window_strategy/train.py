from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the linear rolling-window demo strategy.")
    parser.add_argument("--release-root", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--valid-time-fraction", type=float, default=0.2)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--window-size", type=int, default=5)
    return parser.parse_args()


def _manifest_files(release_root: Path, key: str) -> list[Path]:
    manifest_path = release_root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest.get("files", {}).get(key, [])
        if files:
            return [release_root / str(file) for file in files]
    return sorted((release_root / key).glob("*.parquet"))


def load_train_frame(release_root: Path) -> pd.DataFrame:
    files = _manifest_files(release_root, "train")
    if not files:
        raise ValueError(f"no train parquet files found under {release_root}")
    return pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)


def time_series_split(frame: pd.DataFrame, valid_time_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_times = frame["time_id"].drop_duplicates().sort_values(kind="mergesort").to_numpy()
    if len(unique_times) < 2:
        raise ValueError("at least two time_id values are required for validation split")
    valid_count = max(1, int(round(len(unique_times) * valid_time_fraction)))
    valid_count = min(valid_count, len(unique_times) - 1)
    valid_times = set(unique_times[-valid_count:].tolist())
    valid = frame.loc[frame["time_id"].isin(valid_times)].copy()
    train = frame.loc[~frame["time_id"].isin(valid_times)].copy()
    return train, valid


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [col for col in frame.columns if col.startswith("feature_")]


def rolling_mean_by_asset(values: np.ndarray, asset_ids: np.ndarray, window_size: int) -> np.ndarray:
    rolling = np.empty_like(values, dtype=np.float64)
    boundaries = np.flatnonzero(np.r_[True, asset_ids[1:] != asset_ids[:-1], True])
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        group = values[start:end]
        valid = np.isfinite(group)
        clean = np.where(valid, group, 0.0)
        cumsum = np.vstack([np.zeros((1, clean.shape[1]), dtype=np.float64), np.cumsum(clean, axis=0)])
        counts = np.vstack([np.zeros((1, clean.shape[1]), dtype=np.float64), np.cumsum(valid, axis=0)])
        offsets = np.arange(end - start)
        window_starts = np.maximum(0, offsets + 1 - window_size)
        sums = cumsum[offsets + 1] - cumsum[window_starts]
        count = counts[offsets + 1] - counts[window_starts]
        rolling[start:end] = np.divide(sums, count, out=np.zeros_like(sums), where=count > 0)
    return rolling


def build_training_matrix(frame: pd.DataFrame, columns: list[str], window_size: int) -> pd.DataFrame:
    asset_ids = frame["asset_id"].to_numpy(copy=False)
    time_ids = frame["time_id"].to_numpy(copy=False)
    order = np.lexsort((time_ids, asset_ids))
    ordered_asset_ids = asset_ids[order]
    values = frame.loc[:, columns].to_numpy(dtype=np.float64, copy=True)[order]
    rolling = rolling_mean_by_asset(values, ordered_asset_ids, window_size)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    rolling = np.nan_to_num(rolling, nan=0.0, posinf=0.0, neginf=0.0)
    matrix = np.empty((len(frame), len(columns) * 2), dtype=np.float64)
    matrix[order, : len(columns)] = values
    matrix[order, len(columns) :] = rolling
    matrix_columns = [*columns, *[f"roll_mean_{col}" for col in columns]]
    return pd.DataFrame(matrix, index=frame.index, columns=matrix_columns)


def fit_standardizer(matrix: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    mean = matrix.mean(axis=0).to_numpy(dtype=np.float64)
    scale = matrix.std(axis=0, ddof=0).to_numpy(dtype=np.float64)
    scale[~np.isfinite(scale) | (scale < 1e-12)] = 1.0
    return mean, scale


def fit_ridge(matrix: pd.DataFrame, target: np.ndarray, weight: np.ndarray, alpha: float) -> Ridge:
    model = Ridge(alpha=alpha, solver="lsqr")
    model.fit(
        matrix.to_numpy(dtype=np.float64),
        target,
        sample_weight=np.maximum(weight, 0.0),
    )
    return model


def weighted_zero_mean_r2(y_true: np.ndarray, y_pred: np.ndarray, weight: np.ndarray) -> float:
    denominator = np.sum(weight * y_true * y_true)
    if denominator <= 0:
        return 0.0
    numerator = np.sum(weight * (y_true - y_pred) ** 2)
    return float(1.0 - numerator / denominator)


def main() -> None:
    args = parse_args()
    if args.window_size <= 0:
        raise ValueError("window-size must be positive")

    release_root = Path(args.release_root)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    frame = load_train_frame(release_root)
    columns = feature_columns(frame)
    if not columns:
        raise ValueError("no feature_* columns found")
    train, valid = time_series_split(frame, args.valid_time_fraction)

    x_train_raw = build_training_matrix(train, columns, args.window_size)
    mean, scale = fit_standardizer(x_train_raw)
    x_train = (x_train_raw - mean) / scale
    y_train = pd.to_numeric(train["target"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    w_train = pd.to_numeric(train["weight"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    model = fit_ridge(x_train, y_train, w_train, args.ridge_alpha)
    coef = np.concatenate([[float(model.intercept_)], model.coef_.astype(np.float64)])
    pred_train = model.predict(x_train.to_numpy(dtype=np.float64))
    train_score = weighted_zero_mean_r2(y_train, pred_train, w_train)

    x_valid_raw = build_training_matrix(valid, columns, args.window_size)
    x_valid = (x_valid_raw - mean) / scale
    y_valid = pd.to_numeric(valid["target"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    w_valid = pd.to_numeric(valid["weight"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    pred_valid = model.predict(x_valid.to_numpy(dtype=np.float64))
    valid_score = weighted_zero_mean_r2(y_valid, pred_valid, w_valid)

    payload = {
        "strategy": "linear_window_strategy",
        "estimator": "sklearn.linear_model.Ridge",
        "ridge_solver": "lsqr",
        "feature_columns": columns,
        "derived_columns": x_train_raw.columns.tolist(),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coef": coef.tolist(),
        "window_size": int(args.window_size),
        "ridge_alpha": float(args.ridge_alpha),
        "train_rows": int(len(train)),
        "valid_rows": int(len(valid)),
        "train_score": train_score,
        "valid_score": valid_score,
    }
    (model_dir / "linear_model.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ["strategy", "train_rows", "valid_rows", "train_score", "valid_score"]}, indent=2))


if __name__ == "__main__":
    main()
