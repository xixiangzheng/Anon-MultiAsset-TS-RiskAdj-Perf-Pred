from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def manifest_files(release_root: str | Path, key: str) -> list[Path]:
    release_root = Path(release_root)
    manifest_path = release_root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest.get("files", {}).get(key, [])
        if files:
            return [release_root / str(file) for file in files]
    return sorted((release_root / key).glob("*.parquet"))


def feature_columns_from_path(path: str | Path) -> list[str]:
    import pyarrow.parquet as pq

    columns = pq.ParquetFile(path).schema_arrow.names
    return [col for col in columns if str(col).startswith("feature_")]


def load_parquet_files(paths: list[Path], columns: list[str] | None = None) -> pd.DataFrame:
    if not paths:
        raise ValueError("no parquet files provided")
    return pd.concat((pd.read_parquet(path, columns=columns) for path in paths), ignore_index=True)


def load_train_frame(release_root: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    return load_parquet_files(manifest_files(release_root, "train"), columns=columns)


def sample_by_time(frame: pd.DataFrame, max_rows: int, *, seed: int) -> pd.DataFrame:
    if max_rows <= 0 or len(frame) <= max_rows:
        return frame
    rng = np.random.default_rng(seed)
    times = frame["time_id"].drop_duplicates().to_numpy()
    rows_per_time = max(len(frame) / max(len(times), 1), 1.0)
    n_times = max(1, min(len(times), int(max_rows / rows_per_time)))
    chosen = np.sort(rng.choice(times, size=n_times, replace=False))
    sampled = frame.loc[frame["time_id"].isin(chosen)].copy()
    if len(sampled) > max_rows:
        sampled = sampled.sample(n=max_rows, random_state=seed).sort_values(["time_id", "asset_id"])
    return sampled.reset_index(drop=True)


def top_correlated_features(
    frame: pd.DataFrame,
    feature_cols: list[str],
    *,
    top_k: int,
    sample_rows: int,
    seed: int,
) -> list[str]:
    if top_k <= 0 or top_k >= len(feature_cols):
        return feature_cols
    sample = sample_by_time(
        frame.loc[:, [*feature_cols, "target", "weight", "time_id", "asset_id"]],
        sample_rows,
        seed=seed,
    )
    y = pd.to_numeric(sample["target"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    w = pd.to_numeric(sample["weight"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    w = np.maximum(w, 0.0)
    y_mean = np.average(y, weights=w) if w.sum() > 0 else float(np.mean(y))
    y_centered = y - y_mean
    y_scale = np.sqrt(np.average(y_centered * y_centered, weights=w)) if w.sum() > 0 else float(np.std(y_centered))
    scores: list[tuple[float, str]] = []
    for col in feature_cols:
        x = (
            pd.to_numeric(sample[col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )
        x_mean = np.average(x, weights=w) if w.sum() > 0 else float(np.mean(x))
        x_centered = x - x_mean
        x_scale = np.sqrt(np.average(x_centered * x_centered, weights=w)) if w.sum() > 0 else float(np.std(x_centered))
        if x_scale <= 1e-12 or y_scale <= 1e-12:
            score = 0.0
        else:
            cov = np.average(x_centered * y_centered, weights=w) if w.sum() > 0 else float(np.mean(x_centered * y_centered))
            score = abs(float(cov / (x_scale * y_scale)))
        scores.append((score, col))
    scores.sort(reverse=True)
    return [col for _, col in scores[:top_k]]


def add_group_history_features(
    frame: pd.DataFrame,
    selected_features: list[str],
    *,
    rolling_window: int = 5,
    rolling_windows: tuple[int, ...] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    if not selected_features:
        return frame.copy(), []
    windows = tuple(rolling_windows) if rolling_windows is not None else (int(rolling_window),)
    if not windows or any(int(window) < 1 for window in windows):
        raise ValueError("rolling windows must be positive integers")
    windows = tuple(dict.fromkeys(int(window) for window in windows))

    out = frame.copy()
    order = np.lexsort((out["time_id"].to_numpy(), out["asset_id"].to_numpy()))
    restore = np.empty(len(order), dtype=np.int64)
    restore[order] = np.arange(len(order))
    asset_ids = out["asset_id"].to_numpy()[order]
    values = out.loc[:, selected_features].to_numpy(dtype=np.float32, copy=True)[order]
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    lag = np.zeros_like(values, dtype=np.float32)
    rollings = {window: np.zeros_like(values, dtype=np.float32) for window in windows}
    boundaries = np.flatnonzero(np.r_[True, asset_ids[1:] != asset_ids[:-1], True])
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        group = values[start:end]
        if len(group) > 1:
            lag[start + 1 : end] = group[:-1]
        cumsum = np.vstack([np.zeros((1, group.shape[1]), dtype=np.float64), np.cumsum(group, axis=0)])
        offsets = np.arange(end - start)
        for window in windows:
            window_starts = np.maximum(0, offsets + 1 - window)
            sums = cumsum[offsets + 1] - cumsum[window_starts]
            counts = (offsets + 1 - window_starts).reshape(-1, 1)
            rollings[window][start:end] = (sums / counts).astype(np.float32)

    lag = lag[restore]
    rollings = {window: rolling[restore] for window, rolling in rollings.items()}
    diff = out.loc[:, selected_features].to_numpy(dtype=np.float32, copy=True)
    diff = np.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0) - lag

    new_cols: list[str] = []
    blocks: list[np.ndarray] = []
    for values_block, prefix in ((lag, "lag1"), (diff, "diff1")):
        cols = [f"{prefix}_{col}" for col in selected_features]
        new_cols.extend(cols)
        blocks.append(values_block.astype(np.float32, copy=False))
    for window in windows:
        cols = [f"rmean{window}_{col}" for col in selected_features]
        new_cols.extend(cols)
        blocks.append(rollings[window].astype(np.float32, copy=False))
    engineered = pd.DataFrame(np.hstack(blocks), columns=new_cols, index=out.index)
    return pd.concat([out, engineered], axis=1), new_cols
