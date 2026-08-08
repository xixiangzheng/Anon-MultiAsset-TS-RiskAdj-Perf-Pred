from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the random baseline strategy.")
    parser.add_argument("--release-root", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--valid-time-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
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


def weighted_zero_mean_r2(y_true: np.ndarray, y_pred: np.ndarray, weight: np.ndarray) -> float:
    denominator = np.sum(weight * y_true * y_true)
    if denominator <= 0:
        return 0.0
    numerator = np.sum(weight * (y_true - y_pred) ** 2)
    return float(1.0 - numerator / denominator)


def main() -> None:
    args = parse_args()
    release_root = Path(args.release_root)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    frame = load_train_frame(release_root)
    train, valid = time_series_split(frame, args.valid_time_fraction)
    target = pd.to_numeric(train["target"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    weight = pd.to_numeric(train["weight"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)

    weighted_var = np.average(target * target, weights=np.maximum(weight, 0.0)) if weight.sum() > 0 else np.mean(target * target)
    prediction_scale = float(max(np.sqrt(max(weighted_var, 0.0)) * 0.05, 1e-6))

    valid_pred = np.zeros(len(valid), dtype=np.float64)
    valid_score = weighted_zero_mean_r2(
        pd.to_numeric(valid["target"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64),
        valid_pred,
        pd.to_numeric(valid["weight"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64),
    )
    payload = {
        "strategy": "random_strategy",
        "seed": int(args.seed),
        "prediction_scale": prediction_scale,
        "train_rows": int(len(train)),
        "valid_rows": int(len(valid)),
        "valid_zero_score": valid_score,
    }
    (model_dir / "random_model.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
