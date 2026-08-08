from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read, process, and summarize a release split with pandas.")
    parser.add_argument("--data-root", default=str(Path(__file__).resolve().parents[2] / "data"))
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--row-limit", type=int, default=10000)
    parser.add_argument("--output", default="/tmp/quantcontest_pandas_summary.json")
    return parser.parse_args()


def split_files(data_root: str | Path, split: str) -> list[Path]:
    root = Path(data_root)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest.get("files", {}).get(split, [])
        if files:
            return [root / str(file) for file in files]
    return sorted((root / split).glob("*.parquet"))


def read_split(data_root: str | Path, split: str, row_limit: int) -> pd.DataFrame:
    paths = split_files(data_root, split)
    if not paths:
        raise FileNotFoundError(f"no parquet files found for split={split!r}")
    frames = []
    remaining = row_limit
    for path in paths:
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=min(remaining, 10000)):
            frame = batch.to_pandas()
            frames.append(frame)
            remaining -= len(frame)
            if remaining <= 0:
                break
        if row_limit > 0 and remaining <= 0:
            break
    return pd.concat(frames, ignore_index=True)


def summarize(frame: pd.DataFrame, *, data_root: str | Path, split: str, row_limit: int, output_path: str | Path) -> dict[str, object]:
    feature_cols = [col for col in frame.columns if str(col).startswith("feature_")]
    numeric = frame.loc[:, feature_cols].select_dtypes(include="number")
    first_features = feature_cols[:5]
    return {
        "engine": "pandas",
        "input": {
            "data_root": str(Path(data_root)),
            "split": split,
            "row_limit": int(row_limit),
        },
        "data": {
            "rows": int(len(frame)),
            "columns": int(len(frame.columns)),
            "time_id_min": int(frame["time_id"].min()) if "time_id" in frame.columns and len(frame) else None,
            "time_id_max": int(frame["time_id"].max()) if "time_id" in frame.columns and len(frame) else None,
            "asset_count": int(frame["asset_id"].nunique()) if "asset_id" in frame.columns else None,
        },
        "features": {
            "count": int(len(feature_cols)),
            "first_columns": first_features,
            "mean_abs_first_columns": {
                col: float(numeric[col].abs().mean()) for col in first_features if col in numeric.columns
            },
        },
        "output_path": str(Path(output_path)),
    }


def main() -> None:
    args = parse_args()
    if args.row_limit <= 0:
        raise ValueError("row-limit must be positive")

    total_start = time.perf_counter()
    read_start = time.perf_counter()
    frame = read_split(args.data_root, args.split, args.row_limit)
    read_seconds = time.perf_counter() - read_start

    process_start = time.perf_counter()
    payload = summarize(
        frame,
        data_root=args.data_root,
        split=args.split,
        row_limit=args.row_limit,
        output_path=args.output,
    )
    process_seconds = time.perf_counter() - process_start

    write_start = time.perf_counter()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload["timing"] = {
        "read_seconds": float(read_seconds),
        "process_seconds": float(process_seconds),
        "write_seconds": 0.0,
        "total_seconds": 0.0,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_seconds = time.perf_counter() - write_start
    payload["timing"]["write_seconds"] = float(write_seconds)
    payload["timing"]["total_seconds"] = float(time.perf_counter() - total_start)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
