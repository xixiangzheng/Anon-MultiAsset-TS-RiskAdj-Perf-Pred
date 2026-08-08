from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read, process, and summarize a release split with polars.")
    parser.add_argument("--data-root", default=str(Path(__file__).resolve().parents[2] / "data"))
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--row-limit", type=int, default=10000)
    parser.add_argument("--output", default="/tmp/quantcontest_polars_summary.json")
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


def skipped_payload(message: str) -> dict[str, object]:
    return {"status": "skipped", "engine": "polars", "message": message}


def main() -> None:
    args = parse_args()
    if args.row_limit <= 0:
        raise ValueError("row-limit must be positive")

    try:
        import polars as pl
    except ImportError:
        print(json.dumps(skipped_payload("polars is not installed"), ensure_ascii=False))
        return

    total_start = time.perf_counter()
    files = split_files(args.data_root, args.split)
    if not files:
        raise FileNotFoundError(f"no parquet files found for split={args.split!r}")

    read_start = time.perf_counter()
    frame = pl.concat([pl.scan_parquet(str(path)) for path in files]).head(args.row_limit).collect()
    read_seconds = time.perf_counter() - read_start

    process_start = time.perf_counter()
    feature_cols = [col for col in frame.columns if str(col).startswith("feature_")]
    first_features = feature_cols[:5]
    mean_exprs = [pl.col(col).abs().mean().alias(col) for col in first_features]
    mean_values = frame.select(mean_exprs).to_dicts()[0] if mean_exprs else {}
    payload = {
        "engine": "polars",
        "input": {
            "data_root": str(Path(args.data_root)),
            "split": args.split,
            "row_limit": int(args.row_limit),
        },
        "data": {
            "rows": int(frame.height),
            "columns": int(frame.width),
            "time_id_min": int(frame["time_id"].min()) if "time_id" in frame.columns and frame.height else None,
            "time_id_max": int(frame["time_id"].max()) if "time_id" in frame.columns and frame.height else None,
            "asset_count": int(frame["asset_id"].n_unique()) if "asset_id" in frame.columns else None,
        },
        "features": {
            "count": int(len(feature_cols)),
            "first_columns": first_features,
            "mean_abs_first_columns": {col: float(mean_values[col]) for col in first_features if col in mean_values},
        },
        "output_path": str(Path(args.output)),
    }
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
