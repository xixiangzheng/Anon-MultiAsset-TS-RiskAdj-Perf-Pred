"""P0-2: 用回补标签复算 submissions/*.csv 全部真实分数 → 完整排行榜。"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path("/mnt/iscsi/hd/xxz")
LABELS = ROOT / "public_release_20260823/public_release_20260823/data/train"
SUB = ROOT / "submissions"
OUT = ROOT / "runs" / "backfill_leaderboard.csv"

labels = pd.concat(
    [pd.read_parquet(p, columns=["row_id", "weight", "target"]) for p in sorted(LABELS.glob("*.parquet"))],
    ignore_index=True,
)
print(f"labels: {len(labels):,} rows", flush=True)

rows = []
csvs = sorted(SUB.glob("*.csv"))
print(f"scoring {len(csvs)} submissions...", flush=True)
for i, f in enumerate(csvs):
    try:
        pred = pd.read_csv(f)
        merged = labels.merge(pred.rename(columns={"target": "prediction"}), on="row_id", how="left")
        p = pd.to_numeric(merged["prediction"], errors="coerce")
        p = p.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float64)
        y = merged["target"].to_numpy(np.float64)
        w = merged["weight"].to_numpy(np.float64)
        score = 1.0 - np.sum(w * (y - p) ** 2) / np.sum(w * y * y)
        rows.append({"file": f.name, "backfill_r2": float(score), "n_missing": int(merged["prediction"].isna().sum())})
        print(f"  [{i+1}/{len(csvs)}] {f.name}: {score:+.8f} (missing={rows[-1]['n_missing']})", flush=True)
    except Exception as e:
        print(f"  [{i+1}/{len(csvs)}] {f.name}: ERROR {e}", flush=True)

df = pd.DataFrame(rows).sort_values("backfill_r2", ascending=False).reset_index(drop=True)
df.to_csv(OUT, index=False)
print(f"\n===== 回补真值排行榜（top 25） =====", flush=True)
print(df.head(25).to_string(index=False), flush=True)
print(f"\nsaved {OUT}", flush=True)
