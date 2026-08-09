"""responder 与 target 的加权相关性分析。

加载 1 个训练分区（含 responder_*），计算每个 responder 与 target 的加权 Pearson 相关，
排序输出 top-K。用于判断哪些 responder 携带 target 相关信号（stacking P0 第一步）。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path("/mnt/nv1/home/hexin/xi_workspace/E2E-Markowitz-Toy")
DATA_ROOT = Path("/mnt/iscsi/hd/xxz/public_release_20260630/data")
N_RESP = 47


def wmean(a: np.ndarray, w: np.ndarray) -> float:
    s = w.sum()
    return float((w * a).sum() / s) if s > 0 else float(a.mean())


def wcorr(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    xm, ym = wmean(x, w), wmean(y, w)
    xa, ya = x - xm, y - ym
    num = float((w * xa * ya).sum())
    den = float(np.sqrt((w * xa * xa).sum() * (w * ya * ya).sum()))
    return num / den if den > 0 else 0.0


def main() -> None:
    manifest = json.loads((DATA_ROOT / "manifest.json").read_text(encoding="utf-8"))
    train_file = DATA_ROOT / manifest["files"]["train"][0]
    cols = ["time_id", "weight", "target"] + [f"responder_{i:02d}" for i in range(N_RESP)]
    pf = pd.read_parquet(train_file, columns=cols)
    print(f"loaded {len(pf):,} rows from {train_file.name}")

    y = pd.to_numeric(pf["target"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    w = pd.to_numeric(pf["weight"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(np.float64)
    print(f"target weighted variance: {(w * (y - wmean(y, w)) ** 2).sum() / w.sum():.6g}")

    rows = []
    for i in range(N_RESP):
        col = f"responder_{i:02d}"
        x = (
            pd.to_numeric(pf[col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(np.float64)
        )
        c = wcorr(x, y, w)
        rows.append((col, c))
    rows.sort(key=lambda t: -abs(t[1]))

    print("\n=== Top 15 responder by |weighted corr with target| ===")
    for col, c in rows[:15]:
        print(f"  {col}: {c:+.4f}")
    print("\n=== All 47 (sorted by |corr|) ===")
    for col, c in rows:
        print(f"  {col}: {c:+.4f}")


if __name__ == "__main__":
    main()
