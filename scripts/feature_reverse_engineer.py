"""特征逆向工程：系统搜索两两交互(乘积/比率)。

假设：某些特征的比率(如收益率/波动率=类Sharpe)可能与target高度相关。
对 top-20 相关特征，计算所有两两 × 和 ÷ (380 个交互)，排序找最有价值的。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

STRAT="/mnt/iscsi/hd/xxz/public_release_lightgbm_baseline/examples/lightgbm_baseline"
if STRAT not in sys.path: sys.path.insert(0,STRAT)
from data_utils import manifest_files, feature_columns_from_path  # noqa: E402
DATA_ROOT=Path("/mnt/iscsi/hd/xxz/public_release_20260630/data")


def wcorr(x, y, w):
    wm = (w * x).sum() / w.sum()
    ym = (w * y).sum() / w.sum()
    num = (w * (x - wm) * (y - ym)).sum()
    den = np.sqrt((w * (x - wm) ** 2).sum() * (w * (y - ym) ** 2).sum())
    return abs(num / den) if den > 0 else 0


def main():
    paths = manifest_files(DATA_ROOT, "train")[:2]
    feats = feature_columns_from_path(paths[0])
    pf = pd.read_parquet(paths, columns=["weight", "target"] + feats)
    pf[feats] = np.nan_to_num(pf[feats].to_numpy(np.float32))
    print(f"loaded {len(pf):,} rows", flush=True)
    y = pd.to_numeric(pf["target"], errors="coerce").fillna(0).to_numpy(np.float64)
    w = pd.to_numeric(pf["weight"], errors="coerce").fillna(0).clip(lower=0).to_numpy(np.float64)
    F = pf[feats].to_numpy(np.float64)

    # 单特征加权相关排序
    sc = [(feats[i], wcorr(F[:, i], y, w)) for i in range(len(feats))]
    sc.sort(key=lambda t: -t[1])
    top = [f for f, _ in sc[:20]]
    top_idx = [feats.index(f) for f in top]
    print(f"top-20 特征相关: {[(f, round(s, 4)) for f, s in sc[:20]]}", flush=True)

    # 两两交互(乘积 + 比率)
    interactions = []
    for i in range(20):
        for j in range(i + 1, 20):
            fi, fj = F[:, top_idx[i]], F[:, top_idx[j]]
            # 乘积
            prod = fi * fj
            c_prod = wcorr(np.nan_to_num(prod), y, w)
            interactions.append((f"{top[i]}*{top[j]}", c_prod, "product"))
            # 比率 a/b (避免除零)
            ratio = np.where(np.abs(fj) > 1e-8, fi / fj, 0)
            ratio = np.nan_to_num(ratio, nan=0, posinf=0, neginf=0)
            # clip extreme
            lo, hi = np.percentile(ratio, [1, 99])
            ratio = np.clip(ratio, lo, hi)
            c_ratio = wcorr(ratio, y, w)
            interactions.append((f"{top[i]}/{top[j]}", c_ratio, "ratio"))
            # 比率 b/a
            ratio2 = np.where(np.abs(fi) > 1e-8, fj / fi, 0)
            ratio2 = np.nan_to_num(ratio2, nan=0, posinf=0, neginf=0)
            lo2, hi2 = np.percentile(ratio2, [1, 99])
            ratio2 = np.clip(ratio2, lo2, hi2)
            c_ratio2 = wcorr(ratio2, y, w)
            interactions.append((f"{top[j]}/{top[i]}", c_ratio2, "ratio"))

    interactions.sort(key=lambda t: -t[1])
    print(f"\n=== Top 30 交互特征(相关 with target) ===", flush=True)
    for name, corr, typ in interactions[:30]:
        print(f"  {corr:.4f}  {name} ({typ})", flush=True)
    print(f"\n单特征最高相关: {sc[0][1]:.4f} ({sc[0][0]})", flush=True)
    print(f"交互最高相关: {interactions[0][1]:.4f} ({interactions[0][0]})", flush=True)


if __name__ == "__main__":
    main()
