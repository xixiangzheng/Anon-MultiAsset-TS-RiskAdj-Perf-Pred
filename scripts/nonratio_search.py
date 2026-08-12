"""扩展特征搜索：在 ratio 之外加 diff(A-B), product(A*B), 用 top-10 分母（之前 top-5）。

目的：挖掘 ratio 之外的新信号源。
对每对 top-50 特征，计算 diff / product / sum / max / min，找与 target 相关最高的。
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd

STRAT = "/mnt/iscsi/hd/xxz/public_release_lightgbm_baseline/examples/lightgbm_baseline"
if STRAT not in sys.path: sys.path.insert(0, STRAT)
from data_utils import manifest_files, feature_columns_from_path
DATA_ROOT = Path("/mnt/iscsi/hd/xxz/public_release_20260630/data")
RUN = Path("/mnt/iscsi/hd/xxz/runs")


def wcorr(x, y, w):
    wm = (w*x).sum()/w.sum(); ym = (w*y).sum()/w.sum()
    num = (w*(x-wm)*(y-ym)).sum()
    den = np.sqrt((w*(x-wm)**2).sum()*(w*(y-ym)**2).sum())
    return abs(num/den) if den > 0 else 0


def clip01(x):
    x = np.nan_to_num(x, nan=0, posinf=0, neginf=0)
    lo, hi = np.percentile(x, [1, 99])
    return np.clip(x, lo, hi)


def main():
    paths = manifest_files(DATA_ROOT, "train")[:2]
    feats = feature_columns_from_path(paths[0])
    pf = pd.read_parquet(paths, columns=["weight", "target"] + feats)
    pf[feats] = np.nan_to_num(pf[feats].to_numpy(np.float32))
    print(f"loaded {len(pf):,} rows", flush=True)
    y = pd.to_numeric(pf["target"], errors="coerce").fillna(0).to_numpy(np.float64)
    w = pd.to_numeric(pf["weight"], errors="coerce").fillna(0).clip(lower=0).to_numpy(np.float64)
    F = pf[feats].to_numpy(np.float64)

    sc = np.array([wcorr(F[:, i], y, w) for i in range(len(feats))])
    top50 = np.argsort(-sc)[:50].tolist()
    print(f"top-5 single: {[(feats[i], round(float(sc[i]),4)) for i in np.argsort(-sc)[:5]]}", flush=True)

    # 已有 ratio_top50
    rs = json.loads((RUN/"ratio_top50.json").read_text())["ratios"]
    ratio_set = set((feats.index(r[2]), feats.index(r[3])) for r in rs)
    print(f"existing ratios: {len(ratio_set)}", flush=True)

    inter = []
    t0 = time.time()
    # top-50 × top-50 两两交互（除了已有的 ratio）
    for i_idx, i in enumerate(top50[:30]):  # top-30 数值特征
        for j in top50[i_idx+1:30]:  # top-30 配对
            fi, fj = F[:, i], F[:, j]
            # diff (i - j)
            d = clip01(fi - fj)
            inter.append((f"{feats[i]}-{feats[j]}", wcorr(d, y, w), "diff", feats[i], feats[j]))
            # product
            p = clip01(fi * fj)
            inter.append((f"{feats[i]}*{feats[j]}", wcorr(p, y, w), "product", feats[i], feats[j]))
            # sum
            s = clip01(fi + fj)
            inter.append((f"{feats[i]}+{feats[j]}", wcorr(s, y, w), "sum", feats[i], feats[j]))
            # max
            mx = np.maximum(fi, fj)
            inter.append((f"max({feats[i]},{feats[j]})", wcorr(mx, y, w), "max", feats[i], feats[j]))
    inter.sort(key=lambda x: -x[1])
    print(f"\n=== Top 30 非ratio交互 (top-30×top-30) ({time.time()-t0:.0f}s) ===", flush=True)
    for name, c, typ, *_ in inter[:30]:
        print(f"  {c:.4f}  {name} ({typ})", flush=True)
    print(f"\n单特征最高: {sc[top50[0]]:.4f}")
    print(f"非ratio交互最高: {inter[0][1]:.4f} ({inter[0][0]})")

    # 看哪些非 ratio 交互比 top-5 单特征更强
    n_strong = sum(1 for _, c, *_ in inter if c > 0.032)
    print(f"\n非ratio交互中 >0.032 相关: {n_strong} 个", flush=True)

    # 保存 top-30 非ratio 交互（去重，每个类型留几个）
    by_type = {}
    for name, c, typ, a, b in inter:
        if typ not in by_type: by_type[typ] = []
        if len(by_type[typ]) < 15:
            by_type[typ].append({"name": name, "corr": float(c), "type": typ, "a": a, "b": b})
    json.dump({"top_interactions": inter[:50], "by_type": by_type},
              open(RUN/"nonratio_interactions.json", "w"), indent=2)
    print(f"\nsaved runs/nonratio_interactions.json", flush=True)


if __name__ == "__main__":
    main()
