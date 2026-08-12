"""三方交互特征搜索：(A/B)*C 与 A*(B+C) 等组合，看是否超越两两 ratio。

策略：
1. 用 top-50 ratio（runs/ratio_top50.json，由 ratio_models.py 产出，或现算）作为 (A/B) 基础
2. 对每个 ratio，找最佳的 C（top-30 单特征中），形成 (A/B)*C 三方交互
3. 评估与 target 的加权相关，与原 ratio 对比
4. 输出 top-30 三方交互供训练使用

仅在 part0+1 上搜索，省时。
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd

STRAT = "/mnt/iscsi/hd/xxz/public_release_lightgbm_baseline/examples/lightgbm_baseline"
if STRAT not in sys.path: sys.path.insert(0, STRAT)
from data_utils import manifest_files, feature_columns_from_path  # noqa: E402
DATA_ROOT = Path("/mnt/iscsi/hd/xxz/public_release_20260630/data")
RUN = Path("/mnt/iscsi/hd/xxz/runs")


def wcorr(x, y, w):
    wm = (w * x).sum() / w.sum(); ym = (w * y).sum() / w.sum()
    num = (w * (x - wm) * (y - ym)).sum()
    den = np.sqrt((w * (x - wm) ** 2).sum() * (w * (y - ym) ** 2).sum())
    return abs(num / den) if den > 0 else 0


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
    top_idx = np.argsort(-sc)
    top30 = top_idx[:30].tolist()
    top50 = top_idx[:50].tolist()
    print(f"top-5 single corr: {[(feats[i], round(float(sc[i]),4)) for i in top_idx[:5]]}", flush=True)

    # 已有 ratio_top50.json
    if (RUN/"ratio_top50.json").exists():
        rs = json.loads((RUN/"ratio_top50.json").read_text())["ratios"]
        ratio_pairs = [(feats.index(r[2]), feats.index(r[3])) for r in rs]
        print(f"loaded {len(ratio_pairs)} ratios from runs/ratio_top50.json", flush=True)
    else:
        # 现算 top ratio（同 new_model 逻辑）
        print("no ratio_top50.json; computing...", flush=True)
        denom_scores = []
        for d in range(len(feats)):
            fd = F[:, d]
            if fd.min() < 0: continue
            fd_s = np.clip(fd, 1e-8, np.percentile(fd, 99))
            imps = []
            for ni in top50[:10]:
                r = clip01(F[:, ni] / fd_s)
                imps.append(wcorr(r, y, w) - sc[ni])
            denom_scores.append((d, float(np.mean(imps))))
        denom_scores.sort(key=lambda x: -x[1])
        best_d = [d for d, _ in denom_scores[:5]]
        inter = []
        for d in best_d:
            fd = np.clip(F[:, d], 1e-8, np.percentile(F[:, d], 99))
            for ni in top50:
                r = clip01(F[:, ni] / fd)
                inter.append((ni, d, wcorr(r, y, w)))
        inter.sort(key=lambda x: -x[2])
        ratio_pairs = [(ni, di) for ni, di, _ in inter[:50]]
        print(f"computed {len(ratio_pairs)} ratios", flush=True)

    # 基线：top-50 ratio 自身的 corr
    ratio_corrs = []
    for ni, di in ratio_pairs:
        fd = np.clip(F[:, di], 1e-8, np.percentile(F[:, di], 99))
        r = clip01(F[:, ni] / fd)
        ratio_corrs.append(wcorr(r, y, w))
    print(f"ratio top-5 corr: {[(i, round(c,4)) for i, c in enumerate(ratio_corrs[:5])][:5]}", flush=True)
    print(f"ratio mean corr: {np.mean(ratio_corrs):.4f}", flush=True)

    # 三方：(ratio_i) * feature_C  ——对每个 ratio 找最佳 C
    print("\n=== 三方交互搜索 (ratio * top-30 单特征) ===", flush=True)
    t0 = time.time()
    three_way = []
    for ri, (ni, di) in enumerate(ratio_pairs[:30]):  # top-30 ratio
        fd = np.clip(F[:, di], 1e-8, np.percentile(F[:, di], 99))
        ratio_v = clip01(F[:, ni] / fd)
        base_corr = wcorr(ratio_v, y, w)
        # 对 top-30 单特征，试 ratio*C
        best_c = None; best_corr = base_corr; best_delta = 0
        for ci in top30:
            if ci == ni or ci == di: continue
            v = clip01(ratio_v * F[:, ci])
            c = wcorr(v, y, w)
            if c > best_corr:
                best_corr = c; best_c = ci
        if best_c is not None:
            best_delta = best_corr - base_corr
            three_way.append({
                "type": "ratio_mul", "ratio_idx": ri, "numer": feats[ni], "denom": feats[di],
                "third": feats[best_c], "base_corr": float(base_corr),
                "new_corr": float(best_corr), "delta": float(best_delta)
            })
    three_way.sort(key=lambda x: -x["delta"])
    print(f"done {time.time()-t0:.0f}s", flush=True)
    print(f"\ntop-10 三方 (ratio*C)：")
    for t in three_way[:10]:
        print(f"  +{t['delta']:.5f} ({t['base_corr']:.4f}→{t['new_corr']:.4f})  ({t['numer']}/{t['denom']})*{t['third']}", flush=True)

    n_pos = sum(1 for t in three_way if t["delta"] > 0)
    n_sig = sum(1 for t in three_way if t["delta"] > 0.001)
    print(f"\n三方分析: {n_pos}/{len(three_way)} 有正向 delta，{n_sig} 个 >0.001", flush=True)
    if n_sig > 0:
        print(f"→ 三方交互有 {n_sig} 个显著正向，值得加入训练！", flush=True)
    else:
        print(f"→ 三方交互无显著正向，ratio 已捕获大部分信号", flush=True)

    # 保存
    json.dump({"three_way": three_way, "n_significant": n_sig},
              open(RUN/"three_way_search.json", "w"), indent=2)
    print(f"\nsaved to runs/three_way_search.json", flush=True)


if __name__ == "__main__":
    main()
