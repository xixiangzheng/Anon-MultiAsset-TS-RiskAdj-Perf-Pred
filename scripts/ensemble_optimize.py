"""通用集成权重优化：从 oof.pkl 加载各模型 holdout OOF，SLSQP 优化非负权重最大化加权R²。
套用到 test 预测，输出 submission。

用法：python scripts/ensemble_optimize.py <oof.pkl> <out.csv> [--models m1,m2,...] [--exclude m1,m2]
"""
from __future__ import annotations
import sys, pickle, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize

SUB = Path("/mnt/iscsi/hd/xxz/submissions"); RUN = Path("/mnt/iscsi/hd/xxz/runs")


def wr2(y, p, w):
    d = float(np.sum(w*y*y)); return 0.0 if d<=0 else 1-float(np.sum(w*(y-p)**2)/d)


def main():
    oof_path = sys.argv[1]
    out_csv = sys.argv[2]
    include = None; exclude = set()
    for i, a in enumerate(sys.argv):
        if a == "--models" and i+1 < len(sys.argv): include = sys.argv[i+1].split(",")
        if a == "--exclude" and i+1 < len(sys.argv): exclude = set(sys.argv[i+1].split(","))
    if len(sys.argv) > 3 and sys.argv[3].startswith("--"):
        pass  # argparse 简化处理

    d = pickle.load(open(oof_path, "rb"))
    keys = list(d["oofs"].keys())
    if include: keys = [k for k in keys if k in include]
    keys = [k for k in keys if k not in exclude]
    print(f"models: {keys}", flush=True)

    oofs = d["oofs"]; yv = d["yv"]; wv = d["wv"]
    P = np.array([oofs[k] for k in keys])  # [n_models, n_holdout]
    # 单模型
    print("\n=== 单模型 holdout R² ===")
    for k in keys: print(f"  {k}: {wr2(yv, oofs[k], wv):+.6f}")
    # 相关性
    print("\n=== 相关性 ===")
    for ki in keys:
        print(f"  {ki:>14s}: " + " ".join(f"{np.corrcoef(oofs[ki],oofs[kj])[0,1]:.3f}" for kj in keys))

    # 优化权重
    def neg(w):
        return -wr2(yv, w @ P, wv)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1})
    bnds = [(0, 1)] * len(keys)
    best = None
    for _ in range(60):
        w0 = np.random.dirichlet(np.ones(len(keys)))
        r = minimize(neg, w0, method="SLSQP", bounds=bnds, constraints=cons,
                     options={"maxiter": 200, "ftol": 1e-8})
        if best is None or r.fun < best.fun: best = r
    w = np.maximum(best.x, 0); w = w / w.sum()
    print(f"\n=== 最优集成 ===")
    for k, wi in zip(keys, w): print(f"  {k}: {wi:.4f}")
    print(f"  集成 holdout R² = {wr2(yv, w @ P, wv):+.6f}")

    # 对比等权
    ew = np.ones(len(keys)) / len(keys)
    print(f"  等权 holdout R² = {wr2(yv, ew @ P, wv):+.6f}")

    # 应用到 test
    if "te_preds" in d:
        T = np.array([d["te_preds"][k] for k in keys])
        # NN 去均值对齐（lgb 均值）
        lgb_keys = [k for k in keys if "lgb" in k]
        ref_mean = T[0].mean() if not lgb_keys else np.mean([T[keys.index(k)].mean() for k in lgb_keys])
        for i, k in enumerate(keys):
            if k.startswith("nn"): T[i] = T[i] - T[i].mean() + ref_mean
        pred = w @ T
        pred = np.where(np.isfinite(pred), pred, 0.0)
        rid = d["row_id"]
        out = pd.DataFrame({"row_id": rid, "target": pred})
        out.to_csv(out_csv, index=False)
        print(f"\n→ wrote {out_csv} mean={pred.mean():+.5f} std={pred.std():.5f}", flush=True)
        # 保存权重
        json.dump({"models": keys, "weights": w.tolist(), "holdout_r2": float(wr2(yv, w@P, wv)),
                   "src": oof_path},
                  open(Path(out_csv).with_suffix(".json"), "w"), indent=2)
    else:
        print("(no te_preds in pkl, skip writing test predictions)")


if __name__ == "__main__":
    main()
