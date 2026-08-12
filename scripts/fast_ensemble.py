"""快速集成：合并 oof_all.pkl + ratio4_oof.pkl，SLSQP 优化权重，
test 预测从指定 submission 文件加载。
"""
from __future__ import annotations
import pickle, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize

RUN = Path("/mnt/iscsi/hd/xxz/runs"); SUB = Path("/mnt/iscsi/hd/xxz/submissions")

TE_FILE = {
    "lgb": "lgbm_full_submission.csv",
    "cb": "cb_submission.csv",
    "xgb": "xgb_submission.csv",
    "nn1": "nn1_10s.csv",
    "nn2": "nn2_submission.csv",
    "nn3_emb32": "nn3_10s.csv",
    "nn4_deep4": "nnvar_deep4.csv",
    "ratio_lgb": "ratio4_ratio_lgb.csv",
    "ratio_cb": "ratio4_ratio_cb.csv",
    "ratio_xgb": "ratio4_ratio_xgb.csv",
    "ratio_nn": "ratio4_ratio_nn.csv",
}


def wr2(y, p, w):
    d = float(np.sum(w*y*y)); return 0.0 if d<=0 else 1-float(np.sum(w*(y-p)**2)/d)


def main():
    out_csv = sys.argv[1] if len(sys.argv) > 1 else str(SUB/"final_ens.csv")
    print(f"[start] loading oof_all.pkl", flush=True)
    d1 = pickle.load(open(RUN/"oof_all.pkl","rb"))
    oofs = dict(d1["oofs"]); yv = d1["yv"]; wv = d1["wv"]
    print(f"  oof_all: {list(oofs.keys())}", flush=True)
    print(f"[start] loading ratio4_oof.pkl", flush=True)
    d2 = pickle.load(open(RUN/"ratio4_oof.pkl","rb"))
    for k, v in d2["oofs"].items():
        assert len(v) == len(yv), f"{k}: {len(v)} != {len(yv)}"
        oofs[k] = v
    print(f"  +ratio4: {list(d2['oofs'].keys())}", flush=True)
    keys = list(oofs.keys())
    print(f"\n=== {len(keys)} 模型 holdout R² ===", flush=True)
    for k in keys: print(f"  {k}: {wr2(yv, oofs[k], wv):+.6f}", flush=True)

    P = np.array([oofs[k] for k in keys])
    print(f"\n=== 相关性矩阵 ===", flush=True)
    for ki in keys:
        print(f"  {ki:>14s}: " + " ".join(f"{np.corrcoef(oofs[ki],oofs[kj])[0,1]:>.3f}" for kj in keys), flush=True)

    # 优化
    print(f"\n=== SLSQP 优化（20 restart） ===", flush=True)
    def neg(w): return -wr2(yv, w @ P, wv)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1})
    bnds = [(0, 1)] * len(keys)
    best = None; t0 = time.time()
    np.random.seed(2026)
    for i in range(20):
        w0 = np.random.dirichlet(np.ones(len(keys)))
        r = minimize(neg, w0, method="SLSQP", bounds=bnds, constraints=cons, options={"maxiter": 200, "ftol": 1e-8})
        if best is None or r.fun < best.fun:
            best = r; print(f"  iter {i}: holdout={-r.fun:+.6f}", flush=True)
    w = np.maximum(best.x, 0); w = w / w.sum()
    print(f"\n=== 最优权重（{time.time()-t0:.0f}s） holdout R²={wr2(yv, w@P, wv):+.6f} ===", flush=True)
    for k, wi in sorted(zip(keys, w), key=lambda x: -x[1]):
        print(f"  {k}: {wi:.4f}", flush=True)

    # 加载 test 预测
    print(f"\n=== 加载 test 预测 ===", flush=True)
    base = pd.read_csv(SUB/TE_FILE[keys[0]]).sort_values("row_id").reset_index(drop=True)
    T = []; valid_keys = []
    for k in keys:
        f = TE_FILE.get(k)
        if f is None or not (SUB/f).exists():
            print(f"  ! {k}: no file ({f}), skip", flush=True); continue
        d = pd.read_csv(SUB/f).sort_values("row_id").reset_index(drop=True)
        assert (d["row_id"].to_numpy() == base["row_id"].to_numpy()).all(), f"{k}: row_id mismatch"
        T.append(d["target"].to_numpy(np.float32)); valid_keys.append(k)
        print(f"  +{k} ({f})", flush=True)
    T = np.array(T)
    w_valid = np.array([w[keys.index(k)] for k in valid_keys]); w_valid = w_valid / w_valid.sum()
    # NN 类去均值对齐
    lgb_keys = [k for k in valid_keys if "lgb" in k]
    ref = np.mean([T[valid_keys.index(k)].mean() for k in lgb_keys]) if lgb_keys else T[0].mean()
    for i, k in enumerate(valid_keys):
        if "nn" in k and "lgb" not in k:
            T[i] = T[i] - T[i].mean() + ref
    pred = w_valid @ T; pred = np.where(np.isfinite(pred), pred, 0.0)
    pd.DataFrame({"row_id": base["row_id"], "target": pred}).to_csv(out_csv, index=False)
    print(f"\n→ wrote {out_csv} mean={pred.mean():+.5f} std={pred.std():.5f}", flush=True)
    json.dump({"models": valid_keys, "weights": w_valid.tolist(),
               "holdout_r2": float(wr2(yv, np.array([oofs[k] for k in valid_keys]).T @ w_valid, wv))},
              open(Path(out_csv).with_suffix(".json"), "w"), indent=2)


if __name__ == "__main__":
    main()
