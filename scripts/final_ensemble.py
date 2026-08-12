"""综合集成：合并 oof_all.pkl (7 模型) + ratio4_oof.pkl (4 ratio 模型) + ratio_tw_lgb_oof.pkl，
用 submissions/*.csv 作为 test 预测，SLSQP 优化非负权重最大化加权R² → 写入 submissions/final_ens.csv。
"""
from __future__ import annotations
import sys, pickle, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize

SUB = Path("/mnt/iscsi/hd/xxz/submissions"); RUN = Path("/mnt/iscsi/hd/xxz/runs")


def wr2(y, p, w):
    d = float(np.sum(w*y*y)); return 0.0 if d<=0 else 1-float(np.sum(w*(y-p)**2)/d)


# 各模型 → submission 文件名 映射
TE_FILE = {
    "lgb": "lgbm_full_submission.csv",  # 原 baseline LGBM
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
    "ratio_tw_lgb": "ratio_tw_lgb.csv",
    "ratio_lgb_old": "ens_ratio_nn30.csv",  # 当前公榜最优（已含 ratio 特征 lgb + cb + nn）
}


def main():
    out_csv = sys.argv[1] if len(sys.argv) > 1 else str(SUB / "final_ens.csv")
    # 加载所有 OOF
    d1 = pickle.load(open(RUN/"oof_all.pkl","rb"))
    oofs = dict(d1["oofs"]); yv = d1["yv"]; wv = d1["wv"]
    print(f"oof_all: {list(oofs.keys())}")

    if (RUN/"ratio4_oof.pkl").exists():
        d2 = pickle.load(open(RUN/"ratio4_oof.pkl","rb"))
        for k, v in d2["oofs"].items():
            # 必须相同 holdout 行
            assert len(v) == len(yv), f"{k}: len(v)={len(v)} != {len(yv)}"
            oofs[k] = v
        print(f"+ratio4: added {[k for k in d2['oofs'].keys() if k not in d1['oofs']]}")
    if (RUN/"ratio_tw_lgb_oof.pkl").exists():
        d3 = pickle.load(open(RUN/"ratio_tw_lgb_oof.pkl","rb"))
        assert len(d3["holdout_pred"]) == len(yv)
        oofs["ratio_tw_lgb"] = d3["holdout_pred"]
        print("+ratio_tw_lgb: added")

    keys = list(oofs.keys())
    print(f"\n=== {len(keys)} 模型 holdout R² ===")
    for k in keys: print(f"  {k}: {wr2(yv, oofs[k], wv):+.6f}")

    # 相关性
    print("\n=== 相关性矩阵 ===")
    print(" "*16 + " ".join(f"{k[:8]:>9s}" for k in keys))
    for ki in keys:
        print(f"{ki:>14s} " + " ".join(f"{np.corrcoef(oofs[ki],oofs[kj])[0,1]:>+9.4f}" for kj in keys))

    # 优化权重
    P = np.array([oofs[k] for k in keys])
    def neg(w): return -wr2(yv, w @ P, wv)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1})
    bnds = [(0, 1)] * len(keys)
    best = None
    np.random.seed(2026)
    for _ in range(80):
        w0 = np.random.dirichlet(np.ones(len(keys)))
        r = minimize(neg, w0, method="SLSQP", bounds=bnds, constraints=cons,
                     options={"maxiter": 300, "ftol": 1e-9})
        if best is None or r.fun < best.fun: best = r
    w = np.maximum(best.x, 0); w = w / w.sum()
    print(f"\n=== 最优权重（holdout R²={wr2(yv, w@P, wv):+.6f}） ===")
    for k, wi in sorted(zip(keys, w), key=lambda x: -x[1]):
        print(f"  {k}: {wi:.4f}")

    # 加载 test 预测
    base = pd.read_csv(SUB/TE_FILE[keys[0]]).sort_values("row_id").reset_index(drop=True)
    T = []
    valid_keys = []
    for k in keys:
        f = TE_FILE.get(k)
        if f is None or not (SUB/f).exists():
            print(f"  ! {k}: no test file, skip"); continue
        d = pd.read_csv(SUB/f).sort_values("row_id").reset_index(drop=True)
        assert (d["row_id"].to_numpy() == base["row_id"].to_numpy()).all(), f"{k}: row_id mismatch"
        T.append(d["target"].to_numpy(np.float32)); valid_keys.append(k)
    T = np.array(T); w_valid = np.array([w[keys.index(k)] for k in valid_keys])
    w_valid = w_valid / w_valid.sum()
    # NN 类去均值对齐（按 lgb 均值）
    lgb_keys = [k for k in valid_keys if "lgb" in k]
    ref = np.mean([T[valid_keys.index(k)].mean() for k in lgb_keys]) if lgb_keys else T[0].mean()
    for i,k in enumerate(valid_keys):
        if "nn" in k: T[i] = T[i] - T[i].mean() + ref
    pred = w_valid @ T; pred = np.where(np.isfinite(pred), pred, 0.0)
    pd.DataFrame({"row_id": base["row_id"], "target": pred}).to_csv(out_csv, index=False)
    print(f"\n→ wrote {out_csv} mean={pred.mean():+.5f} std={pred.std():.5f}")
    json.dump({"models": valid_keys, "weights": w_valid.tolist(),
               "holdout_r2": float(wr2(yv, np.array([oofs[k] for k in valid_keys]).T @ w_valid, wv))},
              open(Path(out_csv).with_suffix(".json"), "w"), indent=2)


if __name__ == "__main__":
    main()
