"""稳健权重候选：除了 SLSQP 优化（可能过拟合 holdout），生成多个手设权重候选。

候选：
1. ratio_cb主导: ratio_cb 0.5 + ratio_nn_emb32 0.3 + ratio_lgb 0.2
2. CB家族: ratio_cb 0.4 + ratio_cb_tuned 0.3 + ratio_nn_emb32 0.3
3. 等权 top-5: (ratio_cb, ratio_nn_emb32, ratio_lgb, xgb, nn3_emb32) 各 0.2
4. 公榜冠军配方替代: ratio_cb 0.4 + nn3 0.3 + ratio_lgb 0.3
5. NN主导: ratio_nn_emb32 0.5 + nn3_emb32 0.3 + ratio_cb 0.2

所有候选在 holdout 上评估，记录 R²。
"""
from __future__ import annotations
import pickle, json, sys
from pathlib import Path
import numpy as np, pandas as pd

RUN = Path("/mnt/iscsi/hd/xxz/runs"); SUB = Path("/mnt/iscsi/hd/xxz/submissions")

TE_FILE = {
    "lgb":"lgbm_full_submission.csv","cb":"cb_submission.csv","xgb":"xgb_submission.csv",
    "nn1":"nn1_10s.csv","nn2":"nn2_submission.csv","nn3_emb32":"nn3_10s.csv","nn4_deep4":"nnvar_deep4.csv",
    "ratio_lgb":"ratio4_ratio_lgb.csv","ratio_cb":"ratio4_ratio_cb.csv",
    "ratio_xgb":"ratio4_ratio_xgb.csv","ratio_nn":"ratio4_ratio_nn.csv",
    "ratio_cb_tuned":"v2_ratio_cb_tuned.csv","ratio_cb_deep":"v2_ratio_cb_deep.csv","ratio_nn_emb32":"v2_ratio_nn_emb32.csv",
}


def wr2(y, p, w):
    d = float(np.sum(w*y*y)); return 0.0 if d<=0 else 1-float(np.sum(w*(y-p)**2)/d)


def main():
    d1 = pickle.load(open(RUN/"oof_all.pkl","rb"))
    oofs = dict(d1["oofs"]); yv = d1["yv"]; wv = d1["wv"]
    d2 = pickle.load(open(RUN/"ratio4_oof.pkl","rb"))
    for k,v in d2["oofs"].items(): oofs[k]=v
    d3 = pickle.load(open(RUN/"ratio_v2_oof.pkl","rb"))
    for k,v in d3["oofs"].items(): oofs[k]=v
    keys = list(oofs.keys())

    candidates = {
        "c1_cb_dominant": {"ratio_cb":0.5, "ratio_nn_emb32":0.3, "ratio_lgb":0.2},
        "c2_cb_family": {"ratio_cb":0.4, "ratio_cb_tuned":0.3, "ratio_nn_emb32":0.3},
        "c3_eq_top5": {"ratio_cb":0.2, "ratio_nn_emb32":0.2, "ratio_lgb":0.2, "xgb":0.2, "nn3_emb32":0.2},
        "c4_champion_replace": {"ratio_cb":0.4, "nn3_emb32":0.3, "ratio_lgb":0.3},
        "c5_nn_dominant": {"ratio_nn_emb32":0.5, "nn3_emb32":0.3, "ratio_cb":0.2},
        "c6_cb_nn_balance": {"ratio_cb":0.35, "ratio_nn_emb32":0.35, "ratio_cb_deep":0.15, "xgb":0.15},
        "c7_lgb_cb_nn": {"ratio_lgb":0.4, "ratio_cb":0.4, "ratio_nn_emb32":0.2},
    }

    print("=== 候选权重 holdout 评估 ===")
    results = {}
    for name, weights in candidates.items():
        # 验证 keys 存在
        missing = [k for k in weights if k not in oofs]
        if missing:
            print(f"  {name}: missing {missing}, skip"); continue
        w_arr = np.array([weights[k] for k in weights])
        w_arr = w_arr / w_arr.sum()  # 归一化
        P = np.array([oofs[k] for k in weights])
        r2 = wr2(yv, w_arr @ P, wv)
        results[name] = (r2, weights)
        print(f"  {name}: holdout={r2:+.6f}")

    # 写最优 3 个候选
    sorted_c = sorted(results.items(), key=lambda x: -x[1][0])
    base = pd.read_csv(SUB/TE_FILE["ratio_lgb"]).sort_values("row_id").reset_index(drop=True)
    for name, (r2, weights) in sorted_c[:5]:
        T = []; valid_keys = []
        for k in weights:
            f = TE_FILE.get(k)
            if f is None or not (SUB/f).exists(): continue
            d = pd.read_csv(SUB/f).sort_values("row_id").reset_index(drop=True)
            T.append(d["target"].to_numpy(np.float32)); valid_keys.append(k)
        T = np.array(T); w_arr = np.array([weights[k] for k in valid_keys]); w_arr = w_arr / w_arr.sum()
        lgb_keys = [k for k in valid_keys if "lgb" in k]
        ref = np.mean([T[valid_keys.index(k)].mean() for k in lgb_keys]) if lgb_keys else T[0].mean()
        for i,k in enumerate(valid_keys):
            if "nn" in k and "lgb" not in k:
                T[i] = T[i] - T[i].mean() + ref
        pred = w_arr @ T; pred = np.where(np.isfinite(pred), pred, 0.0)
        out = SUB/f"robust_{name}.csv"
        pd.DataFrame({"row_id": base["row_id"], "target": pred}).to_csv(out, index=False)
        print(f"  → wrote {out.name} mean={pred.mean():+.5f} holdout={r2:+.6f}")


if __name__ == "__main__":
    main()
