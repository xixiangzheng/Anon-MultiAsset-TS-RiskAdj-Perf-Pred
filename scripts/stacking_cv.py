"""时序分块 Stacking：用 16 模型 OOF 作为特征，训练 LGBM 元模型。

关键：用时序分块 CV（避免 holdout 过拟合）。
- 元模型: LGBM (num_leaves=8, depth=3, 强正则)
- CV: 5 fold 时序分块，每 fold 训练元模型 on 4 fold, 预测 1 fold
- 最终 holdout 评估 + test 预测
"""
from __future__ import annotations
import pickle, json, time
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb

RUN = Path("/mnt/iscsi/hd/xxz/runs"); SUB = Path("/mnt/iscsi/hd/xxz/submissions")

TE_FILE = {
    "lgb":"lgbm_full_submission.csv","cb":"cb_submission.csv","xgb":"xgb_submission.csv",
    "nn1":"nn1_10s.csv","nn2":"nn2_submission.csv","nn3_emb32":"nn3_10s.csv","nn4_deep4":"nnvar_deep4.csv",
    "ratio_lgb":"ratio4_ratio_lgb.csv","ratio_cb":"ratio4_ratio_cb.csv",
    "ratio_xgb":"ratio4_ratio_xgb.csv","ratio_nn":"ratio4_ratio_nn.csv",
    "ratio_cb_tuned":"v2_ratio_cb_tuned.csv","ratio_cb_deep":"v2_ratio_cb_deep.csv","ratio_nn_emb32":"v2_ratio_nn_emb32.csv",
    "ratio_sum_lgb":"rsum_ratio_sum_lgb.csv","ratio_sum_cb":"rsum_ratio_sum_cb.csv",
}


def wr2(y, p, w):
    d = float(np.sum(w*y*y)); return 0.0 if d<=0 else 1-float(np.sum(w*(y-p)**2)/d)


def main():
    # 加载所有 OOF
    d1 = pickle.load(open(RUN/"oof_all.pkl","rb"))
    oofs = dict(d1["oofs"]); yv = d1["yv"]; wv = d1["wv"]
    tids = d1.get("tids_holdout")
    d2 = pickle.load(open(RUN/"ratio4_oof.pkl","rb"))
    for k,v in d2["oofs"].items(): oofs[k]=v
    d3 = pickle.load(open(RUN/"ratio_v2_oof.pkl","rb"))
    for k,v in d3["oofs"].items(): oofs[k]=v
    d4 = pickle.load(open(RUN/"ratio_sum_oof.pkl","rb"))
    for k,v in d4["oofs"].items(): oofs[k]=v
    keys = list(oofs.keys())
    P = np.array([oofs[k] for k in keys]).T  # [N_holdout, n_models]
    print(f"{len(keys)} models, holdout size {P.shape}", flush=True)

    # 时序分块（用 tids）
    if tids is None:
        print("warning: no tids, using random split"); tids = np.random.permutation(len(P))
    unique_tids = np.sort(np.unique(tids))
    n_tids = len(unique_tids)
    n_folds = 5
    fold_size = n_tids // n_folds
    folds = []
    for i in range(n_folds):
        va_tids = unique_tids[i*fold_size:(i+1)*fold_size if i < n_folds-1 else n_tids]
        va_mask = np.isin(tids, va_tids)
        folds.append((~va_mask, va_mask))
    print(f"{n_folds} folds, fold sizes: {[f[1].sum() for f in folds]}", flush=True)

    # LGBM 元模型（强正则，避免过拟合）
    params = dict(objective="regression", metric="None", learning_rate=0.01, num_leaves=8,
                  max_depth=3, min_data_in_leaf=5000, feature_fraction=0.8, lambda_l2=10,
                  verbosity=-1, num_threads=32, seed=2026)
    def feval(p, ds):
        y = ds.get_label(); w = ds.get_weight()
        if w is None: w = np.ones_like(y)
        return ("wr2", float(wr2(y, p, w)), True)

    oof_meta = np.zeros(len(P))
    for k, (tr_mask, va_mask) in enumerate(folds):
        Xtr, Xva = P[tr_mask], P[va_mask]
        ytr, yva = yv[tr_mask], yv[va_mask]
        wtr, wva = wv[tr_mask], wv[va_mask]
        dtr = lgb.Dataset(Xtr, label=ytr, weight=wtr, feature_name=keys, free_raw_data=False)
        dva = lgb.Dataset(Xva, label=yva, weight=wva, reference=dtr, free_raw_data=False)
        m = lgb.train(params, dtr, num_boost_round=500, valid_sets=[dva], valid_names=["va"],
                      feval=feval, callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
        bi = m.best_iteration or 500
        oof_meta[va_mask] = m.predict(Xva)
        fold_r2 = wr2(yva, m.predict(Xva), wva)
        print(f"  fold {k}: iter={bi} R²={fold_r2:+.6f}", flush=True)
    meta_r2 = wr2(yv, oof_meta, wv)
    print(f"\nStacking CV holdout R² = {meta_r2:+.6f}", flush=True)

    # 对比线性最优
    from scipy.optimize import minimize
    P2 = P.T  # [n_models, N]
    def neg(w): return -wr2(yv, w @ P2, wv)
    cons = ({"type":"eq","fun":lambda w:w.sum()-1}); bnds=[(0,1)]*len(keys)
    best=None; np.random.seed(2026)
    for _ in range(20):
        r = minimize(neg, np.random.dirichlet(np.ones(len(keys))), method="SLSQP", bounds=bnds, constraints=cons)
        if best is None or r.fun < best.fun: best = r
    print(f"Linear SLSQP holdout R² = {-best.fun:+.6f}", flush=True)

    # 如果 stacking > linear，用全量训练元模型预测 test
    if meta_r2 > -best.fun:
        print("\nStacking 更好，全量训练元模型...", flush=True)
        dtr_full = lgb.Dataset(P, label=yv, weight=wv, feature_name=keys, free_raw_data=False)
        # 用平均 fold iter
        m = lgb.train(params, dtr_full, num_boost_round=200, feval=feval)
        # 加载 test
        base = pd.read_csv(SUB/TE_FILE[keys[0]]).sort_values("row_id").reset_index(drop=True)
        T = []
        for k in keys:
            d = pd.read_csv(SUB/TE_FILE[k]).sort_values("row_id").reset_index(drop=True)
            T.append(d["target"].to_numpy(np.float32))
        T = np.array(T).T  # [N_test, n_models]
        # NN 去均值
        lgb_idx = [i for i,k in enumerate(keys) if "lgb" in k]
        if lgb_idx:
            ref = np.mean([T[:,i].mean() for i in lgb_idx])
            for i,k in enumerate(keys):
                if "nn" in k and "lgb" not in k: T[:,i] = T[:,i] - T[:,i].mean() + ref
        pred = m.predict(T); pred = np.where(np.isfinite(pred), pred, 0.0)
        pd.DataFrame({"row_id": base["row_id"], "target": pred}).to_csv(SUB/"stacking_ens.csv", index=False)
        print(f"wrote stacking_ens.csv mean={pred.mean():+.5f}", flush=True)
    else:
        print("\nStacking 不如 linear，跳过", flush=True)


if __name__ == "__main__":
    main()
