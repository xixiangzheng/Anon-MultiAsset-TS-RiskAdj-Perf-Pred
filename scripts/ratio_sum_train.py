"""Ratio + Sum 交互训练：在 ratio_top50 基础上加 top-30 sum 交互，训练 CB + LGBM。

依据：nonratio_search.py 发现 sum(A+B) 比 ratio(A/B) 信号更强（0.0359 vs 0.0338）。
"""
from __future__ import annotations
import sys, time, json, pickle, gc
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb, catboost as cb

STRAT = "/mnt/iscsi/hd/xxz/public_release_lightgbm_baseline/examples/lightgbm_baseline"
if STRAT not in sys.path: sys.path.insert(0, STRAT)
from data_utils import manifest_files, feature_columns_from_path
DATA_ROOT = Path("/mnt/iscsi/hd/xxz/public_release_20260630/data")
RUN = Path("/mnt/iscsi/hd/xxz/runs"); SUB = Path("/mnt/iscsi/hd/xxz/submissions")
GPU_CB = "1"

TUNED = dict(objective="regression", metric="None", learning_rate=0.010326965981106629,
             num_leaves=79, min_data_in_leaf=556, feature_fraction=0.5974233229491067,
             bagging_fraction=0.9445362670741704, bagging_freq=1, lambda_l1=0.03778653953330111,
             lambda_l2=2.9757802078489703, max_bin=127, verbosity=-1, num_threads=64, seed=2026,
             bagging_seed=2026, feature_fraction_seed=2026, data_random_seed=2026)


def wr2(y, p, w):
    d = float(np.sum(w*y*y)); return 0.0 if d<=0 else 1-float(np.sum(w*(y-p)**2)/d)


def feval_wr2(preds, ds):
    y = ds.get_label(); w = ds.get_weight()
    if w is None: w = np.ones_like(y)
    return ("wr2", float(wr2(y, preds, w)), True)


def clip01_np(x, F_train_for_pct=None):
    x = np.nan_to_num(x, nan=0, posinf=0, neginf=0)
    if F_train_for_pct is None:
        lo, hi = np.percentile(x, [1, 99])
    else:
        lo, hi = np.percentile(F_train_for_pct, [1, 99])
    return np.clip(x, lo, hi)


def main():
    paths = manifest_files(DATA_ROOT, "train"); feats = feature_columns_from_path(paths[0])
    rs = json.loads((RUN/"ratio_top50.json").read_text())["ratios"]
    ratios = [(feats.index(r[2]), feats.index(r[3])) for r in rs]
    # 加载 sum top-30
    nr = json.loads((RUN/"nonratio_interactions.json").read_text())
    sums = [(d["a"], d["b"]) for d in nr["by_type"].get("sum", [])][:30]
    print(f"features: {len(feats)} raw + {len(ratios)} ratio + {len(sums)} sum = {len(feats)+len(ratios)+len(sums)}", flush=True)

    pf = pd.read_parquet(paths, columns=["time_id","asset_id","weight","target"]+feats)
    pf[feats] = np.nan_to_num(pf[feats].to_numpy(np.float32))
    F_all = pf[feats].to_numpy(np.float32)
    te = pd.read_parquet(manifest_files(DATA_ROOT,"test"),columns=["row_id","asset_id"]+feats)
    te[feats] = np.nan_to_num(te[feats].to_numpy(np.float32)); te = te.sort_values("row_id").reset_index(drop=True)
    F_te = te[feats].to_numpy(np.float32)

    times = np.sort(pf["time_id"].unique()); ho = set(times[-max(1,int(len(times)*0.15)):].tolist())
    is_va = pf["time_id"].isin(ho).to_numpy()
    y32 = pd.to_numeric(pf["target"],errors="coerce").fillna(0).to_numpy(np.float32)
    w32 = pd.to_numeric(pf["weight"],errors="coerce").fillna(0).clip(lower=0).to_numpy(np.float32)
    yv = y32[is_va].astype(np.float64); wv = w32[is_va].astype(np.float64)
    a32 = pf["asset_id"].to_numpy(np.float32)
    print(f"data: train {len(pf):,} holdout {is_va.sum():,}", flush=True)

    # 构造 ratio + sum 特征
    def build_extras(F):
        cols = []
        for ni,di in ratios:
            fd = np.clip(F[:,di], 1e-8, np.percentile(F[:,di], 99))
            r = F[:,ni]/fd
            lo, hi = np.percentile(r, [1, 99])
            r = np.nan_to_num(np.clip(r, lo, hi), nan=0, posinf=0, neginf=0)
            cols.append(r.astype(np.float32))
        for ai,bi in sums:
            cols.append((F[:,feats.index(ai)] + F[:,feats.index(bi)]).astype(np.float32))
        return np.column_stack(cols)

    E_all = build_extras(F_all); E_te = build_extras(F_te)
    n_extras = E_all.shape[1]
    extra_cols = [f"e{i}" for i in range(n_extras)]
    print(f"extras: {n_extras}", flush=True)

    oof_pkl = RUN/"ratio_sum_partial.pkl"
    if oof_pkl.exists():
        prev = pickle.load(open(oof_pkl,"rb")); oofs=prev.get("oofs",{}); te_preds=prev.get("te_preds",{})
        print(f"[resume] {list(oofs.keys())}", flush=True)
    else: oofs={}; te_preds={}
    def save_partial():
        pickle.dump({"oofs":oofs,"te_preds":te_preds,"yv":yv,"wv":wv,
                     "row_id":te["row_id"].to_numpy()}, open(oof_pkl,"wb"))

    # === 1. ratio_sum_cb (CB + ratio + sum) ===
    if "ratio_sum_cb" not in oofs:
        print("\n=== ratio_sum_cb (CatBoost + ratio + sum) ===", flush=True); t0=time.time()
        pdf = pf[["asset_id"]+feats].copy(); pdf[extra_cols] = E_all
        pdf["asset_id"] = pdf["asset_id"].astype(np.int32)
        tedf = te[["asset_id"]+feats].copy(); tedf[extra_cols] = E_te
        tedf["asset_id"] = tedf["asset_id"].astype(np.int32)
        tr_df = pdf[~is_va].reset_index(drop=True); va_df = pdf[is_va].reset_index(drop=True)
        trpool = cb.Pool(tr_df, label=y32[~is_va], weight=w32[~is_va], cat_features=["asset_id"])
        vapool = cb.Pool(va_df, label=yv.astype(np.float32), weight=wv.astype(np.float32), cat_features=["asset_id"])
        cbp = dict(loss_function="RMSE", learning_rate=0.05, depth=8, l2_leaf_reg=3.0,
                   iterations=800, random_seed=2026, task_type="GPU", devices=GPU_CB,
                   verbose=False, early_stopping_rounds=50, use_best_model=True)
        mc = cb.train(trpool, cbp, eval_set=vapool, verbose=False)
        bi = mc.tree_count_
        oofs["ratio_sum_cb"] = mc.predict(vapool)
        fullpool = cb.Pool(pdf, label=y32, weight=w32, cat_features=["asset_id"])
        preds = []
        for s in [2026,2027,2028]:
            p = dict(cbp); p["random_seed"]=s; p["iterations"]=bi; p["use_best_model"]=False
            mm = cb.train(fullpool, p, verbose=False); preds.append(mm.predict(tedf))
        te_preds["ratio_sum_cb"] = np.mean(preds, 0)
        print(f"  holdout R²={wr2(yv,oofs['ratio_sum_cb'],wv):+.5f} trees={bi} ({time.time()-t0:.0f}s)", flush=True)
        save_partial(); del trpool,vapool,fullpool,mc,pdf,tr_df,va_df; gc.collect()
    else: print("[skip] ratio_sum_cb", flush=True)

    # === 2. ratio_sum_lgb (LGBM + ratio + sum) ===
    if "ratio_sum_lgb" not in oofs:
        print("\n=== ratio_sum_lgb (LGBM tuned + ratio + sum) ===", flush=True); t0=time.time()
        Xtr_full = np.column_stack([a32, F_all, E_all])
        Xva = Xtr_full[is_va]; Xtr = Xtr_full[~is_va]
        ytr = y32[~is_va]; wtr = w32[~is_va]
        Xte_full = np.column_stack([te["asset_id"].to_numpy(np.float32), F_te, E_te])
        dtr = lgb.Dataset(Xtr, label=ytr, weight=wtr, categorical_feature=[0], free_raw_data=False)
        dva = lgb.Dataset(Xva, label=yv.astype(np.float32), weight=wv.astype(np.float32), reference=dtr, free_raw_data=False)
        m = lgb.train(TUNED, dtr, num_boost_round=1500, valid_sets=[dva], valid_names=["va"], feval=feval_wr2,
                      callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)])
        bi = m.best_iteration or 1500
        oofs["ratio_sum_lgb"] = m.predict(Xva)
        # 全量 3 seed
        Xf = np.concatenate([Xtr, Xva]); yf = np.concatenate([ytr, yv.astype(np.float32)])
        wf = np.concatenate([wtr, wv.astype(np.float32)])
        dtr_full = lgb.Dataset(Xf, label=yf, weight=wf, categorical_feature=[0], free_raw_data=False)
        preds = []
        for s in [2026,2027,2028]:
            p = dict(TUNED); p.update(seed=s, bagging_seed=s, feature_fraction_seed=s)
            mm = lgb.train(p, dtr_full, num_boost_round=bi, feval=feval_wr2); preds.append(mm.predict(Xte_full))
        te_preds["ratio_sum_lgb"] = np.mean(preds, 0)
        print(f"  holdout R²={wr2(yv,oofs['ratio_sum_lgb'],wv):+.5f} iter={bi} ({time.time()-t0:.0f}s)", flush=True)
        save_partial(); del dtr,dva,dtr_full,m; gc.collect()
    else: print("[skip] ratio_sum_lgb", flush=True)

    # 写 submissions + 最终 pkl
    for k,p in te_preds.items():
        p = np.where(np.isfinite(p), p, 0.0)
        pd.DataFrame({"row_id":te["row_id"],"target":p}).to_csv(SUB/f"rsum_{k}.csv", index=False)
    pickle.dump({"oofs":oofs,"te_preds":te_preds,"yv":yv,"wv":wv,"row_id":te["row_id"].to_numpy()},
                open(RUN/"ratio_sum_oof.pkl","wb"))
    print(f"\n[done] {len(oofs)} models saved.", flush=True)


if __name__ == "__main__":
    main()
