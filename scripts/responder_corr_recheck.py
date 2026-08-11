"""严格重测 responder：predicted responder 与 target 的【相关性】(scale-free)。

之前用原始 R²(尺度敏感)误判 responder 无用。这里直接测 corr(target, ŝ_k)：
- 若 corr 高(>0.3)：responder 是金矿，预测+缩放即得强 target 预测 → 冲 0.004 的钥匙
- 若 corr 低(<0.1)：responder 确实无用
同时测最优缩放后的 R² 和多 responder 线性组合。
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb

STRAT="/mnt/iscsi/hd/xxz/public_release_lightgbm_baseline/examples/lightgbm_baseline"
if STRAT not in sys.path: sys.path.insert(0,STRAT)
from data_utils import manifest_files, feature_columns_from_path  # noqa: E402
from validation import make_validation_plan, weighted_zero_mean_r2  # noqa: E402
DATA_ROOT=Path("/mnt/iscsi/hd/xxz/public_release_20260630/data")
RESPS=["responder_03","responder_02","responder_18","responder_19","responder_17","responder_11"]
PARAMS=dict(objective="regression",metric="None",learning_rate=0.05,num_leaves=63,min_data_in_leaf=2000,
            feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10,verbosity=-1,num_threads=32,
            seed=2026,bagging_seed=2026,feature_fraction_seed=2026,data_random_seed=2026)


def feval(p,ds):
    y=ds.get_label(); w=ds.get_weight()
    if w is None: w=np.ones_like(y)
    d=float(np.sum(w*y*y)); s=0 if d<=0 else 1-float(np.sum(w*(y-p)**2)/d)
    return ("wr2",float(s),True)


def main():
    paths=manifest_files(DATA_ROOT,"train")[:2]  # 2分区足够稳定估计
    feats=feature_columns_from_path(paths[0])
    pf=pd.read_parquet(paths,columns=["time_id","asset_id","weight","target"]+RESPS+feats)
    pf[feats]=np.nan_to_num(pf[feats].to_numpy(np.float32))
    print(f"loaded {len(pf):,} rows (2 partitions)", flush=True)
    yt=pd.to_numeric(pf["target"],errors="coerce").fillna(0).to_numpy(np.float32)
    wt=pd.to_numeric(pf["weight"],errors="coerce").fillna(0).clip(lower=0).to_numpy(np.float32)
    tids=pf["time_id"].to_numpy(); asset=pf["asset_id"].to_numpy(np.float32)
    X=np.column_stack([asset, pf[feats].astype(np.float32).to_numpy()])
    plan=make_validation_plan(pd.Series(tids),n_splits=5,holdout_fraction=0.15,purge_steps=30)

    # OOF predict 每个 responder
    oof={}
    for resp in RESPS:
        y=pd.to_numeric(pf[resp],errors="coerce").replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(np.float32)
        w=pd.to_numeric(pf["weight"],errors="coerce").fillna(0).clip(lower=0).to_numpy(np.float32)
        pr=np.zeros(len(pf)); mask=np.zeros(len(pf),bool)
        for f in plan.folds:
            tr=np.isin(tids,list(map(int,f.train_time_ids))); va=np.isin(tids,list(map(int,f.valid_time_ids)))
            dtr=lgb.Dataset(X[tr],label=y[tr],weight=w[tr],categorical_feature=[0],free_raw_data=False)
            dva=lgb.Dataset(X[va],label=y[va],weight=w[va],reference=dtr,free_raw_data=False)
            m=lgb.train(PARAMS,dtr,num_boost_round=250,valid_sets=[dva],valid_names=["va"],feval=feval,
                        callbacks=[lgb.early_stopping(40,verbose=False),lgb.log_evaluation(0)])
            pr[va]=m.predict(X[va],num_iteration=m.best_iteration or 250); mask[va]=True
        oof[resp]=pr
        # 关键: corr(target, ŝ_resp) 和 最优缩放R²
        c=np.corrcoef(yt[mask],pr[mask])[0,1]
        # 最优缩放: target ≈ a*ŝ, 加权最小二乘
        a=np.sum(wt[mask]*yt[mask]*pr[mask])/np.sum(wt[mask]*pr[mask]**2)
        r2scaled=weighted_zero_mean_r2(yt[mask].astype(np.float64), (a*pr)[mask].astype(np.float64), wt[mask].astype(np.float64))
        print(f"{resp}: corr(target,ŝ)={c:+.4f}  最优缩放R²={r2scaled:+.5f}  (scale a={a:.4f})", flush=True)
    # 多 responder 线性组合(交叉拟合)
    from scipy.optimize import minimize
    P=np.array([oof[r] for r in RESPS]); ym=yt[mask]; wm=wt[mask]; Pm=P[:,mask]
    def neg(w): return -weighted_zero_mean_r2(ym.astype(np.float64), (w@Pm).astype(np.float64), wm.astype(np.float64))
    cons={"type":"eq","fun":lambda w:w.sum()-1}; bnds=[(-2,2)]*len(RESPS); best=None
    for _ in range(40):
        r=minimize(neg,np.random.randn(len(RESPS)),method="SLSQP",bounds=bnds,constraints=cons)
        if best is None or r.fun<best.fun: best=r
    print(f"\n多responder线性组合(全OOF, in-sample) R²={-best.fun:+.5f} weights={dict(zip(RESPS,[round(float(x),2) for x in best.x]))}", flush=True)
    print(f"对照: 直接预测target基线 ≈ +0.00200", flush=True)


if __name__=="__main__":
    main()
