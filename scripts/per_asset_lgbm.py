"""per-asset 独立建模：15 个 LGBM，每个标的一个专门模型。

全局模型的 asset_id 类别只学粗粒度调整；独立模型让每标的有自己的特征-目标关系。
holdout 上对比全局 LGBM(0.00170)。若专门化更强→新方向。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb

STRAT="/mnt/iscsi/hd/xxz/public_release_lightgbm_baseline/examples/lightgbm_baseline"
if STRAT not in sys.path: sys.path.insert(0,STRAT)
from data_utils import manifest_files, feature_columns_from_path  # noqa: E402
DATA_ROOT=Path("/mnt/iscsi/hd/xxz/public_release_20260630/data")
PARAMS=lambda s: dict(objective="regression",metric="None",learning_rate=0.05,num_leaves=63,min_data_in_leaf=500,
            feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,lambda_l2=5,verbosity=-1,num_threads=16,
            seed=s,bagging_seed=s,feature_fraction_seed=s,data_random_seed=s)


def wr2(y,p,w):
    d=float(np.sum(w*y*y)); return 0.0 if d<=0 else 1-float(np.sum(w*(y-p)**2)/d)
def feval(p,ds):
    y=ds.get_label(); w=ds.get_weight()
    if w is None: w=np.ones_like(y)
    d=float(np.sum(w*y*y)); s=0 if d<=0 else 1-float(np.sum(w*(y-p)**2)/d)
    return ("wr2",float(s),True)


def main():
    paths=manifest_files(DATA_ROOT,"train")[:3]; feats=feature_columns_from_path(paths[0])
    pf=pd.read_parquet(paths,columns=["time_id","asset_id","weight","target"]+feats)
    pf[feats]=np.nan_to_num(pf[feats].to_numpy(np.float32))
    print(f"loaded {len(pf):,}",flush=True)
    times=np.sort(pf["time_id"].unique()); ho=set(times[-max(1,int(len(times)*0.15)):].tolist())
    pf["is_va"]=pf["time_id"].isin(ho)

    # 全局对照
    a_all=pf["asset_id"].to_numpy(np.float32)
    X_all=np.column_stack([a_all, pf[feats].to_numpy(np.float32)])
    y= pd.to_numeric(pf["target"],errors="coerce").fillna(0).to_numpy(np.float32)
    w= pd.to_numeric(pf["weight"],errors="coerce").fillna(0).clip(lower=0).to_numpy(np.float32)
    va=pf["is_va"].to_numpy()
    dtr=lgb.Dataset(X_all[~va],label=y[~va],weight=w[~va],categorical_feature=[0],free_raw_data=False)
    dva=lgb.Dataset(X_all[va],label=y[va],weight=w[va],reference=dtr,free_raw_data=False)
    m=lgb.train(PARAMS(2026),dtr,num_boost_round=300,valid_sets=[dva],valid_names=["va"],feval=feval,
                callbacks=[lgb.early_stopping(40,verbose=False),lgb.log_evaluation(0)])
    r_global=wr2(y[va],m.predict(X_all[va]),w[va])
    print(f"全局LGBM holdout={r_global:+.5f}",flush=True)

    # per-asset
    oof=np.zeros(len(pf)); mask=np.zeros(len(pf),bool)
    for a in sorted(pf["asset_id"].unique()):
        sub=pf[pf["asset_id"]==a]
        idx=sub.index.to_numpy()
        Xa=sub[feats].to_numpy(np.float32)  # 无需asset_id列(全同一asset)
        ya=pd.to_numeric(sub["target"],errors="coerce").fillna(0).to_numpy(np.float32)
        wa=pd.to_numeric(sub["weight"],errors="coerce").fillna(0).clip(lower=0).to_numpy(np.float32)
        va_a=sub["is_va"].to_numpy()
        if (~va_a).sum()<1000: continue  # 太少跳过
        dtr_a=lgb.Dataset(Xa[~va_a],label=ya[~va_a],weight=wa[~va_a],free_raw_data=False)
        dva_a=lgb.Dataset(Xa[va_a],label=ya[va_a],weight=wa[va_a],reference=dtr_a,free_raw_data=False)
        ma=lgb.train(PARAMS(int(a)),dtr_a,num_boost_round=300,valid_sets=[dva_a],valid_names=["va"],feval=feval,
                     callbacks=[lgb.early_stopping(30,verbose=False),lgb.log_evaluation(0)])
        oof[idx[va_a]]=ma.predict(Xa[va_a]); mask[idx[va_a]]=True
        print(f"  asset {int(a)}: rows={len(sub)} holdout_wr2={wr2(ya[va_a],ma.predict(Xa[va_a]),wa[va_a]):+.5f} iters={ma.best_iteration}",flush=True)
    r_per=wr2(y[mask],oof[mask],w[mask])
    print(f"\nper-asset LGBM holdout={r_per:+.5f} (全局 {r_global:+.5f})",flush=True)


if __name__=="__main__":
    main()
