"""时间衰减加权实验：LGBM 训练时给近期样本更高权重(测试集是未来)。

effective_weight = weight × exp(α × time_rank_normalized)。在 holdout(末15%,未来样)上对比不同 α。
若衰减提升 holdout R²，说明对公榜(同样未来)有益。
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb

STRAT="/mnt/iscsi/hd/xxz/public_release_lightgbm_baseline/examples/lightgbm_baseline"
if STRAT not in sys.path: sys.path.insert(0,STRAT)
from data_utils import manifest_files, feature_columns_from_path  # noqa: E402
DATA_ROOT=Path("/mnt/iscsi/hd/xxz/public_release_20260630/data")
TUNED=dict(objective="regression",metric="None",learning_rate=0.010326965981106629,num_leaves=79,
           min_data_in_leaf=556,feature_fraction=0.5974233229491067,bagging_fraction=0.9445362670741704,
           bagging_freq=1,lambda_l1=0.03778653953330111,lambda_l2=2.9757802078489703,max_bin=127,
           verbosity=-1,num_threads=32,seed=2026,bagging_seed=2026,feature_fraction_seed=2026,data_random_seed=2026)


def wr2(y,p,w):
    d=float(np.sum(w*y*y)); return 0.0 if d<=0 else 1-float(np.sum(w*(y-p)**2)/d)


def feval(p,ds):
    y=ds.get_label(); w=ds.get_weight()
    if w is None: w=np.ones_like(y)
    d=float(np.sum(w*y*y)); s=0 if d<=0 else 1-float(np.sum(w*(y-p)**2)/d)
    return ("wr2",float(s),True)


def main():
    paths=manifest_files(DATA_ROOT,"train"); feats=feature_columns_from_path(paths[0])
    pf=pd.read_parquet(paths,columns=["time_id","asset_id","weight","target"]+feats)
    pf[feats]=np.nan_to_num(pf[feats].to_numpy(np.float32))
    times=np.sort(pf["time_id"].unique()); ho=set(times[-max(1,int(len(times)*0.15)):].tolist())
    is_va=pf["time_id"].isin(ho).to_numpy(); tr_df,va_df=pf[~is_va].reset_index(drop=True),pf[is_va].reset_index(drop=True)
    print(f"train {len(tr_df):,} / holdout {len(va_df):,}", flush=True)
    # 时间衰减基于 train 内的 time rank
    tr_times=np.sort(tr_df["time_id"].unique()); rank={t:i/len(tr_times) for i,t in enumerate(tr_times)}
    decay_tr=np.array([rank[t] for t in tr_df["time_id"]])  # 0(早)~1(晚)
    Xtr=np.column_stack([tr_df["asset_id"].to_numpy(np.float32), tr_df[feats].to_numpy(np.float32)])
    Xva=np.column_stack([va_df["asset_id"].to_numpy(np.float32), va_df[feats].to_numpy(np.float32)])
    ytr=pd.to_numeric(tr_df["target"],errors="coerce").fillna(0).to_numpy(np.float32)
    yv=pd.to_numeric(va_df["target"],errors="coerce").fillna(0).to_numpy(np.float64)
    wbase=pd.to_numeric(tr_df["weight"],errors="coerce").fillna(0).clip(lower=0).to_numpy(np.float32)
    wv=pd.to_numeric(va_df["weight"],errors="coerce").fillna(0).clip(lower=0).to_numpy(np.float64)
    dva=lgb.Dataset(Xva,label=pd.to_numeric(va_df["target"],errors="coerce").fillna(0).to_numpy(np.float32),weight=wv.astype(np.float32),free_raw_data=False)
    print(f"\n{'alpha':>6} {'holdout_R2':>10}", flush=True)
    for alpha in [0.0, 0.5, 1.0, 2.0, 4.0]:
        w=wbase*np.exp(alpha*decay_tr).astype(np.float32)
        dtr=lgb.Dataset(Xtr,label=ytr,weight=w,categorical_feature=[0],free_raw_data=False)
        m=lgb.train(TUNED,dtr,num_boost_round=448)
        r2=wr2(yv, m.predict(Xva), wv)
        print(f"{alpha:>6.1f} {r2:+10.5f}", flush=True)


if __name__=="__main__":
    main()
