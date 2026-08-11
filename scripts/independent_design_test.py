"""独立设计测试1：激进特征选择。

target 难预测→很多特征可能是噪声。按单特征与target的加权相关排序，保留top-K，
看 LGBM 在 top-K 上是否超过全323(baseline holdout 0.00170)。若子集更优→baseline"全用"次优。
同时测 winsorize target 和 Huber loss 两个独立思路。
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb

STRAT="/mnt/iscsi/hd/xxz/public_release_lightgbm_baseline/examples/lightgbm_baseline"
if STRAT not in sys.path: sys.path.insert(0,STRAT)
from data_utils import manifest_files, feature_columns_from_path  # noqa: E402
DATA_ROOT=Path("/mnt/iscsi/hd/xxz/public_release_20260630/data")
PARAMS=dict(objective="regression",metric="None",learning_rate=0.05,num_leaves=63,min_data_in_leaf=2000,
            feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10,verbosity=-1,num_threads=32,
            seed=2026,bagging_seed=2026,feature_fraction_seed=2026,data_random_seed=2026)


def wr2(y,p,w):
    d=float(np.sum(w*y*y)); return 0.0 if d<=0 else 1-float(np.sum(w*(y-p)**2)/d)


def feval(p,ds):
    y=ds.get_label(); w=ds.get_weight()
    if w is None: w=np.ones_like(y)
    d=float(np.sum(w*y*y)); s=0 if d<=0 else 1-float(np.sum(w*(y-p)**2)/d)
    return ("wr2",float(s),True)


def train_eval(Xtr,ytr,wtr,Xva,yva,wva,params=None,feval_only_mse=False):
    p=params or PARAMS
    dtr=lgb.Dataset(Xtr,label=ytr,weight=wtr,categorical_feature=[0],free_raw_data=False)
    dva=lgb.Dataset(Xva,label=yva,weight=wva,reference=dtr,free_raw_data=False)
    m=lgb.train(p,dtr,num_boost_round=300,valid_sets=[dva],valid_names=["va"],feval=feval,
                callbacks=[lgb.early_stopping(40,verbose=False),lgb.log_evaluation(0)])
    return wr2(yva,m.predict(Xva,num_iteration=m.best_iteration or 300),wva)


def main():
    paths=manifest_files(DATA_ROOT,"train")[:2]; feats=feature_columns_from_path(paths[0])
    pf=pd.read_parquet(paths,columns=["time_id","asset_id","weight","target"]+feats)
    pf[feats]=np.nan_to_num(pf[feats].to_numpy(np.float32))
    print(f"loaded {len(pf):,}",flush=True)
    times=np.sort(pf["time_id"].unique()); ho=set(times[-max(1,int(len(times)*0.15)):].tolist())
    is_va=pf["time_id"].isin(ho).to_numpy(); tr,va=pf[~is_va].reset_index(drop=True),pf[is_va].reset_index(drop=True)
    ytr=pd.to_numeric(tr["target"],errors="coerce").fillna(0).to_numpy(np.float32)
    wtr=pd.to_numeric(tr["weight"],errors="coerce").fillna(0).clip(lower=0).to_numpy(np.float32)
    yva=pd.to_numeric(va["target"],errors="coerce").fillna(0).to_numpy(np.float64)
    wva=pd.to_numeric(va["weight"],errors="coerce").fillna(0).clip(lower=0).to_numpy(np.float64)
    asset_tr=tr["asset_id"].to_numpy(np.float32); asset_va=va["asset_id"].to_numpy(np.float32)
    Ftr=tr[feats].to_numpy(np.float32); Fva=va[feats].to_numpy(np.float32)

    # 单特征加权相关排序
    wm=(wtr*wtr).sum()
    wmean=lambda a: (wtr*a).sum()/wm
    scores=[]
    for i,f in enumerate(feats):
        x=Ftr[:,i]; xm=wmean(x); ym=wmean(ytr)
        num=(wtr*(x-xm)*(ytr-ym)).sum()
        den=np.sqrt(((wtr*(x-xm)**2).sum())*((wtr*(ytr-ym)**2).sum()))
        scores.append((f,abs(num/den) if den>0 else 0))
    scores.sort(key=lambda t:-t[1])
    print("top-10 特征(加权相关):",[(f,round(s,4)) for f,s in scores[:10]],flush=True)
    print("相关分布: >0.02:%d, >0.01:%d, <0.005:%d"%(sum(s>0.02 for _,s in scores),sum(s>0.01 for _,s in scores),sum(s<0.005 for _,s in scores)),flush=True)

    print(f"\n{'实验':30s} {'holdout_R2':>10s}",flush=True)
    # 基线: 全323
    Xtr=np.column_stack([asset_tr,Ftr]); Xva=np.column_stack([asset_va,Fva])
    r=train_eval(Xtr,ytr,wtr,Xva,yva,wva); print(f"{'基线 全323特征':30s} {r:+10.5f}",flush=True)
    # 特征选择 top-K
    for K in [30,50,100,200]:
        sel=[f for f,_ in scores[:K]]
        Ftr_s=tr[sel].to_numpy(np.float32); Fva_s=va[sel].to_numpy(np.float32)
        Xtr_s=np.column_stack([asset_tr,Ftr_s]); Xva_s=np.column_stack([asset_va,Fva_s])
        r=train_eval(Xtr_s,ytr,wtr,Xva_s,yva,wva); print(f"{'特征选择 top-%d'%K:30s} {r:+10.5f}",flush=True)
    # winsorize target (clip p1-p99)
    lo,hi=np.quantile(ytr,[0.01,0.99])
    r=train_eval(Xtr,np.clip(ytr,lo,hi),wtr,Xva,yva,wva); print(f"{'winsorize target p1-p99':30s} {r:+10.5f}",flush=True)
    # Huber loss
    p=dict(PARAMS); p["objective"]="huber"; p["alpha"]=0.9
    r=train_eval(Xtr,ytr,wtr,Xva,yva,wva,params=p); print(f"{'Huber loss':30s} {r:+10.5f}",flush=True)


if __name__=="__main__":
    main()
