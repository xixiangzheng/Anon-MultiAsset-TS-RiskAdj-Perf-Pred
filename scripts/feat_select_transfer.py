"""特征选择的迁移验证 + 组合测试。

关键：top-100 在 A 区选、B 区(完全不同分区)评，验证是否稳健迁移(非过拟合 holdout)。
同时测 top-100 + winsorize 组合。若迁移成立→独立设计真有效，可建 top-100 模型入集成。
"""
from __future__ import annotations
import sys
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

def rank_feats(F,y,w):
    wm=(w*w).sum(); wmean=lambda a:(w*a).sum()/wm; ym=wmean(y); den_y=(w*(y-ym)**2).sum()
    sc=[]
    for i in range(F.shape[1]):
        x=F[:,i]; xm=wmean(x); den=np.sqrt(((w*(x-xm)**2).sum())*den_y)
        sc.append(abs((w*(x-xm)*(y-ym)).sum()/den) if den>0 else 0)
    order=np.argsort(-np.array(sc)); return order

def train_eval(Xtr,ytr,wtr,Xva,yva,wva):
    dtr=lgb.Dataset(Xtr,label=ytr,weight=wtr,categorical_feature=[0],free_raw_data=False)
    dva=lgb.Dataset(Xva,label=yva,weight=wva,reference=dtr,free_raw_data=False)
    m=lgb.train(PARAMS,dtr,num_boost_round=300,valid_sets=[dva],valid_names=["va"],feval=feval,
                callbacks=[lgb.early_stopping(40,verbose=False),lgb.log_evaluation(0)])
    return wr2(yva,m.predict(Xva,num_iteration=m.best_iteration or 300),wva)


def main():
    paths=manifest_files(DATA_ROOT,"train"); feats=feature_columns_from_path(paths[0])
    # A 区 = parts 0-2 (选特征 + 训练), B 区 = parts 3-4 (完全未见, 迁移评估)
    A=pd.read_parquet(paths[:3],columns=["time_id","asset_id","weight","target"]+feats)
    B=pd.read_parquet(paths[3:5],columns=["time_id","asset_id","weight","target"]+feats)
    A[feats]=np.nan_to_num(A[feats].to_numpy(np.float32)); B[feats]=np.nan_to_num(B[feats].to_numpy(np.float32))
    def prep(df):
        return (df["asset_id"].to_numpy(np.float32), df[feats].to_numpy(np.float32),
                pd.to_numeric(df["target"],errors="coerce").fillna(0).to_numpy(np.float32),
                pd.to_numeric(df["weight"],errors="coerce").fillna(0).clip(lower=0).to_numpy(np.float32))
    aA,FA,yA,wA=prep(A); aB,FB,yB,wB=prep(B)
    print(f"A(parts0-2) {len(A):,} 选特征+训练, B(parts3-4) {len(B):,} 迁移评估",flush=True)
    # 在 A 区选特征
    order=rank_feats(FA,yA,wA)
    yB64=yB.astype(np.float64); wB64=wB.astype(np.float64)
    print(f"\n{'实验':36s} {'A区holdout':>10s} {'B区迁移':>10s}",flush=True)
    for K,lab in [(323,"全323(baseline)"),(100,"top-100"),(150,"top-150"),(200,"top-200")]:
        sel=order[:K]
        XtrA=np.column_stack([aA,FA[:,sel]]); XvaB=np.column_stack([aB,FB[:,sel]])
        # A 区内部 holdout(末15%)训练评估
        tA=np.sort(A["time_id"].unique()); hoA=set(tA[-max(1,int(len(tA)*0.15)):].tolist())
        isA=A["time_id"].isin(hoA).to_numpy()
        rA=train_eval(XtrA[~isA],yA[~isA],wA[~isA], XtrA[isA],yA[isA],wA[isA])
        # 用全A训练, 在B(完全未见)评估
        rB=train_eval(XtrA,yA,wA, XvaB,yB64,wB64)
        print(f"{lab:36s} {rA:+10.5f} {rB:+10.5f}",flush=True)
    # 组合: top-100 + winsorize, 在 B 区迁移评估
    sel=order[:100]; XtrA=np.column_stack([aA,FA[:,sel]]); XvaB=np.column_stack([aB,FB[:,sel]])
    lo,hi=np.quantile(yA,[0.01,0.99]); yA_w=np.clip(yA,lo,hi)
    rB_combo=train_eval(XtrA,yA_w,wA, XvaB,yB64,wB64)
    print(f"{'top-100 + winsorize':36s} {'':>10s} {rB_combo:+10.5f}",flush=True)


if __name__=="__main__":
    main()
