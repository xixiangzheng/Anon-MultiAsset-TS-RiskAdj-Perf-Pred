"""top-100 特征 LGBM（独立稳健改进）：全量训练，3种子，出 test 预测。

迁移验证证实 top-100 比全323更稳健(未见区+0.00024)。用 baseline 稳健参数(l2=10等)，
在 top-100 特征上训 3 种子 → predict test → lgb_top100_submission。用于替换/增强集成。
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb

STRAT="/mnt/iscsi/hd/xxz/public_release_lightgbm_baseline/examples/lightgbm_baseline"
if STRAT not in sys.path: sys.path.insert(0,STRAT)
from data_utils import manifest_files, feature_columns_from_path  # noqa: E402
DATA_ROOT=Path("/mnt/iscsi/hd/xxz/public_release_20260630/data")
K=100; SEEDS=[2026,2027,2028]
PARAMS=lambda s: dict(objective="regression",metric="None",learning_rate=0.05,num_leaves=63,min_data_in_leaf=2000,
            feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10,verbosity=-1,num_threads=48,
            seed=s,bagging_seed=s,feature_fraction_seed=s,data_random_seed=s)

def wr2(y,p,w):
    d=float(np.sum(w*y*y)); return 0.0 if d<=0 else 1-float(np.sum(w*(y-p)**2)/d)
def feval(p,ds):
    y=ds.get_label(); w=ds.get_weight()
    if w is None: w=np.ones_like(y)
    d=float(np.sum(w*y*y)); s=0 if d<=0 else 1-float(np.sum(w*(y-p)**2)/d)
    return ("wr2",float(s),True)


def main():
    paths=manifest_files(DATA_ROOT,"train"); feats=feature_columns_from_path(paths[0])
    tr=pd.read_parquet(paths,columns=["time_id","asset_id","weight","target"]+feats)
    tr[feats]=np.nan_to_num(tr[feats].to_numpy(np.float32))
    te=pd.read_parquet(manifest_files(DATA_ROOT,"test"),columns=["row_id","asset_id"]+feats)
    te[feats]=np.nan_to_num(te[feats].to_numpy(np.float32)); te=te.sort_values("row_id").reset_index(drop=True)
    print(f"train {len(tr):,} test {len(te):,}",flush=True)
    y=pd.to_numeric(tr["target"],errors="coerce").fillna(0).to_numpy(np.float32)
    w=pd.to_numeric(tr["weight"],errors="coerce").fillna(0).clip(lower=0).to_numpy(np.float32)
    F=tr[feats].to_numpy(np.float32)
    # 全量排序特征(加权相关)
    wm=(w*w).sum(); wmean=lambda a:(w*a).sum()/wm; ym=wmean(y); den_y=(w*(y-ym)**2).sum()
    sc=np.array([abs((w*(F[:,i]-wmean(F[:,i]))*(y-ym)).sum()/np.sqrt(((w*(F[:,i]-wmean(F[:,i]))**2).sum())*den_y)) if den_y>0 else 0 for i in range(len(feats))])
    order=np.argsort(-sc); sel=[feats[i] for i in order[:K]]
    print(f"top-{K} 特征(全量排序): {sel[:8]}...",flush=True)
    # holdout 定 iterations
    times=np.sort(tr["time_id"].unique()); ho=set(times[-max(1,int(len(times)*0.15)):].tolist())
    is_va=tr["time_id"].isin(ho).to_numpy()
    aTr=tr["asset_id"].to_numpy(np.float32); FTr=tr[sel].to_numpy(np.float32)
    aTe=te["asset_id"].to_numpy(np.float32); FTe=te[sel].to_numpy(np.float32)
    Xtr=np.column_stack([aTr,FTr]); Xva=np.column_stack([aTr[is_va],FTr[is_va]])
    dtr=lgb.Dataset(Xtr[~is_va],label=y[~is_va],weight=w[~is_va],categorical_feature=[0],free_raw_data=False)
    dva=lgb.Dataset(Xva,label=y[is_va],weight=w[is_va],reference=dtr,free_raw_data=False)
    m=lgb.train(PARAMS(2026),dtr,num_boost_round=400,valid_sets=[dva],valid_names=["va"],feval=feval,
                callbacks=[lgb.early_stopping(50,verbose=False),lgb.log_evaluation(0)])
    bi=m.best_iteration or 400; print(f"holdout iter={bi} R²={wr2(y[is_va],m.predict(Xva),w[is_va]):+.5f}",flush=True)
    # 3种子全量重训
    Xfull=np.column_stack([aTr,FTr]); Xtest=np.column_stack([aTe,FTe])
    preds=[]
    for s in SEEDS:
        d=lgb.Dataset(Xfull,label=y,weight=w,categorical_feature=[0],free_raw_data=False)
        mm=lgb.train(PARAMS(s),d,num_boost_round=bi); preds.append(mm.predict(Xtest))
    avg=np.mean(preds,0); avg=np.where(np.isfinite(avg),avg,0.0)
    pd.DataFrame({"row_id":te["row_id"],"target":avg}).to_csv("/mnt/iscsi/hd/xxz/submissions/lgb_top100_submission.csv",index=False)
    print(f"wrote lgb_top100_submission mean={avg.mean():+.4f}",flush=True)
    json.dump({"sel_features":sel},open("/mnt/iscsi/hd/xxz/runs/top100_features.json","w"))


if __name__=="__main__":
    main()
