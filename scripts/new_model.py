"""全新模型：大规模特征交互发现 + 从头构建。

1. 搜所有 323 特征作为分母: top-50 特征 ÷ 每个候选分母 → 找最佳分母
2. 用最佳分母组: 构造全部比率/乘积交互
3. 排序留 top-50 交互特征
4. 用 [323原始 + top-50交互] 从头训练 LGBM(purged CV 选轮数 + 3 seed)
5. 出 test 预测 = 全新模型
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb

STRAT="/mnt/iscsi/hd/xxz/public_release_lightgbm_baseline/examples/lightgbm_baseline"
if STRAT not in sys.path: sys.path.insert(0,STRAT)
from data_utils import manifest_files, feature_columns_from_path  # noqa: E402
DATA_ROOT=Path("/mnt/iscsi/hd/xxz/public_release_20260630/data")
PARAMS=lambda s: dict(objective="regression",metric="None",learning_rate=0.05,num_leaves=63,min_data_in_leaf=2000,
            feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10,verbosity=-1,num_threads=48,
            seed=s,bagging_seed=s,feature_fraction_seed=s,data_random_seed=s)


def wcorr(x,y,w):
    wm=(w*x).sum()/w.sum(); ym=(w*y).sum()/w.sum()
    num=(w*(x-wm)*(y-ym)).sum(); den=np.sqrt((w*(x-wm)**2).sum()*(w*(y-ym)**2).sum())
    return abs(num/den) if den>0 else 0

def feval(p,ds):
    y=ds.get_label(); w=ds.get_weight()
    if w is None: w=np.ones_like(y)
    d=float(np.sum(w*y*y)); s=0 if d<=0 else 1-float(np.sum(w*(y-p)**2)/d)
    return ("wr2",float(s),True)
def wr2(y,p,w):
    d=float(np.sum(w*y*y)); return 0.0 if d<=0 else 1-float(np.sum(w*(y-p)**2)/d)


def main():
    paths=manifest_files(DATA_ROOT,"train"); feats=feature_columns_from_path(paths[0])
    tr=pd.read_parquet(paths,columns=["time_id","asset_id","weight","target"]+feats)
    tr[feats]=np.nan_to_num(tr[feats].to_numpy(np.float32))
    te=pd.read_parquet(manifest_files(DATA_ROOT,"test"),columns=["row_id","asset_id"]+feats)
    te[feats]=np.nan_to_num(te[feats].to_numpy(np.float32)); te=te.sort_values("row_id").reset_index(drop=True)
    print(f"train {len(tr):,} test {len(te):,}",flush=True)
    y=pd.to_numeric(tr["target"],errors="coerce").fillna(0).to_numpy(np.float64)
    w=pd.to_numeric(tr["weight"],errors="coerce").fillna(0).clip(lower=0).to_numpy(np.float64)
    F=tr[feats].to_numpy(np.float64); n=len(tr)

    # Step1: 找最佳分母(top-50特征 ÷ 每个候选分母, 看哪个分母最好)
    sc=np.array([wcorr(F[:,i],y,w) for i in range(len(feats))])
    top50_idx=np.argsort(-sc)[:50]
    print("搜索最佳分母(323个候选)...",flush=True); t0=time.time()
    denom_scores=[]
    for d in range(len(feats)):
        fd=F[:,d].copy()
        fd_c=np.clip(fd,1e-6 if fd.min()>=0 else None,None) if fd.min()>=0 else fd
        if fd.min()<0: continue  # 跳过有负值的分母(logic:风险因子应为正)
        fd_s=np.clip(fd,1e-8,np.percentile(fd,99))
        imps=[]
        for ni in top50_idx[:10]:  # top-10特征测试
            r=np.clip(F[:,ni]/fd_s,np.percentile(F[:,ni]/fd_s,1),np.percentile(F[:,ni]/fd_s,99))
            r=np.nan_to_num(r,nan=0,posinf=0,neginf=0)
            imps.append(wcorr(r,y,w)-sc[ni])
        denom_scores.append((feats[d],float(np.mean(imps))))
    denom_scores.sort(key=lambda x:-x[1])
    print(f"  done {time.time()-t0:.0f}s. Top-10 分母:",flush=True)
    for f,s in denom_scores[:10]: print(f"    {f}: avg_improvement={s:+.5f}",flush=True)

    # Step2: 用 top-5 分母, 构造全部交互(top-50特征÷5分母 = 250比率)
    best_denoms=[f for f,_ in denom_scores[:5]]
    print(f"\n用分母 {best_denoms} 构造交互...",flush=True)
    interactions=[]
    for dn in best_denoms:
        di=feats.index(dn); fd=np.clip(F[:,di],1e-8,np.percentile(F[:,di],99))
        for ni in top50_idx:
            r=np.nan_to_num(np.clip(F[:,ni]/fd,np.percentile(F[:,ni]/fd,1),np.percentile(F[:,ni]/fd,99)),nan=0,posinf=0,neginf=0)
            c=wcorr(r,y,w)
            interactions.append((f"{feats[ni]}/{dn}",c,feats[ni],dn))
    interactions.sort(key=lambda x:-x[1])
    top_inter=interactions[:50]
    print(f"top-10 交互:",flush=True)
    for name,c,*_ in top_inter[:10]: print(f"  {c:.4f} {name}",flush=True)

    # Step3: 构建新特征集 + 从头训练
    inter_feats=[f"{name}" for name,_,_,_ in top_inter]
    # 构造 train+test 的新特征
    def add_interactions(df):
        for name,_,numer,denom in top_inter:
            ni=feats.index(numer); di=feats.index(denom)
            fd=np.clip(df[denom].to_numpy(np.float32),1e-8,np.percentile(F[:,feats.index(denom)],99))
            r=np.nan_to_num(np.clip(df[numer].to_numpy(np.float32)/fd,-1e6,1e6),nan=0,posinf=0,neginf=0)
            df[f"int_{numer}_div_{denom}"]=r
        return df
    tr=add_interactions(tr); te=add_interactions(te)
    inter_cols=[f"int_{name.replace('/','_div_')}" for name,_,_,_ in top_inter]
    all_cols=feats+inter_cols
    print(f"\n全新模型: {len(feats)}原始 + {len(inter_cols)}交互 = {len(all_cols)}特征",flush=True)

    # purged holdout 选轮数
    times=np.sort(tr["time_id"].unique()); ho=set(times[-max(1,int(len(times)*0.15)):].tolist())
    is_va=tr["time_id"].isin(ho).to_numpy()
    a_tr=tr["asset_id"].to_numpy(np.float32); a_te=te["asset_id"].to_numpy(np.float32)
    Xtr=np.column_stack([a_tr]+[tr[c].to_numpy(np.float32) for c in all_cols])
    Xva=np.column_stack([a_tr[is_va]]+[tr.loc[is_va,c].to_numpy(np.float32) for c in all_cols])
    y32=pd.to_numeric(tr["target"],errors="coerce").fillna(0).to_numpy(np.float32)
    w32=pd.to_numeric(tr["weight"],errors="coerce").fillna(0).clip(lower=0).to_numpy(np.float32)
    dtr=lgb.Dataset(Xtr[~is_va],label=y32[~is_va],weight=w32[~is_va],categorical_feature=[0],free_raw_data=False)
    dva=lgb.Dataset(Xva,label=y32[is_va],weight=w32[is_va],reference=dtr,free_raw_data=False)
    m=lgb.train(PARAMS(2026),dtr,num_boost_round=300,valid_sets=[dva],valid_names=["va"],feval=feval,
                callbacks=[lgb.early_stopping(40,verbose=False),lgb.log_evaluation(0)])
    bi=m.best_iteration or 300
    print(f"holdout iter={bi} R²={wr2(y32[is_va].astype(np.float64),m.predict(Xva),w32[is_va].astype(np.float64)):+.5f}",flush=True)
    # 3 seed 全量
    Xte=np.column_stack([a_te]+[te[c].to_numpy(np.float32) for c in all_cols])
    preds=[]
    for s in [2026,2027,2028]:
        d=lgb.Dataset(Xtr,label=y32,weight=w32,categorical_feature=[0],free_raw_data=False)
        mm=lgb.train(PARAMS(s),d,num_boost_round=bi); preds.append(mm.predict(Xte))
    avg=np.mean(preds,0); avg=np.where(np.isfinite(avg),avg,0.0)
    pd.DataFrame({"row_id":te["row_id"],"target":avg}).to_csv("/mnt/iscsi/hd/xxz/submissions/new_model_submission.csv",index=False)
    print(f"\nwrote new_model_submission mean={avg.mean():+.4f}",flush=True)
    json.dump({"interactions":top_inter,"best_denoms":best_denoms,"holdout_r2":float(wr2(y32[is_va].astype(np.float64),m.predict(Xva),w32[is_va].astype(np.float64)))},
              open("/mnt/iscsi/hd/xxz/runs/new_model_features.json","w"),indent=2)


if __name__=="__main__":
    main()
