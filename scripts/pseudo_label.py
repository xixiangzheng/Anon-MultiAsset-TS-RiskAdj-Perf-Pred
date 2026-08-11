"""伪标签(Pseudo-Labeling) + winsorize：让模型自适应 test 特征分布。

1. LGBM(baseline参数) 在 train 上训 → 预测 test 得伪标签
2. train + test(伪标签, 降权0.3) 重训 LGBM
3. winsorize target 裁极端值
组合产出 lgb_pseudo_submission，再入集成。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb

STRAT="/mnt/iscsi/hd/xxz/public_release_lightgbm_baseline/examples/lightgbm_baseline"
if STRAT not in sys.path: sys.path.insert(0,STRAT)
from data_utils import manifest_files, feature_columns_from_path  # noqa: E402
DATA_ROOT=Path("/mnt/iscsi/hd/xxz/public_release_20260630/data")
PARAMS=lambda s: dict(objective="regression",metric="None",learning_rate=0.05,num_leaves=63,min_data_in_leaf=2000,
            feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10,verbosity=-1,num_threads=48,
            seed=s,bagging_seed=s,feature_fraction_seed=s,data_random_seed=s)
SEEDS=[2026,2027,2028]; PSEUDO_W=0.3; ITERS=200


def main():
    paths=manifest_files(DATA_ROOT,"train"); feats=feature_columns_from_path(paths[0])
    tr=pd.read_parquet(paths,columns=["asset_id","weight","target"]+feats)
    tr[feats]=np.nan_to_num(tr[feats].to_numpy(np.float32))
    te=pd.read_parquet(manifest_files(DATA_ROOT,"test"),columns=["row_id","asset_id"]+feats)
    te[feats]=np.nan_to_num(te[feats].to_numpy(np.float32)); te=te.sort_values("row_id").reset_index(drop=True)
    print(f"train {len(tr):,} test {len(te):,}",flush=True)
    y=pd.to_numeric(tr["target"],errors="coerce").fillna(0).to_numpy(np.float32)
    w=pd.to_numeric(tr["weight"],errors="coerce").fillna(0).clip(lower=0).to_numpy(np.float32)
    a_tr=tr["asset_id"].to_numpy(np.float32); a_te=te["asset_id"].to_numpy(np.float32)
    Xtr=np.column_stack([a_tr, tr[feats].to_numpy(np.float32)])
    Xte=np.column_stack([a_te, te[feats].to_numpy(np.float32)])
    med_w=np.median(w[w>0])
    print(f"median weight={med_w:.4f}",flush=True)

    # 第1轮: train 上训, 预测 test 伪标签(3 seed 平均)
    pseudo=np.zeros(len(te))
    for s in SEEDS:
        d=lgb.Dataset(Xtr,label=y,weight=w,categorical_feature=[0],free_raw_data=False)
        m=lgb.train(PARAMS(s),d,num_boost_round=ITERS); pseudo+=m.predict(Xte)
    pseudo/=len(SEEDS)
    print(f"伪标签生成: mean={pseudo.mean():+.4f} std={pseudo.std():.4f}",flush=True)

    # 第2轮: train(winsorize) + test(伪标签,降权) 重训
    lo,hi=np.quantile(y,[0.01,0.99]); y_win=np.clip(y,lo,hi)
    w_te=np.full(len(te),PSEUDO_W*med_w,dtype=np.float32)
    Xaug=np.vstack([Xtr,Xte]); yaug=np.concatenate([y_win,pseudo]); waug=np.concatenate([w,w_te])
    aaug_asset=np.concatenate([tr["asset_id"].to_numpy(np.int32), te["asset_id"].to_numpy(np.int32)])
    print(f"增强训练集: {len(Xaug):,} (train {len(Xtr):,} + test伪标签 {len(Xte):,})",flush=True)

    preds=[]
    for s in SEEDS:
        d=lgb.Dataset(Xaug,label=yaug,weight=waug,categorical_feature=[0],free_raw_data=False)
        m=lgb.train(PARAMS(s),d,num_boost_round=ITERS); preds.append(m.predict(Xte))
    avg=np.mean(preds,0); avg=np.where(np.isfinite(avg),avg,0.0)
    pd.DataFrame({"row_id":te["row_id"],"target":avg}).to_csv("/mnt/iscsi/hd/xxz/submissions/lgb_pseudo_submission.csv",index=False)
    print(f"wrote lgb_pseudo_submission mean={avg.mean():+.4f} std={avg.std():.4f}",flush=True)
    # 相关性 vs 原 LGBM
    orig=pd.read_csv("/mnt/iscsi/hd/xxz/submissions/lgbm_full_submission.csv").sort_values("row_id")["target"].to_numpy()
    print(f"corr(伪标签LGBM, 原LGBM) = {np.corrcoef(avg,orig)[0,1]:.3f}",flush=True)


if __name__=="__main__":
    main()
