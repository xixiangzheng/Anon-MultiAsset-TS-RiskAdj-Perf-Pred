"""CatBoost Optuna (GPU)：在 holdout 上搜 CatBoost 超参，强化集成里的 CB 成员。

train 减末15%holdout 训练，holdout 评加权 R²。GPU 让 20-30 trials 可行(~20min)。
"""
from __future__ import annotations

import sys, time
from pathlib import Path
import numpy as np, pandas as pd, catboost as cb, optuna

STRAT = "/mnt/iscsi/hd/xxz/public_release_lightgbm_baseline/examples/lightgbm_baseline"
if STRAT not in sys.path: sys.path.insert(0, STRAT)
from data_utils import manifest_files, feature_columns_from_path  # noqa: E402

DATA_ROOT = Path("/mnt/iscsi/hd/xxz/public_release_20260630/data")
N_TRIALS = 25


def wr2(y,p,w):
    d=float(np.sum(w*y*y)); return 0.0 if d<=0 else 1-float(np.sum(w*(y-p)**2)/d)


def main():
    paths=manifest_files(DATA_ROOT,"train"); feats=feature_columns_from_path(paths[0])
    pf=pd.read_parquet(paths,columns=["time_id","asset_id","weight","target"]+feats)
    pf[feats]=np.nan_to_num(pf[feats].to_numpy(np.float32))
    times=np.sort(pf["time_id"].unique()); ho=set(times[-max(1,int(len(times)*0.15)):].tolist())
    is_va=pf["time_id"].isin(ho).to_numpy(); tr_df,va_df=pf[~is_va].reset_index(drop=True),pf[is_va].reset_index(drop=True)
    print(f"train {len(tr_df):,} / holdout {len(va_df):,}", flush=True)
    ytr=pd.to_numeric(tr_df["target"],errors="coerce").fillna(0).to_numpy(np.float32)
    wtr=pd.to_numeric(tr_df["weight"],errors="coerce").fillna(0).clip(lower=0).to_numpy(np.float32)
    yv=pd.to_numeric(va_df["target"],errors="coerce").fillna(0).to_numpy(np.float64)
    wv=pd.to_numeric(va_df["weight"],errors="coerce").fillna(0).clip(lower=0).to_numpy(np.float64)
    dftr=tr_df[["asset_id"]+feats].copy(); dftr["asset_id"]=dftr["asset_id"].astype(np.int32)
    dfva=va_df[["asset_id"]+feats].copy(); dfva["asset_id"]=dfva["asset_id"].astype(np.int32)
    trpool=cb.Pool(dftr,label=ytr,weight=wtr,cat_features=["asset_id"])
    vapool=cb.Pool(dfva,label=yv,weight=wv,cat_features=["asset_id"])

    def objective(trial):
        p=dict(loss_function="RMSE",task_type="GPU",devices="0",verbose=False,
               learning_rate=trial.suggest_float("lr",0.02,0.1,log=True),
               depth=trial.suggest_int("depth",6,10),
               l2_leaf_reg=trial.suggest_float("l2",0.5,10.0,log=True),
               bagging_temperature=trial.suggest_float("bag",0.0,1.0),
               random_seed=2026, early_stopping_rounds=50, use_best_model=True, iterations=500)
        m=cb.train(trpool,p,eval_set=vapool,verbose=False)
        return wr2(yv, m.predict(vapool), wv)

    study=optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=2026))
    study.optimize(objective,n_trials=N_TRIALS,show_progress_bar=False)
    print(f"\nbest holdout = {study.best_value:+.5f} (原CB 0.00163)", flush=True)
    print("best params:",study.best_params,flush=True)


if __name__=="__main__":
    main()
