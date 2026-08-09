"""集成探测：LGBM vs XGB 预测相关性 + 集成 R²。

1 分区，purged 5-fold OOF。回答：不同 GBDM 族是否去相关到足以让集成获益？
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb

STRAT = "/mnt/iscsi/hd/xxz/public_release_lightgbm_baseline/examples/lightgbm_baseline"
if STRAT not in sys.path:
    sys.path.insert(0, STRAT)

from data_utils import manifest_files, feature_columns_from_path  # noqa: E402
from validation import make_validation_plan, weighted_zero_mean_r2  # noqa: E402

DATA_ROOT = Path("/mnt/iscsi/hd/xxz/public_release_20260630/data")
SEED = 2026


def wr2(y, p, w):
    return weighted_zero_mean_r2(y, p, w)


def main():
    paths = manifest_files(DATA_ROOT, "train")[:1]
    feats = feature_columns_from_path(paths[0])
    pf = pd.read_parquet(paths[0], columns=["time_id", "asset_id", "weight", "target"] + feats)
    pf[feats] = np.nan_to_num(pf[feats].to_numpy(np.float32))
    print(f"loaded {len(pf):,} rows", flush=True)
    y = pd.to_numeric(pf["target"], errors="coerce").fillna(0).to_numpy(np.float32)
    w = pd.to_numeric(pf["weight"], errors="coerce").fillna(0).clip(lower=0).to_numpy(np.float32)
    tids = pf["time_id"].to_numpy()
    asset = pf["asset_id"].to_numpy(np.int32)
    X = np.column_stack([asset.astype(np.float32), pf[feats].astype(np.float32).to_numpy()])
    n = len(pf)

    plan = make_validation_plan(pd.Series(tids), n_splits=5, holdout_fraction=0.15, purge_steps=30)

    # 列名（asset_id + features）
    feat_names = ["asset_id"] + feats

    oof_lgb = np.zeros(n)
    oof_xgb = np.zeros(n)
    mask = np.zeros(n, bool)

    lgb_params = dict(objective="regression", metric="None", learning_rate=0.05, num_leaves=63,
                     min_data_in_leaf=2000, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                     lambda_l2=10.0, verbosity=-1, num_threads=16, seed=SEED,
                     bagging_seed=SEED, feature_fraction_seed=SEED, data_random_seed=SEED)

    def lgb_feval(p, ds):
        yy = ds.get_label(); ww = ds.get_weight()
        if ww is None: ww = np.ones_like(yy)
        d = float(np.sum(ww * yy * yy)); s = 0.0 if d <= 0 else 1 - float(np.sum(ww * (yy - p) ** 2) / d)
        return ("wr2", float(s), True)

    xgb_params = dict(objective="reg:squarederror", learning_rate=0.05, max_depth=8,
                      min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                      reg_lambda=10.0, tree_method="hist", nthread=16, seed=SEED, verbosity=0)

    def xgb_wr2(preds, dtrain):
        yy = dtrain.get_label()
        ww = dtrain.get_weight()
        if ww is None: ww = np.ones_like(yy)
        d = float(np.sum(ww * yy * yy)); s = 0.0 if d <= 0 else 1 - float(np.sum(ww * (yy - preds) ** 2) / d)
        return ("wr2", float(s))

    for k, f in enumerate(plan.folds):
        tr = np.isin(tids, list(map(int, f.train_time_ids)))
        va = np.isin(tids, list(map(int, f.valid_time_ids)))
        # LGBM
        dtr = lgb.Dataset(X[tr], label=y[tr], weight=w[tr], feature_name=feat_names, categorical_feature=["asset_id"], free_raw_data=False)
        dva = lgb.Dataset(X[va], label=y[va], weight=w[va], reference=dtr, free_raw_data=False)
        m = lgb.train(lgb_params, dtr, num_boost_round=250, valid_sets=[dva], valid_names=["va"],
                      feval=lgb_feval, callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)])
        bi = m.best_iteration or 250
        oof_lgb[va] = m.predict(X[va], num_iteration=bi)
        # XGB
        dtr_x = xgb.DMatrix(X[tr], label=y[tr], weight=w[tr], feature_names=feat_names)
        dva_x = xgb.DMatrix(X[va], label=y[va], weight=w[va], feature_names=feat_names)
        mx = xgb.train(xgb_params, dtr_x, num_boost_round=250, evals=[(dva_x, "va")], feval=xgb_wr2,
                       early_stopping_rounds=40, verbose_eval=False)
        oof_xgb[va] = mx.predict(dva_x, iteration_range=(0, mx.best_iteration + 1))
        mask[va] = True
        print(f"  fold {k} done", flush=True)

    ym, wm, om = y[mask], w[mask], mask
    rl = wr2(y[mask], oof_lgb[mask], w[mask])
    rx = wr2(y[mask], oof_xgb[mask], w[mask])
    corr = float(np.corrcoef(oof_lgb[mask], oof_xgb[mask])[0, 1])
    # 平均集成
    ravg = wr2(y[mask], 0.5 * (oof_lgb + oof_xgb)[mask], w[mask])
    # 最优混合系数 a*（加权 R² 下扫描）
    best_a, best_r = 0.5, ravg
    for a in np.arange(0.0, 1.01, 0.05):
        r = wr2(y[mask], (a * oof_lgb + (1 - a) * oof_xgb)[mask], w[mask])
        if r > best_r:
            best_a, best_r = a, r
    print(f"\nLGBM   oof R² = {rl:+.5f}", flush=True)
    print(f"XGB    oof R² = {rx:+.5f}", flush=True)
    print(f"corr(lgb,xgb) = {corr:.4f}", flush=True)
    print(f"avg 0.5/0.5   = {ravg:+.5f}  (Δ vs best-single {ravg-max(rl,rx):+.5f})", flush=True)
    print(f"best blend a={best_a:.2f} = {best_r:+.5f}  (Δ vs best-single {best_r-max(rl,rx):+.5f})", flush=True)


if __name__ == "__main__":
    main()
