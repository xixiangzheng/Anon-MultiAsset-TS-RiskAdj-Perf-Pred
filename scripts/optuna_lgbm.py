"""Optuna 搜 LGBM 超参（raw 特征，1 分区，purged 5-fold，单 seed）。

baseline 只试 4 个手设配置；本搜索覆盖更广空间，寻找能稳定提升 oof(基线~0.00203) 的配置。
单 seed/单分区求快，最优配置后续用多 seed + 全量验证。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna

STRAT = "/mnt/iscsi/hd/xxz/public_release_lightgbm_baseline/examples/lightgbm_baseline"
if STRAT not in sys.path:
    sys.path.insert(0, STRAT)

from data_utils import manifest_files, feature_columns_from_path  # noqa: E402
from validation import make_validation_plan, weighted_zero_mean_r2  # noqa: E402

DATA_ROOT = Path("/mnt/iscsi/hd/xxz/public_release_20260630/data")
SEED = 2026
N_TRIALS = 30


def feval_wr2(p, ds):
    y = ds.get_label(); w = ds.get_weight()
    if w is None: w = np.ones_like(y)
    d = float(np.sum(w * y * y)); s = 0.0 if d <= 0 else 1 - float(np.sum(w * (y - p) ** 2) / d)
    return ("wr2", float(s), True)


def main():
    paths = manifest_files(DATA_ROOT, "train")[:1]
    feats = feature_columns_from_path(paths[0])
    pf = pd.read_parquet(paths[0], columns=["time_id", "asset_id", "weight", "target"] + feats)
    pf[feats] = np.nan_to_num(pf[feats].to_numpy(np.float32))
    print(f"loaded {len(pf):,} rows", flush=True)
    y = pd.to_numeric(pf["target"], errors="coerce").fillna(0).to_numpy(np.float32)
    w = pd.to_numeric(pf["weight"], errors="coerce").fillna(0).clip(lower=0).to_numpy(np.float32)
    tids = pf["time_id"].to_numpy()
    asset = pf["asset_id"].to_numpy(np.float32)
    X = np.column_stack([asset, pf[feats].astype(np.float32).to_numpy()])
    plan = make_validation_plan(pd.Series(tids), n_splits=5, holdout_fraction=0.15, purge_steps=30)
    folds = [(np.isin(tids, list(map(int, f.train_time_ids))), np.isin(tids, list(map(int, f.valid_time_ids)))) for f in plan.folds]

    def objective(trial):
        lr = trial.suggest_float("learning_rate", 0.01, 0.1, log=True)
        num_leaves = trial.suggest_int("num_leaves", 16, 127)
        min_data = trial.suggest_int("min_data_in_leaf", 100, 8000, log=True)
        ff = trial.suggest_float("feature_fraction", 0.5, 1.0)
        bf = trial.suggest_float("bagging_fraction", 0.5, 1.0)
        bg_freq = trial.suggest_categorical("bagging_freq", [0, 1])
        l1 = trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True)
        l2 = trial.suggest_float("lambda_l2", 0.1, 50.0, log=True)
        max_bin = trial.suggest_categorical("max_bin", [127, 255, 511])
        params = dict(objective="regression", metric="None", learning_rate=lr, num_leaves=num_leaves,
                     min_data_in_leaf=min_data, feature_fraction=ff, bagging_fraction=bf, bagging_freq=bg_freq,
                     lambda_l1=l1, lambda_l2=l2, max_bin=max_bin, verbosity=-1, num_threads=16, seed=SEED,
                     bagging_seed=SEED, feature_fraction_seed=SEED, data_random_seed=SEED)
        oof = np.zeros(len(y)); mask = np.zeros(len(y), bool)
        for tr, va in folds:
            dtr = lgb.Dataset(X[tr], label=y[tr], weight=w[tr], categorical_feature=[0], free_raw_data=False)
            dva = lgb.Dataset(X[va], label=y[va], weight=w[va], reference=dtr, free_raw_data=False)
            m = lgb.train(params, dtr, num_boost_round=500, valid_sets=[dva], valid_names=["va"], feval=feval_wr2,
                          callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)])
            oof[va] = m.predict(X[va], num_iteration=m.best_iteration or 500); mask[va] = True
        return weighted_zero_mean_r2(y[mask], oof[mask], w[mask])

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    print(f"\nbaseline-ref oof ≈ +0.00203", flush=True)
    print(f"best oof = {study.best_value:+.5f}  (Δ {study.best_value-0.00203:+.5f})", flush=True)
    print("best params:", flush=True)
    for k, v in study.best_params.items():
        print(f"  {k}: {v}", flush=True)
    # 全部 trial 概览
    trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else -9, reverse=True)
    print("\ntop 8 trials:", flush=True)
    for t in trials[:8]:
        print(f"  {t.value:+.5f}  {t.params}", flush=True)


if __name__ == "__main__":
    main()
