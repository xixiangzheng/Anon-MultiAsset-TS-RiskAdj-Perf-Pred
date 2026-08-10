"""迁移/过拟合检查：best vs base 参数，在未参与调参的 part1+2 上对比。

若 best 在 part1+2 上也明显优于 base，则 tuned 参数可迁移、非过拟合 part0。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

STRAT = "/mnt/iscsi/hd/xxz/public_release_lightgbm_baseline/examples/lightgbm_baseline"
if STRAT not in sys.path:
    sys.path.insert(0, STRAT)

from data_utils import manifest_files, feature_columns_from_path  # noqa: E402
from validation import make_validation_plan, weighted_zero_mean_r2  # noqa: E402

DATA_ROOT = Path("/mnt/iscsi/hd/xxz/public_release_20260630/data")
SEEDS = [2026, 2027, 2028]
BEST = dict(learning_rate=0.010326965981106629, num_leaves=79, min_data_in_leaf=556,
            feature_fraction=0.5974233229491067, bagging_fraction=0.9445362670741704,
            bagging_freq=1, lambda_l1=0.03778653953330111, lambda_l2=2.9757802078489703, max_bin=127)
BASE = dict(learning_rate=0.05, num_leaves=63, min_data_in_leaf=2000, feature_fraction=0.8,
            bagging_fraction=0.8, bagging_freq=1, lambda_l1=0.0, lambda_l2=10.0, max_bin=255)


def feval_wr2(p, ds):
    y = ds.get_label(); w = ds.get_weight()
    if w is None: w = np.ones_like(y)
    d = float(np.sum(w * y * y)); s = 0.0 if d <= 0 else 1 - float(np.sum(w * (y - p) ** 2) / d)
    return ("wr2", float(s), True)


def run(X, y, w, tids, tuned, nbr=500, es=60):
    plan = make_validation_plan(pd.Series(tids), n_splits=5, holdout_fraction=0.15, purge_steps=30)
    out = []
    for sd in SEEDS:
        base = BEST if tuned else BASE
        p = dict(objective="regression", metric="None", verbosity=-1, num_threads=16, seed=sd,
                bagging_seed=sd, feature_fraction_seed=sd, data_random_seed=sd, **base)
        oof = np.zeros(len(y)); mask = np.zeros(len(y), bool)
        for f in plan.folds:
            tr = np.isin(tids, list(map(int, f.train_time_ids))); va = np.isin(tids, list(map(int, f.valid_time_ids)))
            dtr = lgb.Dataset(X[tr], label=y[tr], weight=w[tr], categorical_feature=[0], free_raw_data=False)
            dva = lgb.Dataset(X[va], label=y[va], weight=w[va], reference=dtr, free_raw_data=False)
            m = lgb.train(p, dtr, num_boost_round=nbr, valid_sets=[dva], valid_names=["va"], feval=feval_wr2,
                          callbacks=[lgb.early_stopping(es, verbose=False), lgb.log_evaluation(0)])
            oof[va] = m.predict(X[va], num_iteration=m.best_iteration or nbr); mask[va] = True
        out.append(weighted_zero_mean_r2(y[mask], oof[mask], w[mask]))
    return float(np.mean(out)), float(np.std(out))


def main():
    paths = manifest_files(DATA_ROOT, "train")
    feats = feature_columns_from_path(paths[0])
    pf = pd.read_parquet(paths[1:3], columns=["time_id","asset_id","weight","target"]+feats)
    pf[feats] = np.nan_to_num(pf[feats].to_numpy(np.float32))
    print(f"part1+2: {len(pf):,} rows", flush=True)
    y = pd.to_numeric(pf["target"], errors="coerce").fillna(0).to_numpy(np.float32)
    w = pd.to_numeric(pf["weight"], errors="coerce").fillna(0).clip(lower=0).to_numpy(np.float32)
    tids = pf["time_id"].to_numpy()
    X = np.column_stack([pf["asset_id"].to_numpy(np.float32), pf[feats].astype(np.float32).to_numpy()])
    for name, tuned in [("base/raw/part1+2", False), ("best/raw/part1+2", True)]:
        mean, std = run(X, y, w, tids, tuned)
        print(f"{name:22s} {mean:+.5f} {std:.5f}", flush=True)


if __name__ == "__main__":
    main()
