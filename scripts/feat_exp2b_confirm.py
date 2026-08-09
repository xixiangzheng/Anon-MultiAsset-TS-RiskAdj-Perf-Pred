"""第二批(精简版)：1 分区，3 seed，确认截面特征是否稳健。

仅对比 A_raw vs C_raw+cs(top48) vs C_raw+cs(all323)，每配置 3 seed 取 mean±std。
目的：判定 cross-sectional 的 +Δ 是否在噪声之上。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

STRAT = "/mnt/iscsi/hd/xxz/public_release_lightgbm_baseline/examples/lightgbm_baseline"
if STRAT not in sys.path:
    sys.path.insert(0, STRAT)

from data_utils import manifest_files, feature_columns_from_path, top_correlated_features  # noqa: E402
from validation import make_validation_plan, weighted_zero_mean_r2  # noqa: E402

DATA_ROOT = Path("/mnt/iscsi/hd/xxz/public_release_20260630/data")
SEEDS = [2026, 2027, 2028]


def get_params(seed):
    return dict(objective="regression", metric="None", learning_rate=0.05, num_leaves=63,
               min_data_in_leaf=2000, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
               lambda_l2=10.0, verbosity=-1, num_threads=16, seed=seed,
               bagging_seed=seed, feature_fraction_seed=seed, data_random_seed=seed)


def feval_wr2(preds, ds):
    y = ds.get_label(); w = ds.get_weight()
    if w is None: w = np.ones_like(y)
    d = float(np.sum(w * y * y))
    s = 0.0 if d <= 0 else 1.0 - float(np.sum(w * (y - preds) ** 2) / d)
    return ("wr2", float(s), True)


def cs(frame, feats, mode):
    g = frame.groupby("time_id")[feats]
    parts = []
    if mode in ("both", "rank"):
        r = g.rank(pct=True); r.columns = [f"csr_{c}" for c in feats]; parts.append(r)
    if mode in ("both", "z"):
        m = g.transform("mean"); s = g.transform("std").replace(0.0, 1.0)
        z = ((frame[feats] - m) / s).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        z.columns = [f"csz_{c}" for c in feats]; parts.append(z)
    return pd.concat(parts, axis=1).astype(np.float32).to_numpy()


def cv_oof(X, y, w, tids, seed, nbr=250, es=40):
    plan = make_validation_plan(pd.Series(tids), n_splits=5, holdout_fraction=0.15, purge_steps=30)
    oof = np.zeros(len(y)); mask = np.zeros(len(y), bool)
    for f in plan.folds:
        tr = np.isin(tids, list(map(int, f.train_time_ids))); va = np.isin(tids, list(map(int, f.valid_time_ids)))
        dtr = lgb.Dataset(X[tr], label=y[tr], weight=w[tr], categorical_feature=[0], free_raw_data=False)
        dva = lgb.Dataset(X[va], label=y[va], weight=w[va], reference=dtr, free_raw_data=False)
        mdl = lgb.train(get_params(seed), dtr, num_boost_round=nbr, valid_sets=[dva], valid_names=["va"],
                        feval=feval_wr2, callbacks=[lgb.early_stopping(es, verbose=False), lgb.log_evaluation(0)])
        bi = mdl.best_iteration or nbr
        oof[va] = mdl.predict(X[va], num_iteration=bi); mask[va] = True
    return weighted_zero_mean_r2(y[mask], oof[mask], w[mask])


def main():
    paths = manifest_files(DATA_ROOT, "train")[:1]
    feats = feature_columns_from_path(paths[0])
    pf = pd.read_parquet(paths[0], columns=["time_id", "asset_id", "weight", "target"] + feats)
    pf[feats] = np.nan_to_num(pf[feats].to_numpy(np.float32))
    print(f"loaded {len(pf):,} rows", flush=True)
    y = pd.to_numeric(pf["target"], errors="coerce").fillna(0).to_numpy(np.float32)
    w = pd.to_numeric(pf["weight"], errors="coerce").fillna(0).clip(lower=0).to_numpy(np.float32)
    tids = pf["time_id"].to_numpy(); asset = pf["asset_id"].to_numpy(np.float32)
    raw = pf[feats].astype(np.float32).to_numpy()
    sel48 = top_correlated_features(pf[["time_id","asset_id","target","weight"]+feats], feats, top_k=48, sample_rows=200_000, seed=2026)
    cs48 = cs(pf, sel48, "both"); cs_all = cs(pf, feats, "both")
    configs = [("A_raw", raw), ("C_cs48", np.column_stack([raw, cs48])), ("C_cs_all", np.column_stack([raw, cs_all]))]
    print(f"\n{'exp':10s} {'nfeat':>5s} {'mean':>9s} {'std':>7s} {'delta':>8s}", flush=True)
    ref = None
    for name, fm in configs:
        per = []
        for sd in SEEDS:
            t0 = time.time(); r2 = cv_oof(np.column_stack([asset, fm]), y, w, tids, sd); per.append(r2)
            print(f"  {name} seed={sd}: {r2:+.5f} ({int(time.time()-t0)}s)", flush=True)
        mean = float(np.mean(per)); std = float(np.std(per))
        if ref is None: ref = mean
        print(f"{name:10s} {fm.shape[1]+1:>5d} {mean:+.5f} {std:.5f} {mean-ref:+.5f}", flush=True)


if __name__ == "__main__":
    main()
