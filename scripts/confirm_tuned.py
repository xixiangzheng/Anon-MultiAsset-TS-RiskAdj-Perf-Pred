"""确认 Optuna 最优参数的稳健性 + 迁移 + history 叠加。

5 配置 × 3 seed，回答：
  C1 best-params × raw × 分区0   —— 多 seed 稳健?
  C2 best-params × raw × 分区1+2 —— 迁移到未见分区(过拟合检查)?
  C3 best-params × raw+hist × 分区0 —— tuned 参数下 history 是否额外加成?
  C4 baseline-params × raw × 分区0 —— 同条件对照
  C5 baseline-params × raw+hist × 分区0 —— baseline 配置对照
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

from data_utils import manifest_files, feature_columns_from_path, top_correlated_features, add_group_history_features  # noqa: E402
from validation import make_validation_plan, weighted_zero_mean_r2  # noqa: E402

DATA_ROOT = Path("/mnt/iscsi/hd/xxz/public_release_20260630/data")
SEEDS = [2026, 2027, 2028]

BEST = dict(learning_rate=0.010326965981106629, num_leaves=79, min_data_in_leaf=556,
            feature_fraction=0.5974233229491067, bagging_fraction=0.9445362670741704,
            bagging_freq=1, lambda_l1=0.03778653953330111, lambda_l2=2.9757802078489703, max_bin=127)
BASE = dict(learning_rate=0.05, num_leaves=63, min_data_in_leaf=2000,
            feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
            lambda_l1=0.0, lambda_l2=10.0, max_bin=255)  # baseline 近似配置


def feval_wr2(p, ds):
    y = ds.get_label(); w = ds.get_weight()
    if w is None: w = np.ones_like(y)
    d = float(np.sum(w * y * y)); s = 0.0 if d <= 0 else 1 - float(np.sum(w * (y - p) ** 2) / d)
    return ("wr2", float(s), True)


def make_params(tuned, seed):
    base = BEST if tuned else BASE
    return dict(objective="regression", metric="None", verbosity=-1, num_threads=16, seed=seed,
               bagging_seed=seed, feature_fraction_seed=seed, data_random_seed=seed, **base)


def run(X, y, w, tids, tuned, seeds, nbr=500, es=60):
    plan = make_validation_plan(pd.Series(tids), n_splits=5, holdout_fraction=0.15, purge_steps=30)
    out = []
    for sd in seeds:
        p = make_params(tuned, sd); oof = np.zeros(len(y)); mask = np.zeros(len(y), bool)
        for f in plan.folds:
            tr = np.isin(tids, list(map(int, f.train_time_ids))); va = np.isin(tids, list(map(int, f.valid_time_ids)))
            dtr = lgb.Dataset(X[tr], label=y[tr], weight=w[tr], categorical_feature=[0], free_raw_data=False)
            dva = lgb.Dataset(X[va], label=y[va], weight=w[va], reference=dtr, free_raw_data=False)
            m = lgb.train(p, dtr, num_boost_round=nbr, valid_sets=[dva], valid_names=["va"], feval=feval_wr2,
                          callbacks=[lgb.early_stopping(es, verbose=False), lgb.log_evaluation(0)])
            oof[va] = m.predict(X[va], num_iteration=m.best_iteration or nbr); mask[va] = True
        out.append(weighted_zero_mean_r2(y[mask], oof[mask], w[mask]))
    return float(np.mean(out)), float(np.std(out))


def load(parts, feats):
    cols = ["time_id", "asset_id", "weight", "target"] + feats
    pf = pd.read_parquet(parts, columns=cols)
    pf[feats] = np.nan_to_num(pf[feats].to_numpy(np.float32))
    y = pd.to_numeric(pf["target"], errors="coerce").fillna(0).to_numpy(np.float32)
    w = pd.to_numeric(pf["weight"], errors="coerce").fillna(0).clip(lower=0).to_numpy(np.float32)
    tids = pf["time_id"].to_numpy(); asset = pf["asset_id"].to_numpy(np.float32)
    raw = pf[feats].astype(np.float32).to_numpy()
    sel48 = top_correlated_features(pf[["time_id","asset_id","target","weight"]+feats], feats, top_k=48, sample_rows=200_000, seed=2026)
    sub = pf[["time_id","asset_id","target","weight"]+feats].copy()
    eng, _ = add_group_history_features(sub, sel48, rolling_windows=(5,))
    hc = [c for c in eng.columns if c.startswith(("lag1_","diff1_","rmean"))]
    hist = eng[hc].astype(np.float32).to_numpy()
    return pf, y, w, tids, asset, raw, hist


def main():
    paths = manifest_files(DATA_ROOT, "train")
    feats = feature_columns_from_path(paths[0])
    p0 = paths[:1]; p12 = paths[1:3]
    print("loading partition 0...", flush=True)
    _, y0, w0, t0, a0, raw0, hist0 = load(p0, feats)
    print(f"  part0 {len(y0):,} rows. loading part1+2...", flush=True)
    _, y12, w12, t12, a12, raw12, hist12 = load(p12, feats)
    print(f"  part1+2 {len(y12):,} rows", flush=True)

    configs = [
        ("C1 best/raw/part0",      True,  a0, raw0, y0, w0, t0),
        ("C2 best/raw/part1+2",    True,  a12, raw12, y12, w12, t12),
        ("C3 best/raw+hist/part0", True,  a0, np.column_stack([raw0, hist0]), y0, w0, t0),
        ("C4 base/raw/part0",      False, a0, raw0, y0, w0, t0),
        ("C5 base/raw+hist/part0", False, a0, np.column_stack([raw0, hist0]), y0, w0, t0),
    ]
    print(f"\n{'exp':26s} {'mean':>9s} {'std':>7s}", flush=True)
    for name, tuned, asset, fm, y, w, tids in configs:
        X = np.column_stack([asset, fm])
        mean, std = run(X, y, w, tids, tuned, SEEDS)
        print(f"{name:26s} {mean:+.5f} {std:.5f}", flush=True)


if __name__ == "__main__":
    main()
