"""第二批：确认截面特征的稳健性 + 找最佳变体。

3 分区数据、每配置 3 seed 取均值±std，对比 A_raw 与若干截面变体。
若截面稳健正向，则方向确立。严守 purged CV。
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
N_PARTITIONS = 3
SEEDS = [2026, 2027, 2028]


def get_params(seed: int) -> dict:
    return dict(
        objective="regression", metric="None", learning_rate=0.05, num_leaves=63,
        min_data_in_leaf=2000, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
        lambda_l2=10.0, verbosity=-1, num_threads=16, seed=seed,
        bagging_seed=seed, feature_fraction_seed=seed, data_random_seed=seed,
    )


def feval_wr2(preds, ds):
    y = ds.get_label()
    w = ds.get_weight()
    if w is None:
        w = np.ones_like(y)
    denom = float(np.sum(w * y * y))
    s = 0.0 if denom <= 0 else 1.0 - float(np.sum(w * (y - preds) ** 2) / denom)
    return ("wr2", float(s), True)


def cs_features(frame, feats, mode):
    """mode: 'both'|'rank'|'z'。截面 pct rank / z-score，groupby time_id。"""
    g = frame.groupby("time_id")[feats]
    parts = []
    if mode in ("both", "rank"):
        r = g.rank(pct=True)
        r.columns = [f"csr_{c}" for c in feats]
        parts.append(r)
    if mode in ("both", "z"):
        mean = g.transform("mean")
        std = g.transform("std").replace(0.0, 1.0)
        z = ((frame[feats] - mean) / std).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        z.columns = [f"csz_{c}" for c in feats]
        parts.append(z)
    return pd.concat(parts, axis=1).astype(np.float32)


def cv_oof(X, y, w, tids, seed, num_boost_round=250, es=40):
    plan = make_validation_plan(pd.Series(tids), n_splits=5, holdout_fraction=0.15, purge_steps=30)
    oof = np.zeros(len(y), dtype=np.float64)
    mask = np.zeros(len(y), dtype=bool)
    for f in plan.folds:
        tr = np.isin(tids, list(map(int, f.train_time_ids)))
        va = np.isin(tids, list(map(int, f.valid_time_ids)))
        dtr = lgb.Dataset(X[tr], label=y[tr], weight=w[tr], categorical_feature=[0], free_raw_data=False)
        dva = lgb.Dataset(X[va], label=y[va], weight=w[va], reference=dtr, free_raw_data=False)
        m = lgb.train(get_params(seed), dtr, num_boost_round=num_boost_round, valid_sets=[dva], valid_names=["va"],
                      feval=feval_wr2, callbacks=[lgb.early_stopping(es, verbose=False), lgb.log_evaluation(0)])
        bi = m.best_iteration or num_boost_round
        oof[va] = m.predict(X[va], num_iteration=bi)
        mask[va] = True
    return weighted_zero_mean_r2(y[mask], oof[mask], w[mask])


def main():
    paths = manifest_files(DATA_ROOT, "train")[:N_PARTITIONS]
    feats = feature_columns_from_path(paths[0])
    cols = ["time_id", "asset_id", "weight", "target"] + feats
    pf = pd.read_parquet(paths, columns=cols)  # 多分区
    pf[feats] = np.nan_to_num(pf[feats].to_numpy(np.float32))
    print(f"loaded {len(pf):,} rows ({N_PARTITIONS} partitions), {len(feats)} feats", flush=True)

    y = pd.to_numeric(pf["target"], errors="coerce").fillna(0.0).to_numpy(np.float32)
    w = pd.to_numeric(pf["weight"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(np.float32)
    tids = pf["time_id"].to_numpy()
    asset = pf["asset_id"].to_numpy(np.float32)
    raw = pf[feats].astype(np.float32).to_numpy()

    # top-48 特征（按与 target 相关）
    sub = pf[["time_id", "asset_id", "target", "weight"] + feats]
    sel48 = top_correlated_features(sub, feats, top_k=48, sample_rows=300_000, seed=2026)
    print(f"top-48 selected", flush=True)

    cs48_both = cs_features(pf, sel48, "both").to_numpy()
    cs48_rank = cs_features(pf, sel48, "rank").to_numpy()
    cs48_z = cs_features(pf, sel48, "z").to_numpy()
    cs_all = cs_features(pf, feats, "both").to_numpy()

    configs = [
        ("A_raw",            raw),
        ("C_cs48_both",      np.column_stack([raw, cs48_both])),
        ("C_cs48_rank",      np.column_stack([raw, cs48_rank])),
        ("C_cs48_z",         np.column_stack([raw, cs48_z])),
        ("C_cs_all_both",    np.column_stack([raw, cs_all])),
    ]

    print(f"\n{'exp':16s} {'nfeat':>5s} {'mean oof':>9s} {'std':>7s} {'delta':>8s}", flush=True)
    ref_mean = None
    for name, feat_mat in configs:
        per_seed = []
        for sd in SEEDS:
            X = np.column_stack([asset, feat_mat])
            t0 = time.time()
            r2 = cv_oof(X, y, w, tids, sd)
            per_seed.append(r2)
            print(f"  {name} seed={sd}: {r2:+.5f} ({int(time.time()-t0)}s)", flush=True)
        mean = float(np.mean(per_seed))
        std = float(np.std(per_seed))
        if ref_mean is None:
            ref_mean = mean
        print(f"{name:16s} {X.shape[1]:>5d} {mean:+.5f} {std:.5f} {mean-ref_mean:+.5f}", flush=True)


if __name__ == "__main__":
    main()
