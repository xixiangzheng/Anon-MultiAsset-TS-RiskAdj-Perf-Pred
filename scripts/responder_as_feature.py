"""重检：predicted responder 作为树模型特征（未验证的最大潜在增益）。

关键：responder OOF 预测是无泄漏的 stacking 特征（每行的预测来自未见过它的模型）。
对比：
  E1 raw（参考）
  E2 raw + 6 个可预测 responder 的 OOF 预测
  E3 raw + baseline 历史(top48, win5)
  E4 raw + 历史 + responder OOF
每配置 3 seed。看 responder 作特征能否给出 >+0.0003 的稳定增益。
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

from data_utils import manifest_files, feature_columns_from_path, top_correlated_features, add_group_history_features  # noqa: E402
from validation import make_validation_plan, weighted_zero_mean_r2  # noqa: E402

DATA_ROOT = Path("/mnt/iscsi/hd/xxz/public_release_20260630/data")
SEEDS = [2026, 2027, 2028]
# 可预测的 responder（probe 测得 self R²>0.4）
RESPS = ["responder_02", "responder_03", "responder_11", "responder_17", "responder_18", "responder_19"]


def feval_wr2(p, ds):
    y = ds.get_label(); w = ds.get_weight()
    if w is None: w = np.ones_like(y)
    d = float(np.sum(w * y * y)); s = 0.0 if d <= 0 else 1 - float(np.sum(w * (y - p) ** 2) / d)
    return ("wr2", float(s), True)


def get_params(seed):
    return dict(objective="regression", metric="None", learning_rate=0.05, num_leaves=63,
               min_data_in_leaf=2000, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
               lambda_l2=10.0, verbosity=-1, num_threads=16, seed=seed,
               bagging_seed=seed, feature_fraction_seed=seed, data_random_seed=seed)


def cv_oof_target(X, y, w, tids, seeds, nbr=250, es=40):
    """target 模型多 seed OOF，返回每 seed 的 R² 列表。"""
    plan = make_validation_plan(pd.Series(tids), n_splits=5, holdout_fraction=0.15, purge_steps=30)
    out = []
    for sd in seeds:
        oof = np.zeros(len(y)); mask = np.zeros(len(y), bool)
        p = get_params(sd)
        for f in plan.folds:
            tr = np.isin(tids, list(map(int, f.train_time_ids))); va = np.isin(tids, list(map(int, f.valid_time_ids)))
            dtr = lgb.Dataset(X[tr], label=y[tr], weight=w[tr], categorical_feature=[0], free_raw_data=False)
            dva = lgb.Dataset(X[va], label=y[va], weight=w[va], reference=dtr, free_raw_data=False)
            m = lgb.train(p, dtr, num_boost_round=nbr, valid_sets=[dva], valid_names=["va"], feval=feval_wr2,
                          callbacks=[lgb.early_stopping(es, verbose=False), lgb.log_evaluation(0)])
            oof[va] = m.predict(X[va], num_iteration=m.best_iteration or nbr); mask[va] = True
        out.append(weighted_zero_mean_r2(y[mask], oof[mask], w[mask]))
    return out


def responder_oof(Xfeat, resp_arrays, w, tids, seeds, nbr=300, es=40):
    """对每个 responder 用每 seed 做 OOF 预测，跨 seed 平均，返回 [n_rows, n_resp] 特征矩阵。"""
    plan = make_validation_plan(pd.Series(tids), n_splits=5, holdout_fraction=0.15, purge_steps=30)
    n = Xfeat.shape[0]; R = len(resp_arrays)
    accum = np.zeros((n, R), dtype=np.float64)
    for sd in seeds:
        p = get_params(sd)
        tmp = np.zeros((n, R), dtype=np.float64)
        for f in plan.folds:
            tr = np.isin(tids, list(map(int, f.train_time_ids))); va = np.isin(tids, list(map(int, f.valid_time_ids)))
            # 用 weight（responder 也按 weight 加权拟合，近似）
            for j, yr in enumerate(resp_arrays):
                dtr = lgb.Dataset(Xfeat[tr], label=yr[tr], weight=w[tr], categorical_feature=[0], free_raw_data=False)
                dva = lgb.Dataset(Xfeat[va], label=yr[va], weight=w[va], reference=dtr, free_raw_data=False)
                m = lgb.train(p, dtr, num_boost_round=nbr, valid_sets=[dva], valid_names=["va"], feval=feval_wr2,
                              callbacks=[lgb.early_stopping(es, verbose=False), lgb.log_evaluation(0)])
                tmp[va, j] = m.predict(Xfeat[va], num_iteration=m.best_iteration or nbr)
        accum += tmp
    return accum / len(seeds)


def main():
    paths = manifest_files(DATA_ROOT, "train")[:1]
    feats = feature_columns_from_path(paths[0])
    cols = ["time_id", "asset_id", "weight", "target"] + RESPS + feats
    pf = pd.read_parquet(paths[0], columns=cols)
    pf[feats] = np.nan_to_num(pf[feats].to_numpy(np.float32))
    print(f"loaded {len(pf):,} rows", flush=True)
    y = pd.to_numeric(pf["target"], errors="coerce").fillna(0).to_numpy(np.float32)
    w = pd.to_numeric(pf["weight"], errors="coerce").fillna(0).clip(lower=0).to_numpy(np.float32)
    tids = pf["time_id"].to_numpy(); asset = pf["asset_id"].to_numpy(np.float32)
    raw = pf[feats].astype(np.float32).to_numpy()
    Xfeat = np.column_stack([asset, raw])  # 用于预测 responder 的特征（asset_id + raw）

    resp_arrays = [pd.to_numeric(pf[r], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float32) for r in RESPS]
    print("computing responder OOF features...", flush=True)
    t0 = time.time()
    resp_feat = responder_oof(Xfeat, resp_arrays, w, tids, SEEDS)
    print(f"  done in {int(time.time()-t0)}s, shape={resp_feat.shape}", flush=True)

    # 历史 top-48
    sel48 = top_correlated_features(pf[["time_id","asset_id","target","weight"]+feats], feats, top_k=48, sample_rows=200_000, seed=2026)
    sub = pf[["time_id","asset_id","target","weight"]+feats].copy()
    eng, _ = add_group_history_features(sub, sel48, rolling_windows=(5,))
    hist_cols = [c for c in eng.columns if c.startswith(("lag1_","diff1_","rmean"))]
    hist = eng[hist_cols].astype(np.float32).to_numpy()

    configs = [
        ("E1_raw",          raw),
        ("E2_raw+resp",     np.column_stack([raw, resp_feat])),
        ("E3_raw+hist",     np.column_stack([raw, hist])),
        ("E4_raw+hist+resp", np.column_stack([raw, hist, resp_feat])),
    ]
    print(f"\n{'exp':18s} {'nfeat':>5s} {'mean':>9s} {'std':>7s} {'delta':>8s}", flush=True)
    ref = None
    for name, fm in configs:
        X = np.column_stack([asset, fm])
        per = cv_oof_target(X, y, w, tids, SEEDS)
        mean = float(np.mean(per)); std = float(np.std(per))
        if ref is None: ref = mean
        print(f"{name:18s} {X.shape[1]:>5d} {mean:+.5f} {std:.5f} {mean-ref:+.5f}", flush=True)


if __name__ == "__main__":
    main()
