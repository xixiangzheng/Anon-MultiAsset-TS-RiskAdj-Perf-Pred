"""特征工程实验框架（小数据 purged 5-fold CV 衡量 Δ）。

在 1 个训练分区上，固定模型/purged CV，对比不同特征组合对 target 的 OOF 加权 R²。
目的：寻找能稳定提升 baseline(~0.002) 的特征方向（截面/多尺度/波动率）。

复用 baseline 的 add_group_history_features / make_validation_plan / weighted_zero_mean_r2。
所有特征严格因果：历史特征只用过去；截面特征只用当前 time_id 内跨标的信息（非未来）。
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

from data_utils import add_group_history_features, top_correlated_features  # noqa: E402
from validation import make_validation_plan, weighted_zero_mean_r2  # noqa: E402

DATA_ROOT = Path("/mnt/iscsi/hd/xxz/public_release_20260630/data")
PARAMS = dict(
    objective="regression", metric="None", learning_rate=0.05, num_leaves=63,
    min_data_in_leaf=2000, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
    lambda_l2=10.0, verbosity=-1, num_threads=16, seed=2026,
    bagging_seed=2026, feature_fraction_seed=2026, data_random_seed=2026,
)


def feval_wr2(preds, ds):
    y = ds.get_label()
    w = ds.get_weight()
    if w is None:
        w = np.ones_like(y)
    denom = float(np.sum(w * y * y))
    s = 0.0 if denom <= 0 else 1.0 - float(np.sum(w * (y - preds) ** 2) / denom)
    return ("wr2", float(s), True)


# ---------- 特征构建器（都因果）----------
def build_raw(frame, feats):
    return frame[feats].astype(np.float32)


def build_history(frame, feats, top_k, windows, target="target"):
    """top-K(按与 target 相关) 原始特征 → lag1/diff1/rmean{w}。"""
    sub = frame[["time_id", "asset_id", target, "weight"] + feats].copy()
    sel = top_correlated_features(sub, feats, top_k=top_k, sample_rows=200_000, seed=2026)
    eng, _ = add_group_history_features(sub, sel, rolling_windows=tuple(windows))
    cols = [c for c in eng.columns if c.startswith(("lag1_", "diff1_", "rmean"))]
    return eng[cols].astype(np.float32), sel


def build_cross_sectional(frame, feats):
    """跨标的截面：每个 time_id 内的 pct rank + z-score（只用当前时刻横向信息）。"""
    g = frame.groupby("time_id")[feats]
    rank = g.rank(pct=True)
    rank.columns = [f"csr_{c}" for c in feats]
    mean = g.transform("mean")
    std = g.transform("std").replace(0.0, 1.0)
    z = (frame[feats] - mean) / std
    z = z.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    z.columns = [f"csz_{c}" for c in feats]
    return pd.concat([rank, z], axis=1).astype(np.float32)


def build_volatility(frame, feats, window):
    """每标的滚动 std（波动率），因果。"""
    vals = frame[feats].to_numpy(np.float32)
    vals = np.nan_to_num(vals)
    aids = frame["asset_id"].to_numpy()
    order = np.lexsort((frame["time_id"].to_numpy(), aids))
    restore = np.empty(len(order), dtype=np.int64)
    restore[order] = np.arange(len(order))
    ao = aids[order]
    vo = vals[order]
    out = np.zeros_like(vo)
    bounds = np.flatnonzero(np.r_[True, ao[1:] != ao[:-1], True])
    for s, e in zip(bounds[:-1], bounds[1:]):
        grp = vo[s:e].astype(np.float64)
        cs = np.vstack([np.zeros((1, grp.shape[1])), np.cumsum(grp, axis=0)])
        cs2 = np.vstack([np.zeros((1, grp.shape[1])), np.cumsum(grp * grp, axis=0)])
        offs = np.arange(e - s)
        ws = np.maximum(0, offs + 1 - window)
        n = (offs + 1 - ws).reshape(-1, 1)
        mean = (cs[offs + 1] - cs[ws]) / n
        e2 = (cs2[offs + 1] - cs2[ws]) / n
        var = np.maximum(e2 - mean * mean, 0.0)
        out[s:e] = np.sqrt(var).astype(np.float32)
    out = out[restore]
    return pd.DataFrame(out, columns=[f"rstd{window}_{c}" for c in feats], index=frame.index)


# ---------- CV ----------
def cv_oof(X, y, w, time_ids, num_boost_round=250, es=40):
    plan = make_validation_plan(pd.Series(time_ids), n_splits=5, holdout_fraction=0.15, purge_steps=30)
    tids = time_ids
    oof = np.zeros(len(y), dtype=np.float64)
    mask = np.zeros(len(y), dtype=bool)
    for f in plan.folds:
        tr = np.isin(tids, list(map(int, f.train_time_ids)))
        va = np.isin(tids, list(map(int, f.valid_time_ids)))
        dtr = lgb.Dataset(X[tr], label=y[tr], weight=w[tr], categorical_feature=[0], free_raw_data=False)
        dva = lgb.Dataset(X[va], label=y[va], weight=w[va], reference=dtr, free_raw_data=False)
        m = lgb.train(PARAMS, dtr, num_boost_round=num_boost_round, valid_sets=[dva], valid_names=["va"],
                      feval=feval_wr2, callbacks=[lgb.early_stopping(es, verbose=False), lgb.log_evaluation(0)])
        bi = m.best_iteration or num_boost_round
        oof[va] = m.predict(X[va], num_iteration=bi)
        mask[va] = True
    return weighted_zero_mean_r2(y[mask], oof[mask], w[mask])


def main():
    from data_utils import manifest_files, feature_columns_from_path
    paths = manifest_files(DATA_ROOT, "train")
    feats = feature_columns_from_path(paths[0])
    cols = ["time_id", "asset_id", "weight", "target"] + feats
    pf = pd.read_parquet(paths[0], columns=cols)
    pf[feats] = np.nan_to_num(pf[feats].to_numpy(np.float32))
    print(f"loaded {len(pf):,} rows, {len(feats)} feats", flush=True)

    y = pd.to_numeric(pf["target"], errors="coerce").fillna(0.0).to_numpy(np.float32)
    w = pd.to_numeric(pf["weight"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(np.float32)
    tids = pf["time_id"].to_numpy()
    asset = pf["asset_id"].to_numpy(np.float32)

    # 预构建共享组件
    print("building shared components...", flush=True)
    raw = build_raw(pf, feats)
    hist, sel_hist = build_history(pf, feats, top_k=48, windows=(5,))
    cs = build_cross_sectional(pf, sel_hist)  # 截面只对 top-48(与历史同集)，控维度
    hist_multi, sel_multi = build_history(pf, feats, top_k=48, windows=(5, 10, 20, 60))
    vol = build_volatility(pf, sel_hist, window=20)
    print(f"history feats: {hist.shape[1]} | cs: {cs.shape[1]} | hist_multi: {hist_multi.shape[1]} | vol: {vol.shape[1]}", flush=True)

    def assemble(*dfs):
        mats = [raw] + list(dfs)
        X = np.column_stack([asset] + [d.to_numpy(np.float32) for d in mats])
        return X

    experiments = [
        ("A_raw",            []),
        ("B_raw+hist5",      [hist]),
        ("C_raw+cs",         [cs]),
        ("D_raw+hist5+cs",   [hist, cs]),
        ("E_raw+hist_multi", [hist_multi]),
        ("F_raw+hist5+vol",  [hist, vol]),
        ("G_all",            [hist, cs, vol]),
        ("H_raw+hist_multi+cs", [hist_multi, cs]),
    ]

    ref = None
    print(f"\n{'exp':22s} {'nfeat':>6s} {'oof_R2':>10s} {'delta':>9s} {'min':>5s}", flush=True)
    results = []
    for name, dfs in experiments:
        t0 = time.time()
        try:
            X = assemble(*dfs)
            r2 = cv_oof(X, y, w, tids)
        except Exception as e:
            print(f"{name:22s} ERROR: {e}", flush=True)
            continue
        if ref is None:
            ref = r2
        d = r2 - ref
        print(f"{name:22s} {X.shape[1]:>6d} {r2:+.5f} {d:+.5f} {int(time.time()-t0):>4d}s", flush=True)
        results.append((name, X.shape[1], r2, d))
    print("\n=== summary ===", flush=True)
    for name, nf, r2, d in results:
        print(f"{name:22s} nfeat={nf:<5d} oof={r2:+.5f} delta={d:+.5f}", flush=True)


if __name__ == "__main__":
    main()
