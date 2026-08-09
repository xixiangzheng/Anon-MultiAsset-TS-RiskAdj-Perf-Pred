"""探测 target 与 top responder 能否由 features 预测（purged 5-fold OOF 加权 R²）。

stacking 的价值取决于 responder 是否可由 features 预测。本脚本对 target 和相关性最高的
6 个 responder 各跑一遍 LGBM purged CV，报告 OOF 加权 R²。仅用原始特征(无历史衍生)，快速探测。
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

from data_utils import feature_columns_from_path, manifest_files  # noqa: E402
from validation import make_validation_plan, weighted_zero_mean_r2  # noqa: E402

DATA_ROOT = Path("/mnt/iscsi/hd/xxz/public_release_20260630/data")
# target + 相关性最高的 6 个 responder
RESPONSES = ["target", "responder_03", "responder_28", "responder_02", "responder_29", "responder_18", "responder_19"]


def feval_wr2(preds, ds):
    y = ds.get_label()
    w = ds.get_weight()
    if w is None:
        w = np.ones_like(y)
    denom = float(np.sum(w * y * y))
    s = 0.0 if denom <= 0 else 1.0 - float(np.sum(w * (y - preds) ** 2) / denom)
    return ("wr2", float(s), True)


def main() -> None:
    paths = manifest_files(DATA_ROOT, "train")
    feats = feature_columns_from_path(paths[0])
    cols = ["time_id", "asset_id", "weight"] + RESPONSES + feats
    pf = pd.read_parquet(paths[0], columns=cols)
    print(f"loaded {len(pf):,} rows, {len(feats)} features", flush=True)

    # 清洗特征
    xf = np.nan_to_num(pf[feats].to_numpy(np.float32))
    asset = pf["asset_id"].to_numpy(np.float32)
    Xfull = np.column_stack([asset, xf])  # 列0 = asset_id(类别)
    n = len(pf)

    plan = make_validation_plan(pf["time_id"], n_splits=5, holdout_fraction=0.15, purge_steps=30)
    tr_masks = [pf["time_id"].isin(set(map(int, f.train_time_ids))).to_numpy() for f in plan.folds]
    va_masks = [pf["time_id"].isin(set(map(int, f.valid_time_ids))).to_numpy() for f in plan.folds]

    params = dict(
        objective="regression", metric="None", learning_rate=0.05, num_leaves=63,
        min_data_in_leaf=2000, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
        lambda_l2=10.0, verbosity=-1, num_threads=16, seed=2026,
        bagging_seed=2026, feature_fraction_seed=2026, data_random_seed=2026,
    )

    print("\n=== OOF weighted R² (feature → response) ===", flush=True)
    for resp in RESPONSES:
        y = pd.to_numeric(pf[resp], errors="coerce").fillna(0.0).to_numpy(np.float32)
        w = pd.to_numeric(pf["weight"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(np.float32)
        oof = np.zeros(n, dtype=np.float64)
        mask = np.zeros(n, dtype=bool)
        for k in range(len(plan.folds)):
            tr, va = tr_masks[k], va_masks[k]
            dtr = lgb.Dataset(Xfull[tr], label=y[tr], weight=w[tr], categorical_feature=[0], free_raw_data=False)
            dva = lgb.Dataset(Xfull[va], label=y[va], weight=w[va], reference=dtr, free_raw_data=False)
            model = lgb.train(
                params, dtr, num_boost_round=250,
                valid_sets=[dva], valid_names=["va"],
                feval=feval_wr2,
                callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
            )
            bi = model.best_iteration or 250
            oof[va] = model.predict(Xfull[va], num_iteration=bi)
            mask[va] = True
        r2 = weighted_zero_mean_r2(y[mask], oof[mask], w[mask])
        flag = "  ← target" if resp == "target" else ""
        print(f"  {resp:14s}: {r2:+.5f}{flag}", flush=True)


if __name__ == "__main__":
    main()
