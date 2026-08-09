"""决定性实验：predicted-responder 对 target 的预测力（stacking 天花板）。

对 target 相关性最高的若干 responder 做 purged 5-fold OOF 预测(从 features)，然后：
1. 每个 responder 的 OOF 预测 ŝ_k 单独对 target 的加权 R²
2. 交叉拟合的加权线性组合 Σ b_k ŝ_k → target（honest stacking 天花板）
回答："预测 responder 再组合，能把 target R² 推到多少？"
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
# 与 target 相关性最高的 10 个 responder（按 analyze_responder_corr 结果）
RESPONDERS = ["responder_03", "responder_02", "responder_28", "responder_29",
              "responder_18", "responder_19", "responder_17", "responder_11",
              "responder_30", "responder_04"]


def feval_wr2(preds, ds):
    y = ds.get_label()
    w = ds.get_weight()
    if w is None:
        w = np.ones_like(y)
    denom = float(np.sum(w * y * y))
    s = 0.0 if denom <= 0 else 1.0 - float(np.sum(w * (y - preds) ** 2) / denom)
    return ("wr2", float(s), True)


def oof_predict(X, y, w, train_idx, valid_idx, params):
    dtr = lgb.Dataset(X[train_idx], label=y[train_idx], weight=w[train_idx], categorical_feature=[0], free_raw_data=False)
    dva = lgb.Dataset(X[valid_idx], label=y[valid_idx], weight=w[valid_idx], reference=dtr, free_raw_data=False)
    model = lgb.train(params, dtr, num_boost_round=300, valid_sets=[dva], valid_names=["va"],
                      feval=feval_wr2, callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)])
    bi = model.best_iteration or 300
    return model.predict(X[valid_idx], num_iteration=bi)


def main() -> None:
    paths = manifest_files(DATA_ROOT, "train")
    feats = feature_columns_from_path(paths[0])
    cols = ["time_id", "asset_id", "weight", "target"] + RESPONDERS + feats
    pf = pd.read_parquet(paths[0], columns=cols)
    print(f"loaded {len(pf):,} rows, {len(feats)} features", flush=True)

    xf = np.nan_to_num(pf[feats].to_numpy(np.float32))
    Xfull = np.column_stack([pf["asset_id"].to_numpy(np.float32), xf])
    yt = pd.to_numeric(pf["target"], errors="coerce").fillna(0.0).to_numpy(np.float32)
    wt = pd.to_numeric(pf["weight"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(np.float32)
    n = len(pf)

    plan = make_validation_plan(pf["time_id"], n_splits=5, holdout_fraction=0.15, purge_steps=30)
    fold_valid_idx = [np.where(pf["time_id"].isin(set(map(int, f.valid_time_ids))).to_numpy())[0] for f in plan.folds]
    fold_train_idx = [np.where(pf["time_id"].isin(set(map(int, f.train_time_ids))).to_numpy())[0] for f in plan.folds]

    params = dict(objective="regression", metric="None", learning_rate=0.05, num_leaves=63,
                  min_data_in_leaf=2000, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  lambda_l2=10.0, verbosity=-1, num_threads=16, seed=2026,
                  bagging_seed=2026, feature_fraction_seed=2026, data_random_seed=2026)

    # 1) 每个 responder 的 OOF 预测
    oof = {}  # responder -> OOF pred aligned to full frame
    for resp in RESPONDERS:
        y = pd.to_numeric(pf[resp], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)
        w = pd.to_numeric(pf["weight"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(np.float32)
        pred = np.zeros(n, dtype=np.float64)
        for k in range(len(plan.folds)):
            pred[fold_valid_idx[k]] = oof_predict(Xfull, y, w, fold_train_idx[k], fold_valid_idx[k], params)
        oof[resp] = pred
        r2_self = weighted_zero_mean_r2(y, pred, w)  # 对自身的 R²(可预测性)
        r2_tgt = weighted_zero_mean_r2(yt, pred, wt)  # 对 target 的 R²(有用性)
        print(f"  {resp:14s}: 自身R²={r2_self:+.4f}  对target R²={r2_tgt:+.5f}", flush=True)

    # 2) 交叉拟合线性组合（fit folds {0,1,2}, eval {3,4}；再 fit {3,4}, eval {0,1,2}）
    def wlstsq_fit_eval(fit_blocks, eval_blocks):
        fi = np.concatenate([fold_valid_idx[b] for b in fit_blocks])
        ei = np.concatenate([fold_valid_idx[b] for b in eval_blocks])
        A_fit = np.column_stack([oof[r][fi] for r in RESPONDERS] + [np.ones(len(fi))])
        A_eval = np.column_stack([oof[r][ei] for r in RESPONDERS] + [np.ones(len(ei))])
        sw = np.sqrt(wt[fi])
        coef, *_ = np.linalg.lstsq(A_fit * sw[:, None], yt[fi] * sw, rcond=None)
        pred = A_eval @ coef
        return weighted_zero_mean_r2(yt[ei], pred, wt[ei])

    r2_a = wlstsq_fit_eval([0, 1, 2], [3, 4])
    r2_b = wlstsq_fit_eval([3, 4], [0, 1, 2])
    print(f"\n交叉拟合线性组合 target~Σŝ_k:  R²={0.5*(r2_a+r2_b):+.5f}  (两半 {r2_a:+.5f}/{r2_b:+.5f})", flush=True)
    print(f"对照: 直接预测 target 基线 ≈ +0.00200", flush=True)


if __name__ == "__main__":
    main()
