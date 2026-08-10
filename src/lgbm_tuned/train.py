"""tuned LGBM 策略训练入口（Optuna 最优参数 + 纯 raw 特征，丢弃 history）。

依据：docs/探索日志.md 第七批——tuned 参数比 baseline 稳健 +0.0002~0.0004（迁移确认），
history 特征有害故弃用。purged 5-fold 定 best_iteration，3 种子全量重训。
"""
from __future__ import annotations

import argparse
import json
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

# Optuna 搜出的最优参数（1 分区 raw，2026-08-10）
TUNED = dict(
    objective="regression", metric="None",
    learning_rate=0.010326965981106629, num_leaves=79, min_data_in_leaf=556,
    feature_fraction=0.5974233229491067, bagging_fraction=0.9445362670741704, bagging_freq=1,
    lambda_l1=0.03778653953330111, lambda_l2=2.9757802078489703, max_bin=127,
    verbosity=-1,
)
SEEDS = (2026, 2027, 2028)


def feval_wr2(preds, ds):
    y = ds.get_label(); w = ds.get_weight()
    if w is None:
        w = np.ones_like(y)
    d = float(np.sum(w * y * y))
    s = 0.0 if d <= 0 else 1.0 - float(np.sum(w * (y - preds) ** 2) / d)
    return ("wr2", float(s), True)


def params_with(seed, num_threads):
    p = dict(TUNED)
    p.update(num_threads=num_threads, seed=seed, bagging_seed=seed,
             feature_fraction_seed=seed, data_random_seed=seed)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--num-threads", type=int, default=64)
    ap.add_argument("--num-boost-round", type=int, default=1000)
    ap.add_argument("--early-stopping-rounds", type=int, default=60)
    ap.add_argument("--max-train-rows", type=int, default=0, help="0=全量(开发期可设小)")
    args = ap.parse_args()

    paths = manifest_files(args.release_root, "train")
    feats = feature_columns_from_path(paths[0])
    cols = ["row_id", "time_id", "asset_id", "weight", "target"] + feats
    pf = pd.read_parquet(paths, columns=cols)
    pf[feats] = np.nan_to_num(pf[feats].to_numpy(np.float32))
    print(f"loaded {len(pf):,} rows, {len(feats)} features", flush=True)

    if args.max_train_rows > 0:  # 开发期：按 time 采样
        times = pf["time_id"].drop_duplicates().to_numpy()
        n_t = max(1, int(args.max_train_rows / max(len(pf) / max(len(times), 1), 1.0)))
        rng = np.random.default_rng(2026)
        chosen = np.sort(rng.choice(times, size=min(n_t, len(times)), replace=False))
        pf = pf.loc[pf["time_id"].isin(chosen)].reset_index(drop=True)
        print(f"subsampled to {len(pf):,} rows", flush=True)

    y = pd.to_numeric(pf["target"], errors="coerce").fillna(0.0).to_numpy(np.float32)
    w = pd.to_numeric(pf["weight"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(np.float32)
    tids = pf["time_id"].to_numpy()
    asset = pf["asset_id"].to_numpy(np.float32)
    X = np.column_stack([asset, pf[feats].astype(np.float32).to_numpy()])  # 列0=asset_id(类别)
    n = len(pf)

    # purged 5-fold 定 best_iteration
    plan = make_validation_plan(pd.Series(tids), n_splits=5, holdout_fraction=0.15, purge_steps=30)
    oof = np.zeros(n); mask = np.zeros(n, bool)
    fold_iters = []
    for k, f in enumerate(plan.folds):
        tr = np.isin(tids, list(map(int, f.train_time_ids))); va = np.isin(tids, list(map(int, f.valid_time_ids)))
        dtr = lgb.Dataset(X[tr], label=y[tr], weight=w[tr], categorical_feature=[0], free_raw_data=False)
        dva = lgb.Dataset(X[va], label=y[va], weight=w[va], reference=dtr, free_raw_data=False)
        m = lgb.train(params_with(SEEDS[0], args.num_threads), dtr, num_boost_round=args.num_boost_round,
                      valid_sets=[dva], valid_names=["va"], feval=feval_wr2,
                      callbacks=[lgb.early_stopping(args.early_stopping_rounds, verbose=False), lgb.log_evaluation(0)])
        bi = m.best_iteration or args.num_boost_round
        fold_iters.append(bi)
        oof[va] = m.predict(X[va], num_iteration=bi); mask[va] = True
        print(f"[cv] fold {k} best_iter={bi}", flush=True)
    best_iter = max(1, int(round(float(np.mean(fold_iters)))))
    cv_oof = weighted_zero_mean_r2(y[mask], oof[mask], w[mask])
    print(f"[cv] mean best_iter={best_iter}  oof_wr2={cv_oof:+.6f}", flush=True)

    # 3 种子全量重训
    model_dir = Path(args.model_dir); model_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for sd in SEEDS:
        dtr = lgb.Dataset(X, label=y, weight=w, categorical_feature=[0], free_raw_data=False)
        m = lgb.train(params_with(sd, args.num_threads), dtr, num_boost_round=best_iter, feval=feval_wr2,
                      valid_sets=[dtr], valid_names=["train"], callbacks=[lgb.log_evaluation(0)])
        fname = f"model_seed{sd}.txt"
        m.save_model(str(model_dir / fname))
        files.append(fname)
        print(f"[final] seed={sd} saved ({best_iter} rounds)", flush=True)

    report = {
        "strategy": "lgbm_tuned", "feature_cols": feats, "params": TUNED,
        "best_iteration": best_iter, "fold_iterations": fold_iters, "cv_oof_wr2": cv_oof,
        "seeds": list(SEEDS), "model_files": files, "uses_history": False,
    }
    (model_dir / "tuned_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"best_iteration": best_iter, "cv_oof_wr2": cv_oof, "model_dir": str(model_dir)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
