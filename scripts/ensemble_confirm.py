"""集成确认：3-seed × 三模型(LGBM/XGB/CatBoost)，1 分区 purged 5-fold。

回答：① 集成增益是否在多 seed 下稳健；② 加 CatBoost 是否进一步提升多样性。
输出每模型 3-seed 平均 oof R²、模型间相关矩阵、各集成组合的 R² + Δ。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

STRAT = "/mnt/iscsi/hd/xxz/public_release_lightgbm_baseline/examples/lightgbm_baseline"
if STRAT not in sys.path:
    sys.path.insert(0, STRAT)

from data_utils import manifest_files, feature_columns_from_path  # noqa: E402
from validation import make_validation_plan, weighted_zero_mean_r2  # noqa: E402

DATA_ROOT = Path("/mnt/iscsi/hd/xxz/public_release_20260630/data")
SEEDS = [2026, 2027, 2028]


def wr2(y, p, w):
    return weighted_zero_mean_r2(y, p, w)


def lgb_feval(p, ds):
    y = ds.get_label(); w = ds.get_weight()
    if w is None: w = np.ones_like(y)
    d = float(np.sum(w * y * y)); s = 0.0 if d <= 0 else 1 - float(np.sum(w * (y - p) ** 2) / d)
    return ("wr2", float(s), True)


def run_lgb(Xtr, ytr, wtr, Xva, yva, wva, feat_names, seed):
    p = dict(objective="regression", metric="None", learning_rate=0.05, num_leaves=63,
            min_data_in_leaf=2000, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
            lambda_l2=10.0, verbosity=-1, num_threads=16, seed=seed,
            bagging_seed=seed, feature_fraction_seed=seed, data_random_seed=seed)
    dtr = lgb.Dataset(Xtr, label=ytr, weight=wtr, feature_name=feat_names, categorical_feature=["asset_id"], free_raw_data=False)
    dva = lgb.Dataset(Xva, label=yva, weight=wva, reference=dtr, free_raw_data=False)
    m = lgb.train(p, dtr, num_boost_round=250, valid_sets=[dva], valid_names=["va"], feval=lgb_feval,
                  callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)])
    return m.predict(Xva, num_iteration=m.best_iteration or 250)


def run_xgb(Xtr, ytr, wtr, Xva, yva, wva, feat_names, seed):
    p = dict(objective="reg:squarederror", learning_rate=0.05, max_depth=8, min_child_weight=5,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=10.0, tree_method="hist", nthread=16, seed=seed, verbosity=0)
    dtr = xgb.DMatrix(Xtr, label=ytr, weight=wtr, feature_names=feat_names)
    dva = xgb.DMatrix(Xva, label=yva, weight=wva, feature_names=feat_names)
    m = xgb.train(p, dtr, num_boost_round=250, evals=[(dva, "va")], early_stopping_rounds=40, verbose_eval=False)
    return m.predict(dva, iteration_range=(0, m.best_iteration + 1))


def run_cb(Xtr, ytr, wtr, Xva, yva, wva, seed):
    # CatBoost: asset_id(列0)为类别
    p = dict(loss_function="RMSE", learning_rate=0.05, depth=8, l2_leaf_reg=10.0,
            iterations=250, random_seed=seed, thread_count=16, verbose=False,
            early_stopping_rounds=40, od_type="Iter", od_wait=40)
    trpool = cb.Pool(Xtr, label=ytr, weight=wtr, cat_features=[0])
    vapool = cb.Pool(Xva, label=yva, weight=wva, cat_features=[0])
    m = cb.train(trpool, p, eval_set=vapool, use_best_model=True, verbose=False)
    return m.predict(vapool)


def best_blend(oofs, y, w):
    """oofs: list of arrays。在加权R²下网格搜权重(限制非负、和为1，粗粒度)。"""
    n = len(oofs)
    best_r, best_w = -9, None
    # 随机/网格采样权重（三模型时用较多采样）
    rng = np.random.default_rng(0)
    for _ in range(4000 if n >= 3 else 21):
        if n == 2:
            a = rng.random(); wt = [a, 1 - a]
        else:
            r = rng.random(n); wt = r / r.sum()
        pred = sum(wt[i] * oofs[i] for i in range(n))
        r = wr2(y, pred, w)
        if r > best_r:
            best_r, best_w = r, wt
    return best_r, best_w


def main():
    paths = manifest_files(DATA_ROOT, "train")[:1]
    feats = feature_columns_from_path(paths[0])
    pf = pd.read_parquet(paths[0], columns=["time_id", "asset_id", "weight", "target"] + feats)
    pf[feats] = np.nan_to_num(pf[feats].to_numpy(np.float32))
    print(f"loaded {len(pf):,} rows", flush=True)
    y = pd.to_numeric(pf["target"], errors="coerce").fillna(0).to_numpy(np.float32)
    w = pd.to_numeric(pf["weight"], errors="coerce").fillna(0).clip(lower=0).to_numpy(np.float32)
    tids = pf["time_id"].to_numpy()
    asset = pf["asset_id"].to_numpy(np.int32)
    X = np.column_stack([asset.astype(np.float32), pf[feats].astype(np.float32).to_numpy()])
    feat_names = ["asset_id"] + feats
    n = len(pf)
    plan = make_validation_plan(pd.Series(tids), n_splits=5, holdout_fraction=0.15, purge_steps=30)

    # 每模型每 seed 的 OOF，最后跨 seed 平均
    oof = {m: {s: np.zeros(n) for s in SEEDS} for m in ["lgb", "xgb", "cb"]}
    mask = np.zeros(n, bool)
    runners = {"lgb": run_lgb, "xgb": run_xgb, "cb": run_cb}
    for k, f in enumerate(plan.folds):
        tr = np.isin(tids, list(map(int, f.train_time_ids)))
        va = np.isin(tids, list(map(int, f.valid_time_ids)))
        mask[va] = True
        for s in SEEDS:
            oof["lgb"][s][va] = runners["lgb"](X[tr], y[tr], w[tr], X[va], y[va], w[va], feat_names, s)
            oof["xgb"][s][va] = runners["xgb"](X[tr], y[tr], w[tr], X[va], y[va], w[va], feat_names, s)
            oof["cb"][s][va] = runners["cb"](X[tr], y[tr], w[tr], X[va], y[va], w[va], s)
        print(f"  fold {k} done (3 models × 3 seeds)", flush=True)

    # 跨 seed 平均
    oof_lgb = np.mean([oof["lgb"][s] for s in SEEDS], axis=0)
    oof_xgb = np.mean([oof["xgb"][s] for s in SEEDS], axis=0)
    oof_cb = np.mean([oof["cb"][s] for s in SEEDS], axis=0)
    ym, wm = y[mask], w[mask]
    L = oof_lgb[mask]; Xx = oof_xgb[mask]; C = oof_cb[mask]

    rl = wr2(ym, L, wm); rx = wr2(ym, Xx, wm); rc = wr2(ym, C, wm)
    best_single = max(rl, rx, rc)
    print(f"\nLGBM(3seed) R² = {rl:+.5f}", flush=True)
    print(f"XGB (3seed) R² = {rx:+.5f}", flush=True)
    print(f"CB  (3seed) R² = {rc:+.5f}", flush=True)
    print(f"corr(lgb,xgb)={np.corrcoef(L,Xx)[0,1]:.3f} corr(lgb,cb)={np.corrcoef(L,C)[0,1]:.3f} corr(xgb,cb)={np.corrcoef(Xx,C)[0,1]:.3f}", flush=True)
    for name, oofs in [("lgb+xgb", [L, Xx]), ("lgb+cb", [L, C]), ("xgb+cb", [Xx, C]), ("lgb+xgb+cb", [L, Xx, C])]:
        r, wt = best_blend(oofs, ym, wm)
        print(f"{name:12s} best-blend R²={r:+.5f}  Δ={r-best_single:+.5f}  w={[round(x,2) for x in wt]}", flush=True)


if __name__ == "__main__":
    main()
