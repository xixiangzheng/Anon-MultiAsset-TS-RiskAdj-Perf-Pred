"""holdout 上评估 2 模型 vs 3 模型集成（判断 XGBoost 是否值得加入）。

在 train 末尾 15% 时间(holdout)上，预测 tuned-LGBM / CatBoost / XGBoost(各 3 种子平均)，
算各自 R²、相关性、以及 2 模型/3 模型最优混合的 R²。若 3 模型明显优于 2 模型才值得提交。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import catboost as cb
import xgboost as xgb

STRAT = "/mnt/iscsi/hd/xxz/public_release_lightgbm_baseline/examples/lightgbm_baseline"
if STRAT not in sys.path:
    sys.path.insert(0, STRAT)
from data_utils import manifest_files, feature_columns_from_path  # noqa: E402

DATA_ROOT = Path("/mnt/iscsi/hd/xxz/public_release_20260630/data")


def wr2(y, p, w):
    d = float(np.sum(w * y * y)); return 0.0 if d <= 0 else 1 - float(np.sum(w * (y - p) ** 2) / d)


def best_blend(oofs, y, w, n_samples=8000):
    n = len(oofs); best_r, best_wt = -9, None
    rng = np.random.default_rng(0)
    for _ in range(n_samples):
        r = rng.random(n); wt = r / r.sum()
        if wr2(y, sum(wt[i] * oofs[i] for i in range(n)), w) > best_r:
            best_r = wr2(y, sum(wt[i] * oofs[i] for i in range(n)), w); best_wt = wt
    return best_r, best_wt


def main():
    paths = manifest_files(DATA_ROOT, "train")
    feats = feature_columns_from_path(paths[0])
    pf = pd.read_parquet(paths, columns=["time_id", "asset_id", "weight", "target"] + feats)
    pf[feats] = np.nan_to_num(pf[feats].to_numpy(np.float32))
    times = np.sort(pf["time_id"].unique())
    ho = set(times[-max(1, int(len(times) * 0.15)):].tolist())
    h = pf[pf["time_id"].isin(ho)].reset_index(drop=True)
    print(f"holdout {len(h):,} rows", flush=True)
    y = pd.to_numeric(h["target"], errors="coerce").fillna(0).to_numpy(np.float64)
    w = pd.to_numeric(h["weight"], errors="coerce").fillna(0).clip(lower=0).to_numpy(np.float64)
    asset = h["asset_id"].to_numpy(np.float32)
    raw = h[feats].astype(np.float32).to_numpy()

    # tuned LGBM
    md = Path("/mnt/iscsi/hd/xxz/src/lgbm_tuned/model"); rep = json.loads((md / "tuned_report.json").read_text())
    bi = rep["best_iteration"]
    lgb_models = [lgb.Booster(model_file=str(md / f"model_seed{s}.txt")) for s in rep["seeds"]]
    p_lgb = np.mean([m.predict(np.column_stack([asset, raw]), num_iteration=bi) for m in lgb_models], axis=0)

    # CatBoost
    md = Path("/mnt/iscsi/hd/xxz/src/cb_baseline/model"); rep = json.loads((md / "cb_report.json").read_text())
    df_cb = h[["asset_id"] + feats].copy(); df_cb["asset_id"] = df_cb["asset_id"].astype("int32")
    cb_models = [cb.CatBoost() for _ in rep["model_files"]]
    for m, f in zip(cb_models, rep["model_files"]): m.load_model(str(md / f))
    p_cb = np.mean([m.predict(df_cb) for m in cb_models], axis=0)

    # XGBoost
    md = Path("/mnt/iscsi/hd/xxz/src/xgb_baseline/model"); rep = json.loads((md / "xgb_report.json").read_text())
    feat_cols = ["asset_id"] + feats
    dx = xgb.DMatrix(h[feat_cols].astype(np.float32).to_numpy(), feature_names=feat_cols)
    xgb_models = [xgb.Booster() for _ in rep["model_files"]]
    for m, f in zip(xgb_models, rep["model_files"]): m.load_model(str(md / f))
    p_xgb = np.mean([m.predict(dx) for m in xgb_models], axis=0)

    rl, rc, rx = wr2(y, p_lgb, w), wr2(y, p_cb, w), wr2(y, p_xgb, w)
    print(f"\n单模型 holdout R²: LGBM={rl:+.5f}  CB={rc:+.5f}  XGB={rx:+.5f}", flush=True)
    print(f"corr: lgb-cb={np.corrcoef(p_lgb,p_cb)[0,1]:.3f} lgb-xgb={np.corrcoef(p_lgb,p_xgb)[0,1]:.3f} cb-xgb={np.corrcoef(p_cb,p_xgb)[0,1]:.3f}", flush=True)
    best_single = max(rl, rc, rx)
    r2, wt = best_blend([p_lgb, p_cb], y, w); print(f"2模型 LGBM+CB: {r2:+.5f} Δ={r2-best_single:+.5f} w={[round(x,2) for x in wt]}", flush=True)
    r3, wt = best_blend([p_lgb, p_cb, p_xgb], y, w); print(f"3模型 LGBM+CB+XGB: {r3:+.5f} Δ={r3-best_single:+.5f} w={[round(x,2) for x in wt]}", flush=True)
    print(f"\n3模型 vs 2模型 Δ = {r3-r2:+.5f}  (>+0.00005 才值得提交)", flush=True)


if __name__ == "__main__":
    main()
