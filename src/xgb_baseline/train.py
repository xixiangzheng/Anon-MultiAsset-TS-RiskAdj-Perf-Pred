"""XGBoost GPU 训练（全量，为 3 模型集成提供多样性）。

GPU (device=cuda)，1 次 holdout 定 iterations，3 种子全量重训。asset_id 作数值处理（0-14 小整数）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

STRAT = "/mnt/iscsi/hd/xxz/public_release_lightgbm_baseline/examples/lightgbm_baseline"
if STRAT not in sys.path:
    sys.path.insert(0, STRAT)
from data_utils import manifest_files, feature_columns_from_path  # noqa: E402

XGB_PARAMS = dict(tree_method="hist", device="cuda", objective="reg:squarederror",
                  learning_rate=0.05, max_depth=8, min_child_weight=5,
                  subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0, verbosity=0)
SEEDS = (2026, 2027, 2028)
MAX_ITERS = 800


def wr2(y, p, w):
    d = float(np.sum(w * y * y)); return 0.0 if d <= 0 else 1 - float(np.sum(w * (y - p) ** 2) / d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--max-train-rows", type=int, default=0)
    args = ap.parse_args()

    paths = manifest_files(args.release_root, "train")
    feats = feature_columns_from_path(paths[0])
    cols = ["time_id", "weight", "target", "asset_id"] + feats
    pf = pd.read_parquet(paths, columns=cols)
    pf[feats] = np.nan_to_num(pf[feats].to_numpy(np.float32))
    print(f"loaded {len(pf):,} rows, {len(feats)} features", flush=True)

    if args.max_train_rows > 0:
        times = pf["time_id"].drop_duplicates().to_numpy()
        n_t = max(1, int(args.max_train_rows / max(len(pf) / max(len(times), 1), 1.0)))
        rng = np.random.default_rng(2026)
        chosen = np.sort(rng.choice(times, size=min(n_t, len(times)), replace=False))
        pf = pf.loc[pf["time_id"].isin(chosen)].reset_index(drop=True)
        print(f"subsampled to {len(pf):,} rows", flush=True)

    times = np.sort(pf["time_id"].unique())
    ho_n = max(1, int(len(times) * 0.15))
    ho_times = set(times[-ho_n:].tolist())
    is_va = pf["time_id"].isin(ho_times).to_numpy()
    train_df, valid_df = pf[~is_va].copy(), pf[is_va].copy()
    print(f"holdout: train {len(train_df):,} / valid {len(valid_df):,}", flush=True)

    feat_cols = ["asset_id"] + feats

    def dmat(df):
        X = df[feat_cols].astype(np.float32).to_numpy()
        y = pd.to_numeric(df["target"], errors="coerce").fillna(0).to_numpy(np.float32)
        w = pd.to_numeric(df["weight"], errors="coerce").fillna(0).clip(lower=0).to_numpy(np.float32)
        return xgb.DMatrix(X, label=y, weight=w, feature_names=feat_cols)

    dtr, dva = dmat(train_df), dmat(valid_df)
    p = dict(XGB_PARAMS); p["seed"] = 2026
    m = xgb.train(p, dtr, num_boost_round=MAX_ITERS, evals=[(dva, "va")],
                  early_stopping_rounds=50, verbose_eval=False)
    bi = m.best_iteration + 1
    print(f"[holdout] best_iter={bi}  holdout_wr2={wr2(dva.get_label(), m.predict(dva), dva.get_weight()):+.6f}", flush=True)

    model_dir = Path(args.model_dir); model_dir.mkdir(parents=True, exist_ok=True)
    dfull = dmat(pd.concat([train_df, valid_df], ignore_index=True))
    files = []
    for sd in SEEDS:
        p = dict(XGB_PARAMS); p["seed"] = int(sd)
        mm = xgb.train(p, dfull, num_boost_round=bi, verbose_eval=False)
        fname = f"model_seed{sd}.json"
        mm.save_model(str(model_dir / fname)); files.append(fname)
        print(f"[final] seed={sd} saved ({bi} rounds)", flush=True)

    report = {"strategy": "xgb_baseline", "feature_cols": feats, "iterations": int(bi),
              "seeds": list(SEEDS), "model_files": files}
    (model_dir / "xgb_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"iterations": int(bi), "model_dir": str(model_dir)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
