"""CatBoost 策略训练（全量，为 LGBM+CB 集成提供多样性候选）。

轻正则(depth8, l2=3)，1 次 holdout 定 iterations，3 种子全量重训。
目的：与 LGBM 形成去相关集成（corr~0.84），预期集成 +0.0001。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import catboost as cb

STRAT = "/mnt/iscsi/hd/xxz/public_release_lightgbm_baseline/examples/lightgbm_baseline"
if STRAT not in sys.path:
    sys.path.insert(0, STRAT)

from data_utils import manifest_files, feature_columns_from_path  # noqa: E402

CB_PARAMS = dict(loss_function="RMSE", learning_rate=0.05, depth=8, l2_leaf_reg=3.0,
                 random_seed=2026, thread_count=64, verbose=False, early_stopping_rounds=50,
                 use_best_model=True)
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

    # 按时间 holdout(末 15%) 定 iterations
    times = np.sort(pf["time_id"].unique())
    ho_n = max(1, int(len(times) * 0.15))
    ho_times = set(times[-ho_n:].tolist())
    is_va = pf["time_id"].isin(ho_times).to_numpy()
    train_df = pf[~is_va].copy(); valid_df = pf[is_va].copy()
    print(f"holdout: train {len(train_df):,} / valid {len(valid_df):,}", flush=True)

    def make_pool(df):
        X = df[["asset_id"] + feats].copy(); X["asset_id"] = X["asset_id"].astype(np.int32)
        y = pd.to_numeric(df["target"], errors="coerce").fillna(0).to_numpy(np.float32)
        w = pd.to_numeric(df["weight"], errors="coerce").fillna(0).clip(lower=0).to_numpy(np.float32)
        return cb.Pool(X, label=y, weight=w, cat_features=["asset_id"])

    trpool = make_pool(train_df); vapool = make_pool(valid_df)
    yva = vapool.get_label(); wva = vapool.get_weight()
    p = dict(CB_PARAMS); p["iterations"] = MAX_ITERS
    m = cb.train(trpool, p, eval_set=vapool, verbose=False)
    bi = m.tree_count_
    pred_va = m.predict(vapool)
    print(f"[holdout] tree_count={bi}  holdout_wr2={wr2(yva, pred_va, wva):+.6f}", flush=True)

    # 全量(含 holdout) 3 种子重训
    full_df = pd.concat([train_df, valid_df], ignore_index=True)
    fullpool = make_pool(full_df)
    model_dir = Path(args.model_dir); model_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for sd in SEEDS:
        p = dict(CB_PARAMS); p["random_seed"] = int(sd); p["iterations"] = int(bi); p["use_best_model"] = False
        mm = cb.train(fullpool, p, verbose=False)
        fname = f"model_seed{sd}.cbm"
        mm.save_model(str(model_dir / fname)); files.append(fname)
        print(f"[final] seed={sd} saved ({bi} trees)", flush=True)

    report = {"strategy": "cb_baseline", "feature_cols": feats, "iterations": int(bi),
              "holdout_wr2": float(wr2(yva, pred_va, wva)), "seeds": list(SEEDS), "model_files": files}
    (model_dir / "cb_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"iterations": int(bi), "holdout_wr2": report["holdout_wr2"], "model_dir": str(model_dir)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
