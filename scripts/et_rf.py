"""ExtraTrees 回归作为集成多样性来源（与 GBDT/NN 完全不同的随机分割策略）。

用部分数据训练（前 3 分区 ~4M 行，省时；ExtraTrees 在 4M 已足够稳定）。
holdout 评估用 train 末 15%。
"""
from __future__ import annotations
import sys, time, pickle
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor

STRAT = "/mnt/iscsi/hd/xxz/public_release_lightgbm_baseline/examples/lightgbm_baseline"
if STRAT not in sys.path: sys.path.insert(0, STRAT)
from data_utils import manifest_files, feature_columns_from_path
DATA_ROOT = Path("/mnt/iscsi/hd/xxz/public_release_20260630/data")
RUN = Path("/mnt/iscsi/hd/xxz/runs"); SUB = Path("/mnt/iscsi/hd/xxz/submissions")


def wr2(y, p, w):
    d = float(np.sum(w*y*y)); return 0.0 if d<=0 else 1-float(np.sum(w*(y-p)**2)/d)


def main():
    paths = manifest_files(DATA_ROOT, "train"); feats = feature_columns_from_path(paths[0])
    # 前 4 分区训练（~5.8M 行）
    pf = pd.read_parquet(paths[:4], columns=["time_id","asset_id","weight","target"]+feats)
    pf[feats] = np.nan_to_num(pf[feats].to_numpy(np.float32))
    print(f"loaded {len(pf):,} rows from parts 0-3", flush=True)

    times = np.sort(pf["time_id"].unique()); ho = set(times[-max(1,int(len(times)*0.15)):].tolist())
    is_va = pf["time_id"].isin(ho).to_numpy()
    a = pf["asset_id"].to_numpy(np.float32)
    F = pf[feats].to_numpy(np.float32)
    X = np.column_stack([a, F])
    y = pd.to_numeric(pf["target"],errors="coerce").fillna(0).to_numpy(np.float32)
    w = pd.to_numeric(pf["weight"],errors="coerce").fillna(0).clip(lower=0).to_numpy(np.float32)
    Xtr, Xva = X[~is_va], X[is_va]
    ytr, yva = y[~is_va], y[is_va]
    wtr, wva = w[~is_va], w[is_va]
    print(f"train {len(Xtr):,} holdout {len(Xva):,}", flush=True)

    te = pd.read_parquet(manifest_files(DATA_ROOT,"test"),columns=["row_id","asset_id"]+feats)
    te[feats] = np.nan_to_num(te[feats].to_numpy(np.float32)); te = te.sort_values("row_id").reset_index(drop=True)
    X_te = np.column_stack([te["asset_id"].to_numpy(np.float32), te[feats].to_numpy(np.float32)])

    models = {
        "et": ExtraTreesRegressor(n_estimators=80, max_depth=12, min_samples_leaf=200,
                                  n_jobs=32, random_state=2026, verbose=1),
        "rf": RandomForestRegressor(n_estimators=80, max_depth=12, min_samples_leaf=200,
                                    n_jobs=32, random_state=2026, verbose=1),
    }
    oofs = {}; te_preds = {}
    for name, m in models.items():
        print(f"\n=== {name} ===", flush=True); t0 = time.time()
        m.fit(Xtr, ytr, sample_weight=wtr)
        oofs[name] = m.predict(Xva).astype(np.float32)
        te_preds[name] = m.predict(X_te).astype(np.float32)
        print(f"  holdout R²={wr2(yva, oofs[name], wva):+.5f} ({time.time()-t0:.0f}s)", flush=True)

    # 保存
    for k,p in te_preds.items():
        p = np.where(np.isfinite(p), p, 0.0)
        pd.DataFrame({"row_id":te["row_id"],"target":p}).to_csv(SUB/f"{k}_submission.csv", index=False)
    pickle.dump({"oofs":oofs,"te_preds":te_preds,"yv":yva,"wv":wva,
                 "row_id":te["row_id"].to_numpy()},
                open(RUN/"et_rf_oof.pkl","wb"))
    print("\n[done] saved.", flush=True)


if __name__ == "__main__":
    main()
