"""XGBoost 批量推理：一次性预测全部测试行，3 模型平均。"""
from __future__ import annotations

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

DATA_ROOT = Path("/mnt/iscsi/hd/xxz/public_release_20260630/data")
MODEL_DIR = Path("/mnt/iscsi/hd/xxz/src/xgb_baseline/model")
OUT = Path("/mnt/iscsi/hd/xxz/submissions/xgb_submission.csv")


def main():
    report = json.loads((MODEL_DIR / "xgb_report.json").read_text(encoding="utf-8"))
    feats = list(report["feature_cols"])
    pf = pd.read_parquet(manifest_files(DATA_ROOT, "test"), columns=["row_id", "asset_id"] + feats)
    pf[feats] = np.nan_to_num(pf[feats].to_numpy(np.float32))
    pf = pf.sort_values("row_id").reset_index(drop=True)
    print(f"loaded {len(pf):,} test rows", flush=True)
    feat_cols = ["asset_id"] + feats
    dtest = xgb.DMatrix(pf[feat_cols].astype(np.float32).to_numpy(), feature_names=feat_cols)
    model_files = report.get("model_files") or sorted(p.name for p in MODEL_DIR.glob("model_seed*.json"))
    preds = []
    for mf in model_files:
        m = xgb.Booster(); m.load_model(str(MODEL_DIR / mf))
        preds.append(m.predict(dtest))
        print(f"predicted {mf}", flush=True)
    avg = np.mean(preds, axis=0); avg = np.where(np.isfinite(avg), avg, 0.0)
    out = pd.DataFrame({"row_id": pf["row_id"], "target": avg})
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT} finite={np.isfinite(out['target']).all()}", flush=True)
    print(out["target"].describe(), flush=True)


if __name__ == "__main__":
    main()
