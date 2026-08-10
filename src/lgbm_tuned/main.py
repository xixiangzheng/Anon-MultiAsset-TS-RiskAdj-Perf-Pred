"""tuned LGBM 策略 Time-Series API 推理入口（纯 raw 特征，无 history）。"""
from __future__ import annotations

import json
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "model"


class Model:
    def __init__(self):
        model_dir = Path(os.environ.get("LGBM_TUNED_MODEL_DIR", DEFAULT_MODEL_DIR))
        report = json.loads((model_dir / "tuned_report.json").read_text(encoding="utf-8"))
        self.feature_cols = list(report["feature_cols"])
        self.best_iteration = int(report["best_iteration"])
        model_files = list(report.get("model_files") or [])
        if not model_files:
            model_files = sorted(p.name for p in model_dir.glob("model_seed*.txt"))
        self.boosters = []
        for mf in model_files:
            self.boosters.append(lgb.Booster(model_file=str(model_dir / mf)))
        self.last_time_id = None

    def predict(self, test):
        time_id = int(test["time_id"].iloc[0])
        if self.last_time_id is not None and time_id <= self.last_time_id:
            raise ValueError("time_id must increase")
        self.last_time_id = time_id
        asset = test["asset_id"].to_numpy(np.float32)
        raw = np.nan_to_num(test[self.feature_cols].to_numpy(np.float32))
        x = np.column_stack([asset, raw])  # 与训练一致：列0=asset_id
        preds = [b.predict(x, num_iteration=self.best_iteration) for b in self.boosters]
        return np.mean(np.asarray(preds, dtype=np.float64), axis=0)
