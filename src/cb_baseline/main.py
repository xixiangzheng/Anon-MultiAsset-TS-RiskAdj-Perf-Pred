"""CatBoost 策略 Time-Series API 推理入口。"""
from __future__ import annotations

import json
import os
from pathlib import Path

import catboost as cb
import pandas as pd

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "model"


class Model:
    def __init__(self):
        model_dir = Path(os.environ.get("CB_BASELINE_MODEL_DIR", DEFAULT_MODEL_DIR))
        report = json.loads((model_dir / "cb_report.json").read_text(encoding="utf-8"))
        self.feature_cols = list(report["feature_cols"])
        model_files = list(report.get("model_files") or [])
        if not model_files:
            model_files = sorted(p.name for p in model_dir.glob("model_seed*.cbm"))
        self.models = [cb.CatBoost(); self.models[-1].load_model(str(model_dir / mf)) for mf in model_files]
        self.last_time_id = None

    def predict(self, test):
        time_id = int(test["time_id"].iloc[0])
        if self.last_time_id is not None and time_id <= self.last_time_id:
            raise ValueError("time_id must increase")
        self.last_time_id = time_id
        X = test[["asset_id"] + self.feature_cols].copy()
        X["asset_id"] = X["asset_id"].astype("int32")
        preds = [m.predict(X) for m in self.models]
        import numpy as np
        return np.mean(np.asarray(preds, dtype=np.float64), axis=0)
