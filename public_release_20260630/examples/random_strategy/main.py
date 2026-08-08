from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class Model:
    def __init__(self):
        model_path = Path(__file__).resolve().parent / "model" / "random_model.json"
        payload = json.loads(model_path.read_text(encoding="utf-8"))
        self.rng = np.random.default_rng(int(payload["seed"]))
        self.prediction_scale = float(payload["prediction_scale"])
        self.last_time_id: int | None = None
        self.calls = 0

    def predict(self, test):
        time_id = int(test["time_id"].iloc[0])
        if self.last_time_id is not None and time_id <= self.last_time_id:
            raise ValueError("time_id must increase in Time-Series API order")
        self.last_time_id = time_id
        self.calls += 1
        return self.rng.normal(loc=0.0, scale=self.prediction_scale, size=len(test))
