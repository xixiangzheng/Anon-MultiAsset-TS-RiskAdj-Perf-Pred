from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np


class Model:
    def __init__(self):
        model_path = Path(__file__).resolve().parent / "model" / "linear_model.json"
        payload = json.loads(model_path.read_text(encoding="utf-8"))
        self.feature_columns = list(payload["feature_columns"])
        self.mean = np.asarray(payload["mean"], dtype=np.float64)
        self.scale = np.asarray(payload["scale"], dtype=np.float64)
        self.coef = np.asarray(payload["coef"], dtype=np.float64)
        self.window_size = int(payload["window_size"])
        self.history = defaultdict(lambda: deque(maxlen=self.window_size))
        self.last_time_id: int | None = None

    def predict(self, test):
        time_id = int(test["time_id"].iloc[0])
        if self.last_time_id is not None and time_id <= self.last_time_id:
            raise ValueError("time_id must increase in Time-Series API order")
        self.last_time_id = time_id

        current = test.loc[:, self.feature_columns].to_numpy(dtype=np.float64, copy=True)
        current = np.nan_to_num(current, nan=0.0, posinf=0.0, neginf=0.0)
        asset_ids = test["asset_id"].to_numpy(dtype=np.int64, copy=False)
        rows = []
        for idx, asset_id in enumerate(asset_ids):
            current_values = current[idx]
            history_values = list(self.history[asset_id])
            if history_values:
                rolling_mean = np.mean(np.vstack([*history_values, current_values]), axis=0)
            else:
                rolling_mean = current_values.copy()
            self.history[asset_id].append(current_values)
            rows.append(np.concatenate([current_values, rolling_mean]))

        x = np.vstack(rows)
        x = (x - self.mean) / self.scale
        x = np.column_stack([np.ones(len(x), dtype=np.float64), x])
        return x @ self.coef
