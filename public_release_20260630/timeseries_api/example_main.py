from __future__ import annotations

import numpy as np


class Model:
    def __init__(self):
        self.seen_time_ids: list[int] = []

    def predict(self, test):
        self.seen_time_ids.append(int(test["time_id"].iloc[0]))
        return np.zeros(len(test), dtype=np.float32)
