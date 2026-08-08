from __future__ import annotations

import json
import os
import re
from collections import defaultdict, deque
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "model"


class Model:
    """LightGBM baseline 的 Time-Series API 推理入口。

    默认从策略目录下 ``model/`` 加载报告与多种子模型；可用环境变量
    ``LIGHTGBM_BASELINE_MODEL_DIR`` 覆盖。``fitted_oof_scale`` 仅作诊断，
    不会乘到提交预测上。
    """

    def __init__(self):
        model_dir = Path(os.environ.get("LIGHTGBM_BASELINE_MODEL_DIR", DEFAULT_MODEL_DIR))
        report_path = model_dir / "lightgbm_report.json"
        if not report_path.exists():
            raise FileNotFoundError(f"未找到 baseline 报告文件: {report_path}")

        report = json.loads(report_path.read_text(encoding="utf-8"))
        features = report.get("features", {})
        self.raw_features = list(features["selected_raw_features"])
        self.history_features = list(features["history_features"])
        self.rolling_windows = tuple(int(window) for window in features.get("rolling_windows", [5]))
        if not self.rolling_windows:
            raise ValueError("rolling_windows 不能为空")
        self.max_history = max(self.rolling_windows) - 1
        # scale 仅诊断，推理恒为 1.0
        self.prediction_scale = 1.0
        self.fitted_oof_scale = float(report.get("fitted_oof_scale", 1.0))

        model_files = list(report.get("model_files") or ["model.txt"])
        best_iterations = list(report.get("best_iterations") or [report.get("best_iteration")])
        if len(best_iterations) == 1 and len(model_files) > 1:
            best_iterations = best_iterations * len(model_files)
        if len(best_iterations) != len(model_files):
            raise ValueError("model_files and best_iterations length mismatch")

        self.boosters: list[lgb.Booster] = []
        self.best_iterations: list[int] = []
        for model_file, best_iteration in zip(model_files, best_iterations):
            model_path = model_dir / model_file
            if not model_path.exists():
                raise FileNotFoundError(f"未找到 LightGBM 模型文件: {model_path}")
            booster = lgb.Booster(model_file=str(model_path))
            self.boosters.append(booster)
            self.best_iterations.append(int(best_iteration or booster.current_iteration()))

        self.model_columns = list(self.boosters[0].feature_name())
        for booster in self.boosters[1:]:
            if list(booster.feature_name()) != self.model_columns:
                raise ValueError("bagging 模型的特征名/顺序必须一致")

        self.raw_feature_set = set(self.raw_features)
        self.raw_feature_index = {feature: idx for idx, feature in enumerate(self.raw_features)}
        self.history_index = {feature: idx for idx, feature in enumerate(self.history_features)}
        self.history_raw_positions = [self.raw_feature_index[feature] for feature in self.history_features]
        self.model_column_plan = [self._column_plan(column) for column in self.model_columns]
        self.history = defaultdict(lambda: deque(maxlen=self.max_history if self.max_history > 0 else 1))
        self.last_time_id: int | None = None

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        time_id = int(test["time_id"].iloc[0])
        if self.last_time_id is not None and time_id <= self.last_time_id:
            raise ValueError("Time-Series API 要求 time_id 严格递增")
        self.last_time_id = time_id

        missing = [column for column in self.raw_features if column not in test.columns]
        if missing:
            raise ValueError(f"test 缺少特征列: {missing[:5]}")

        raw = test.loc[:, self.raw_features].to_numpy(dtype=np.float32, copy=True)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        lag, diff, rolling_by_window = self._history_arrays(test, raw)

        x = np.empty((len(test), len(self.model_columns)), dtype=np.float32)
        asset_id_values = test["asset_id"].to_numpy(dtype=np.float32, copy=False)
        for output_idx, plan in enumerate(self.model_column_plan):
            kind, name, window = plan
            if kind == "asset_id":
                x[:, output_idx] = asset_id_values
            elif kind == "raw":
                x[:, output_idx] = raw[:, self.raw_feature_index[name]]
            elif kind == "lag1":
                x[:, output_idx] = lag[:, self.history_index[name]]
            elif kind == "diff1":
                x[:, output_idx] = diff[:, self.history_index[name]]
            elif kind == "rmean":
                x[:, output_idx] = rolling_by_window[window][:, self.history_index[name]]
            else:
                x[:, output_idx] = 0.0

        preds = [
            booster.predict(x, num_iteration=best_iteration)
            for booster, best_iteration in zip(self.boosters, self.best_iterations)
        ]
        return np.mean(np.asarray(preds, dtype=np.float64), axis=0) * self.prediction_scale

    def _column_plan(self, column: str) -> tuple[str, str, int | None]:
        if column == "asset_id":
            return ("asset_id", column, None)
        if column in self.raw_feature_set:
            return ("raw", column, None)
        if column.startswith("lag1_"):
            return ("lag1", column[len("lag1_") :], None)
        if column.startswith("diff1_"):
            return ("diff1", column[len("diff1_") :], None)
        match = re.fullmatch(r"rmean(\d+)_(.+)", column)
        if match:
            return ("rmean", match.group(2), int(match.group(1)))
        return ("missing", column, None)

    def _history_arrays(
        self,
        test: pd.DataFrame,
        raw: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
        if not self.history_features:
            empty = np.zeros((len(test), 0), dtype=np.float32)
            return empty, empty, {window: empty for window in self.rolling_windows}

        values = raw[:, self.history_raw_positions]
        asset_ids = test["asset_id"].to_numpy(dtype=np.int64, copy=False)
        lag = np.zeros_like(values, dtype=np.float32)
        rolling_by_window = {window: np.zeros_like(values, dtype=np.float32) for window in self.rolling_windows}

        for row_idx, asset_id in enumerate(asset_ids):
            current = values[row_idx]
            previous = list(self.history[asset_id])
            if previous:
                lag[row_idx] = previous[-1]
            for window in self.rolling_windows:
                window_values = previous[-(window - 1) :] + [current] if window > 1 else [current]
                rolling_by_window[window][row_idx] = np.mean(np.vstack(window_values), axis=0, dtype=np.float64)
            self.history[asset_id].append(current.copy())

        return lag, values - lag, rolling_by_window
