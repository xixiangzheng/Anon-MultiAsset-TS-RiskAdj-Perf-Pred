"""私榜交付策略：ens_ratio_nn30 配方（公榜 0.003345 验证）。

配方：0.35 * ratio_lgb(3seed) + 0.35 * catboost(3seed) + 0.30 * nn_emb32(10seed)
- ratio_lgb: tuned LGBM + simple ratio（全部特征 ÷ feature_157，train 分位数 clip）
- catboost: raw 323 特征 + asset_id 数值列
- nn_emb32: MLP(emb32, 512/256/128) 标准化 raw 特征

合规要点：
- 所有路径相对 Path(__file__)，无绝对路径
- 推理线程数 <= 4
- 纯 per-row 模型，无历史状态依赖（负 row_id 行自然兼容）
- predict(test) 返回 len(test) 一维有限浮点数组
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

STRATEGY_DIR = Path(__file__).resolve().parent
MODEL_DIR = STRATEGY_DIR / "model"
CONFIG_PATH = MODEL_DIR / "inference_config.json"

NUM_THREADS = 4  # 评测环境 4 核

# 集成权重（从 inference_config.json 读取，构建时由 SLSQP 优化得到）
W_LGB = 0.36
W_CB = 0.25
W_NN = 0.39


def _build_mlp(n_feat, emb_dim, hidden):
    """MLP(emb32, 512/256/128, GELU, BN, Dropout0.3) — 与训练脚本一致"""
    import torch
    import torch.nn as nn
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(15, emb_dim)
            layers = []
            d = n_feat + emb_dim
            for h in hidden:
                layers += [nn.Linear(d, h), nn.GELU(), nn.BatchNorm1d(h), nn.Dropout(0.3)]
                d = h
            layers += [nn.Linear(d, 1)]
            self.net = nn.Sequential(*layers)

        def forward(self, x, a):
            return self.net(torch.cat([x, self.emb(a)], 1)).squeeze(-1)
    return MLP()


class Model:
    def __init__(self):
        import lightgbm as lgb
        import catboost as cb
        import torch

        self._torch = torch
        cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        self.feature_cols = list(cfg["feature_cols"])
        self.ratio_denom_idx = int(cfg["ratio_denom_idx"])
        self.ratio_hi_pct = float(cfg["ratio_hi_pct"])
        self.ratio_lo = np.asarray(cfg["ratio_lo"], dtype=np.float32)
        self.ratio_hi = np.asarray(cfg["ratio_hi"], dtype=np.float32)
        self.nn_mean = np.asarray(cfg["nn_mean"], dtype=np.float32)
        self.nn_std = np.asarray(cfg["nn_std"], dtype=np.float32)
        self.raw_lo = np.asarray(cfg["raw_lo"], dtype=np.float32)
        self.raw_hi = np.asarray(cfg["raw_hi"], dtype=np.float32)
        self.nn_seeds = list(cfg["nn_seeds"])
        self.nn_hidden = tuple(cfg["nn_hidden"])
        self.nn_emb_dim = int(cfg["nn_emb_dim"])
        self.lgb_best_iter = int(cfg["lgb_best_iter"])
        wts = cfg.get("weights", {})
        self.w_lgb = float(wts.get("lgb", W_LGB))
        self.w_cb = float(wts.get("cb", W_CB))
        self.w_nn = float(wts.get("nn", W_NN))

        torch.set_num_threads(NUM_THREADS)

        # 1. LightGBM（3 seed）
        self.boosters = [lgb.Booster(model_file=str(mf))
                         for mf in sorted(MODEL_DIR.glob("ratio_lgb_seed*.txt"))]

        # 2. CatBoost（3 seed；predict 时显式传 thread_count）
        self.cb_models = [cb.CatBoost() for _ in sorted(MODEL_DIR.glob("cb_seed*.cbm"))]
        for m, mf in zip(self.cb_models, sorted(MODEL_DIR.glob("cb_seed*.cbm"))):
            m.load_model(str(mf))

        # 3. NN（10 seed）
        n_feat = len(self.feature_cols)
        self.nns = []
        for sd in self.nn_seeds:
            mlp = _build_mlp(n_feat, self.nn_emb_dim, self.nn_hidden)
            mlp.load_state_dict(torch.load(MODEL_DIR / f"nn_emb32_seed{sd}.pt",
                                           map_location="cpu", weights_only=True))
            mlp.eval()
            self.nns.append(mlp)

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        raw = np.nan_to_num(test[self.feature_cols].to_numpy(np.float32))
        # 防极端值：clip 到 train 分位数（test 存在极端 outlier，曾致 NN 外推爆炸）
        raw = np.clip(raw, self.raw_lo, self.raw_hi)

        # --- LGBM: asset + raw + ratio ---
        fd = np.clip(raw[:, self.ratio_denom_idx], 1e-8, self.ratio_hi_pct)
        r = raw / fd[:, None]
        r = np.clip(np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0), self.ratio_lo, self.ratio_hi)
        x_lgb = np.column_stack([test["asset_id"].to_numpy(np.float32), raw, r])
        lgb_pred = np.mean([b.predict(x_lgb, num_iteration=self.lgb_best_iter,
                                      num_threads=NUM_THREADS)
                            for b in self.boosters], axis=0)

        # --- CatBoost: asset_id(int) + raw ---
        cb_df = pd.DataFrame(raw, columns=self.feature_cols, copy=False)
        cb_df.insert(0, "asset_id", test["asset_id"].to_numpy(np.int32))
        cb_pred = np.mean([m.predict(cb_df, thread_count=NUM_THREADS)
                           for m in self.cb_models], axis=0)

        # --- NN: 标准化 raw ---
        torch = self._torch
        xs = np.nan_to_num((raw - self.nn_mean) / self.nn_std,
                           nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        xt = torch.from_numpy(xs)
        at = torch.from_numpy(test["asset_id"].to_numpy(np.int64))
        nn_pred_parts = []
        with torch.no_grad():
            for i in range(0, len(xt), 8192):
                xb, ab = xt[i:i+8192], at[i:i+8192]
                nn_pred_parts.append(np.mean([m(xb, ab).numpy() for m in self.nns], axis=0))
        nn_pred = np.concatenate(nn_pred_parts)

        # --- 融合 ---
        pred = self.w_lgb * lgb_pred + self.w_cb * cb_pred + self.w_nn * nn_pred
        pred = np.where(np.isfinite(pred), pred, 0.0)
        return pred.astype(np.float64)
