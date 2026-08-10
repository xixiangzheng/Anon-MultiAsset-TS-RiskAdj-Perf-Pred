"""神经网络 MLP baseline（GPU）。验证 NN 能否达到 ~0.001+ holdout R²。

MLP: asset_id embedding + 323 raw → 512→256→128→1，dropout，加权 MSE。
单 holdout(末15%时间) 验证。若 R² 可接受，再训全量出 test 预测入集成。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

STRAT = "/mnt/iscsi/hd/xxz/public_release_lightgbm_baseline/examples/lightgbm_baseline"
if STRAT not in sys.path:
    sys.path.insert(0, STRAT)
from data_utils import manifest_files, feature_columns_from_path  # noqa: E402

DATA_ROOT = Path("/mnt/iscsi/hd/xxz/public_release_20260630/data")
DEV = "cuda:0"
N_ASSET = 15
FEATS = None  # 设为 None 在 main 里读


def wr2_np(y, p, w):
    d = float(np.sum(w * y * y)); return 0.0 if d <= 0 else 1 - float(np.sum(w * (y - p) ** 2) / d)


class MLP(nn.Module):
    def __init__(self, n_feat, emb_dim=8, hidden=(512, 256, 128), dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(N_ASSET, emb_dim)
        layers = []
        d = n_feat + emb_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.GELU(), nn.BatchNorm1d(h), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, asset):
        e = self.emb(asset)
        return self.net(torch.cat([x, e], 1)).squeeze(-1)


def main():
    paths = manifest_files(DATA_ROOT, "train")
    feats = feature_columns_from_path(paths[0])
    pf = pd.read_parquet(paths, columns=["time_id", "asset_id", "weight", "target"] + feats)
    pf[feats] = np.nan_to_num(pf[feats].to_numpy(np.float32))
    print(f"loaded {len(pf):,} rows", flush=True)

    times = np.sort(pf["time_id"].unique())
    ho = set(times[-max(1, int(len(times) * 0.15)):].tolist())
    is_va = pf["time_id"].isin(ho).to_numpy()
    tr_df, va_df = pf[~is_va], pf[is_va]
    print(f"train {len(tr_df):,} / holdout {len(va_df):,}", flush=True)

    def prep(df, mean=None, std=None):
        x = df[feats].to_numpy(np.float32)
        if mean is None:
            mean = x.mean(0); std = x.std(0) + 1e-6
        x = (x - mean) / std
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        a = df["asset_id"].to_numpy(np.int64)
        y = pd.to_numeric(df["target"], errors="coerce").fillna(0).to_numpy(np.float32)
        w = pd.to_numeric(df["weight"], errors="coerce").fillna(0).clip(lower=0).to_numpy(np.float32)
        return x, a, y, w, mean, std

    xt, at, yt, wt, mean, std = prep(tr_df)
    xv, av, yv, wv, _, _ = prep(va_df, mean, std)

    Xt = torch.from_numpy(xt).to(DEV); At = torch.from_numpy(at).to(DEV); Yt = torch.from_numpy(yt).to(DEV); Wt = torch.from_numpy(wt).to(DEV)
    Xv = torch.from_numpy(xv).to(DEV); Av = torch.from_numpy(av).to(DEV); Yv = torch.from_numpy(yv).to(DEV); Wv = torch.from_numpy(wv).to(DEV)

    model = MLP(len(feats)).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    bs = 16384; n_tr = len(Xt); best = -9; best_state = None; patience = 5; bad = 0
    for ep in range(40):
        model.train(); perm = torch.randperm(n_tr, device=DEV)
        t0 = time.time(); tot_loss = 0
        for i in range(0, n_tr, bs):
            idx = perm[i:i+bs]
            opt.zero_grad()
            p = model(Xt[idx], At[idx])
            loss = (Wt[idx] * (p - Yt[idx])**2).mean()
            loss.backward(); opt.step(); tot_loss += loss.item()*len(idx)
        model.eval()
        with torch.no_grad():
            pv = model(Xv, Av)
            r2 = wr2_np(yv, pv.cpu().numpy(), wv)
        print(f"ep{ep} loss={tot_loss/n_tr:.6f} holdout_wr2={r2:+.5f} ({time.time()-t0:.0f}s) best={best:+.5f}", flush=True)
        if r2 > best:
            best = r2; best_state = {k: v.clone() for k,v in model.state_dict().items()}; bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"early stop at ep{ep}", flush=True); break
    print(f"\nBEST holdout_wr2 = {best:+.5f}", flush=True)


if __name__ == "__main__":
    main()
