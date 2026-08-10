"""NN v2：更强正则 + 更低 lr，让 NN 能训更多轮(原版 epoch0 就过拟合)。

改动：lr 1e-3→3e-4，dropout 0.3→0.5，weight_decay 1e-5→1e-4，加 early stop 找最佳 epoch。
holdout 验证：目标 R² > 0.00112(原版)，越强对集成越有利。
"""
from __future__ import annotations

import sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn

STRAT = "/mnt/iscsi/hd/xxz/public_release_lightgbm_baseline/examples/lightgbm_baseline"
if STRAT not in sys.path: sys.path.insert(0, STRAT)
from data_utils import manifest_files, feature_columns_from_path  # noqa: E402

DATA_ROOT = Path("/mnt/iscsi/hd/xxz/public_release_20260630/data")
DEV = "cuda:0"; N_ASSET = 15


def wr2_np(y, p, w):
    d = float(np.sum(w * y * y)); return 0.0 if d <= 0 else 1 - float(np.sum(w * (y - p) ** 2) / d)


class MLP(nn.Module):
    def __init__(self, n_feat, emb_dim=8, hidden=(256, 128), dropout=0.5):
        super().__init__(); self.emb = nn.Embedding(N_ASSET, emb_dim)
        layers = []; d = n_feat + emb_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.GELU(), nn.BatchNorm1d(h), nn.Dropout(dropout)]; d = h
        layers += [nn.Linear(d, 1)]; self.net = nn.Sequential(*layers)
    def forward(self, x, a): return self.net(torch.cat([x, self.emb(a)], 1)).squeeze(-1)


def main():
    paths = manifest_files(DATA_ROOT, "train")
    feats = feature_columns_from_path(paths[0])
    pf = pd.read_parquet(paths, columns=["time_id", "asset_id", "weight", "target"] + feats)
    pf[feats] = np.nan_to_num(pf[feats].to_numpy(np.float32))
    print(f"loaded {len(pf):,}", flush=True)
    times = np.sort(pf["time_id"].unique()); ho = set(times[-max(1, int(len(times)*0.15)):].tolist())
    is_va = pf["time_id"].isin(ho).to_numpy(); tr_df, va_df = pf[~is_va], pf[is_va]
    def prep(df, mean=None, std=None):
        x = df[feats].to_numpy(np.float32)
        if mean is None: mean = x.mean(0); std = x.std(0)+1e-6
        x = np.nan_to_num((x-mean)/std, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return (x, df["asset_id"].to_numpy(np.int64),
                pd.to_numeric(df["target"],errors="coerce").fillna(0).to_numpy(np.float32),
                pd.to_numeric(df["weight"],errors="coerce").fillna(0).clip(lower=0).to_numpy(np.float32), mean, std)
    xt, at, yt, wt, mean, std = prep(tr_df); xv, av, yv, wv, _, _ = prep(va_df, mean, std)
    Xt=torch.from_numpy(xt).to(DEV); At=torch.from_numpy(at).to(DEV); Yt=torch.from_numpy(yt).to(DEV); Wt=torch.from_numpy(wt).to(DEV)
    Xv=torch.from_numpy(xv).to(DEV); Av=torch.from_numpy(av).to(DEV)
    torch.manual_seed(2026)
    m = MLP(len(feats)).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
    bs=16384; n_tr=len(Xt); best=-9; bad=0
    for ep in range(40):
        m.train(); perm=torch.randperm(n_tr, device=DEV); t0=time.time(); tot=0
        for i in range(0, n_tr, bs):
            idx=perm[i:i+bs]; opt.zero_grad()
            loss=(Wt[idx]*(m(Xt[idx],At[idx])-Yt[idx])**2).mean()
            loss.backward(); opt.step(); tot+=loss.item()*len(idx)
        m.eval()
        with torch.no_grad(): r2=wr2_np(yv, m(Xv,Av).cpu().numpy(), wv)
        print(f"ep{ep} loss={tot/n_tr:.6f} holdout={r2:+.5f} ({time.time()-t0:.0f}s) best={best:+.5f}", flush=True)
        if r2>best: best=r2; bad=0
        else:
            bad+=1
            if bad>=8: print("early stop"); break
    print(f"\nBEST holdout = {best:+.5f} (原版 0.00112)", flush=True)


if __name__ == "__main__":
    main()
