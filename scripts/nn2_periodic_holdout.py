"""NN-2 周期嵌入(Periodic) MLP：每个特征展成 sin/cos 频率族，再 MLP。

目的：架构与 NN-1(线性输入MLP)截然不同 → 产生去相关预测，为 5 模型集成提供第二个多样性来源。
holdout 验证：(1) R² 是否接近/超过 NN-1(0.00112)；(2) 与 NN-1 预测的相关性是否够低。
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
N_FREQ = 8  # 每特征 8 个频率 → sin+cos = 16 维/特征


def wr2_np(y, p, w):
    d = float(np.sum(w * y * y)); return 0.0 if d <= 0 else 1 - float(np.sum(w * (y - p) ** 2) / d)


class PeriodicMLP(nn.Module):
    def __init__(self, n_feat, emb_dim=8, hidden=(512, 256, 128), dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(N_ASSET, emb_dim)
        # 固定 log-spaced 频率 (1e-1 ~ 1e2)
        freq = torch.logspace(-1, 2, N_FREQ)  # [n_freq]
        self.register_buffer("freq", freq)  # broadcast 用
        d = n_feat * 2 * N_FREQ + emb_dim
        layers = []
        for h in hidden:
            layers += [nn.Linear(d, h), nn.GELU(), nn.BatchNorm1d(h), nn.Dropout(dropout)]; d = h
        layers += [nn.Linear(d, 1)]; self.net = nn.Sequential(*layers)

    def forward(self, x, a):
        # x: [B, n_feat] → [B, n_feat, 1] * freq [n_freq] → [B, n_feat, n_freq]
        xf = x.unsqueeze(-1) * self.freq  # [B, n_feat, n_freq]
        per = torch.cat([torch.sin(xf), torch.cos(xf)], dim=-1)  # [B, n_feat, 2*n_freq]
        per = per.flatten(1)  # [B, n_feat*2*n_freq]
        return self.net(torch.cat([per, self.emb(a)], 1)).squeeze(-1)


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
    m = PeriodicMLP(len(feats)).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5)
    bs=8192; n_tr=len(Xt); best=-9; bad=0
    print(f"params: {sum(p.numel() for p in m.parameters()):,}", flush=True)
    for ep in range(30):
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
            if bad>=6: print("early stop"); break
    print(f"\nBEST holdout = {best:+.5f} (NN-1=0.00112)", flush=True)


if __name__ == "__main__":
    main()
