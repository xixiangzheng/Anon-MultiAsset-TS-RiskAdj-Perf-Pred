"""FT-Transformer（前沿表格DL）：特征tokenize + 自注意力，正经训练。

突破小MLP的弱表现：每个特征→embedding token，transformer自注意力捕获特征交互。
warmup+cosine LR、多epoch、early stop。holdout上看能否突破GBDT的0.0017天花板。
"""
from __future__ import annotations
import sys, time, math, json
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn

STRAT="/mnt/iscsi/hd/xxz/public_release_lightgbm_baseline/examples/lightgbm_baseline"
if STRAT not in sys.path: sys.path.insert(0,STRAT)
from data_utils import manifest_files, feature_columns_from_path  # noqa: E402
DATA_ROOT=Path("/mnt/iscsi/hd/xxz/public_release_20260630/data"); DEV="cuda:0"; N_ASSET=15


def wr2(y,p,w):
    d=float(np.sum(w*y*y)); return 0.0 if d<=0 else 1-float(np.sum(w*(y-p)**2)/d)


class FTTransformer(nn.Module):
    def __init__(self, n_feat, d=64, nhead=8, nlayer=3, ff=256, dropout=0.2, n_asset=15, emb_dim=16):
        super().__init__()
        self.ft_tok = nn.Linear(n_feat, d)   # 特征 tokenize: 每个特征→d维(共享线性? 不,逐特征)
        # 实际: 逐特征 tokenize (FT-Transformer 标准做法)
        self.feat_w = nn.Parameter(torch.randn(n_feat, d) * 0.02)  # 每特征独立投影权重
        self.feat_b = nn.Parameter(torch.zeros(n_feat, d))
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)  # CLS token
        self.asset_emb = nn.Embedding(n_asset, d)
        self.pos = nn.Parameter(torch.randn(1, n_feat + 2, d) * 0.02)  # 位置编码(n_feat + cls + asset)
        enc_layer = nn.TransformerEncoderLayer(d_model=d, nhead=nhead, dim_feedforward=ff,
                                               dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayer)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, 1))

    def forward(self, x, asset):  # x [B, n_feat], asset [B]
        B = x.shape[0]
        tok = x.unsqueeze(-1) * self.feat_w + self.feat_b  # [B, n_feat, d]
        cls = self.cls.expand(B, -1, -1)  # [B, 1, d]
        ast = self.asset_emb(asset).unsqueeze(1)  # [B, 1, d]
        tokens = torch.cat([cls, ast, tok], dim=1)  # [B, n_feat+2, d]
        tokens = tokens + self.pos
        out = self.encoder(tokens)  # [B, n_feat+2, d]
        return self.head(out[:, 0]).squeeze(-1)  # CLS → predict


def main():
    paths = manifest_files(DATA_ROOT, "train")[:3]  # 3分区
    sel = json.load(open("/mnt/iscsi/hd/xxz/runs/top100_features.json"))["sel_features"][:50]  # top-50(减token)
    feats = sel
    pf = pd.read_parquet(paths, columns=["time_id", "asset_id", "weight", "target"] + feats)
    pf[feats] = np.nan_to_num(pf[feats].to_numpy(np.float32))
    print(f"loaded {len(pf):,} rows, {len(feats)} feats (top-50)", flush=True)
    times = np.sort(pf["time_id"].unique()); ho = set(times[-max(1, int(len(times)*0.15)):].tolist())
    is_va = pf["time_id"].isin(ho).to_numpy(); tr, va = pf[~is_va].reset_index(drop=True), pf[is_va].reset_index(drop=True)
    mean = tr[feats].to_numpy(np.float32).mean(0); std = tr[feats].to_numpy(np.float32).std(0) + 1e-6
    def prep(df):
        x = np.nan_to_num((df[feats].to_numpy(np.float32) - mean) / std, nan=0, posinf=0, neginf=0).astype(np.float32)
        return torch.from_numpy(x).to(DEV), torch.from_numpy(df["asset_id"].to_numpy(np.int64)).to(DEV)
    Xt, At = prep(tr); Xv, Av = prep(va)
    ytr = pd.to_numeric(tr["target"], errors="coerce").fillna(0).to_numpy(np.float32)
    wtr = pd.to_numeric(tr["weight"], errors="coerce").fillna(0).clip(lower=0).to_numpy(np.float32)
    yv = pd.to_numeric(va["target"], errors="coerce").fillna(0).to_numpy(np.float64)
    wv = pd.to_numeric(va["weight"], errors="coerce").fillna(0).clip(lower=0).to_numpy(np.float64)
    Ytr = torch.from_numpy(ytr).to(DEV); Wtr = torch.from_numpy(wtr).to(DEV)

    torch.manual_seed(2026)
    m = FTTransformer(len(feats), d=64, nhead=8, nlayer=2, ff=256, dropout=0.2).to(DEV)  # 2层(减计算)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"FT-Transformer params: {n_params:,}", flush=True)
    bs = 8192; n_tr = len(Xt); n_batch = (n_tr + bs - 1) // bs
    epochs = 20; warmup = 2; lr0 = 1e-4
    opt = torch.optim.AdamW(m.parameters(), lr=lr0, weight_decay=1e-4)
    def lr_at(ep):
        if ep < warmup: return lr0 * (ep + 1) / warmup
        return lr0 * 0.5 * (1 + math.cos(math.pi * (ep - warmup) / (epochs - warmup)))
    best = -9; bad = 0
    for ep in range(epochs):
        lr = lr_at(ep)
        for g in opt.param_groups: g["lr"] = lr
        m.train(); perm = torch.randperm(n_tr, device=DEV); t0 = time.time(); tot = 0
        for i in range(0, n_tr, bs):
            idx = perm[i:i+bs]; opt.zero_grad()
            loss = (Wtr[idx] * (m(Xt[idx], At[idx]) - Ytr[idx])**2).mean()
            loss.backward(); opt.step(); tot += loss.item() * len(idx)
        m.eval()
        with torch.no_grad():
            ps = []
            for i in range(0, len(Xv), 16384): ps.append(m(Xv[i:i+16384], Av[i:i+16384]).cpu().numpy())
            r2 = wr2(yv, np.concatenate(ps), wv)
        print(f"ep{ep} lr={lr:.6f} loss={tot/n_tr:.6f} holdout={r2:+.5f} ({time.time()-t0:.0f}s) best={max(best,r2):+.5f}", flush=True)
        if r2 > best: best = r2; bad = 0
        else:
            bad += 1
            if bad >= 5: print("early stop"); break
    print(f"\nBEST holdout={best:+.5f} (GBDT 0.00170, MLP 0.0011)", flush=True)


if __name__ == "__main__":
    main()
