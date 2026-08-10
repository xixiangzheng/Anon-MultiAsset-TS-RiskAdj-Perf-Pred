"""NN 全量训练 + test 预测（1 epoch × 3 seed，最佳 epoch=0）。出 nn_submission 供集成。"""
from __future__ import annotations

import sys
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
DEV = "cuda:0"; N_ASSET = 15
SEEDS = [2026, 2027, 2028]


class MLP(nn.Module):
    def __init__(self, n_feat, emb_dim=8, hidden=(512, 256, 128), dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(N_ASSET, emb_dim)
        layers = []; d = n_feat + emb_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.GELU(), nn.BatchNorm1d(h), nn.Dropout(dropout)]; d = h
        layers += [nn.Linear(d, 1)]; self.net = nn.Sequential(*layers)
    def forward(self, x, a):
        return self.net(torch.cat([x, self.emb(a)], 1)).squeeze(-1)


def main():
    paths = manifest_files(DATA_ROOT, "train")
    feats = feature_columns_from_path(paths[0])
    tr = pd.read_parquet(paths, columns=["asset_id", "weight", "target"] + feats)
    tr[feats] = np.nan_to_num(tr[feats].to_numpy(np.float32))
    te = pd.read_parquet(manifest_files(DATA_ROOT, "test"), columns=["row_id", "asset_id"] + feats)
    te[feats] = np.nan_to_num(te[feats].to_numpy(np.float32))
    te = te.sort_values("row_id").reset_index(drop=True)
    print(f"train {len(tr):,} / test {len(te):,}", flush=True)

    mean = tr[feats].to_numpy(np.float32).mean(0)
    std = tr[feats].to_numpy(np.float32).std(0) + 1e-6
    def prep(df):
        x = np.nan_to_num((df[feats].to_numpy(np.float32) - mean) / std, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return torch.from_numpy(x).to(DEV), torch.from_numpy(df["asset_id"].to_numpy(np.int64)).to(DEV)
    Xt, At = prep(tr); Xe, Ae = prep(te)
    Yt = torch.from_numpy(pd.to_numeric(tr["target"], errors="coerce").fillna(0).to_numpy(np.float32)).to(DEV)
    Wt = torch.from_numpy(pd.to_numeric(tr["weight"], errors="coerce").fillna(0).clip(lower=0).to_numpy(np.float32)).to(DEV)

    n_tr = len(Xt); bs = 16384
    test_preds = []
    for sd in SEEDS:
        torch.manual_seed(sd); np.random.seed(sd)
        m = MLP(len(feats)).to(DEV)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5)
        m.train(); perm = torch.randperm(n_tr, device=DEV)
        for i in range(0, n_tr, bs):
            idx = perm[i:i+bs]; opt.zero_grad()
            loss = (Wt[idx] * (m(Xt[idx], At[idx]) - Yt[idx])**2).mean()
            loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            # 分块预测 test (避免 OOM)
            ps = []
            for i in range(0, len(Xe), 200000):
                ps.append(m(Xe[i:i+200000], Ae[i:i+200000]).cpu().numpy())
            test_preds.append(np.concatenate(ps))
        print(f"seed {sd} done", flush=True)
    avg = np.mean(test_preds, axis=0); avg = np.where(np.isfinite(avg), avg, 0.0)
    out = pd.DataFrame({"row_id": te["row_id"], "target": avg})
    out.to_csv("/mnt/iscsi/hd/xxz/submissions/nn_submission.csv", index=False)
    print(f"wrote nn_submission finite={np.isfinite(avg).all()} mean={avg.mean():+.4f} std={avg.std():.4f}", flush=True)


if __name__ == "__main__":
    main()
