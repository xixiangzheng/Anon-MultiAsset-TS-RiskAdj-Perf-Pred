"""更大 NN：hidden=(1024,512,256), emb32, 20 seed, + ratio 特征。
当前 ratio_nn_emb32 (hidden 512/256/128) 权重 0.39 是集成最大贡献者。
预期：更大 NN + 更多 seed 显著提升 holdout。
"""
from __future__ import annotations
import sys, time, json, pickle, gc
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn

STRAT = "/mnt/iscsi/hd/xxz/public_release_lightgbm_baseline/examples/lightgbm_baseline"
if STRAT not in sys.path: sys.path.insert(0, STRAT)
from data_utils import manifest_files, feature_columns_from_path
DATA_ROOT = Path("/mnt/iscsi/hd/xxz/public_release_20260630/data")
RUN = Path("/mnt/iscsi/hd/xxz/runs"); SUB = Path("/mnt/iscsi/hd/xxz/submissions")
DEV = "cuda:4"; N_ASSET = 15


def wr2(y, p, w):
    d = float(np.sum(w*y*y)); return 0.0 if d<=0 else 1-float(np.sum(w*(y-p)**2)/d)


class MLP(nn.Module):
    def __init__(self, n_feat, emb_dim=32, hidden=(1024,512,256), dropout=0.3, in_drop=0.2):
        super().__init__(); self.emb=nn.Embedding(N_ASSET,emb_dim); self.in_drop=in_drop
        layers=[]; d=n_feat+emb_dim
        for h in hidden: layers+=[nn.Linear(d,h),nn.GELU(),nn.BatchNorm1d(h),nn.Dropout(dropout)]; d=h
        layers+=[nn.Linear(d,1)]; self.net=nn.Sequential(*layers)
    def forward(self,x,a):
        if self.training and self.in_drop>0:
            x=x*(torch.rand(x.shape[0],x.shape[1],device=x.device)>self.in_drop).float()/(1-self.in_drop)
        return self.net(torch.cat([x,self.emb(a)],1)).squeeze(-1)


def add_ratio(F, ratios):
    cols=[]
    for ni,di in ratios:
        fd=np.clip(F[:,di],1e-8,np.percentile(F[:,di],99))
        r=F[:,ni]/fd
        r=np.clip(r,np.percentile(r,1),np.percentile(r,99))
        r=np.nan_to_num(r,nan=0,posinf=0,neginf=0)
        cols.append(r.astype(np.float32))
    return np.column_stack(cols)


def main():
    paths = manifest_files(DATA_ROOT,"train"); feats = feature_columns_from_path(paths[0])
    rs = json.loads((RUN/"ratio_top50.json").read_text())["ratios"]
    ratios = [(feats.index(r[2]), feats.index(r[3])) for r in rs]
    pf = pd.read_parquet(paths, columns=["time_id","asset_id","weight","target"]+feats)
    pf[feats] = np.nan_to_num(pf[feats].to_numpy(np.float32))
    F_all = pf[feats].to_numpy(np.float32); R_all = add_ratio(F_all, ratios)
    te = pd.read_parquet(manifest_files(DATA_ROOT,"test"),columns=["row_id","asset_id"]+feats)
    te[feats] = np.nan_to_num(te[feats].to_numpy(np.float32)); te = te.sort_values("row_id").reset_index(drop=True)
    F_te = te[feats].to_numpy(np.float32); R_te = add_ratio(F_te, ratios)
    times = np.sort(pf["time_id"].unique()); ho = set(times[-max(1,int(len(times)*0.15)):].tolist())
    is_va = pf["time_id"].isin(ho).to_numpy()
    y32 = pd.to_numeric(pf["target"],errors="coerce").fillna(0).to_numpy(np.float32)
    w32 = pd.to_numeric(pf["weight"],errors="coerce").fillna(0).clip(lower=0).to_numpy(np.float32)
    yv = y32[is_va].astype(np.float64); wv = w32[is_va].astype(np.float64)
    print(f"data: train {len(pf):,} holdout {is_va.sum():,}", flush=True)

    raw_mean = F_all[~is_va].mean(0); raw_std = F_all[~is_va].std(0)+1e-6
    raw_mean = np.concatenate([raw_mean, R_all[~is_va].mean(0)])
    raw_std = np.concatenate([raw_std, R_all[~is_va].std(0)+1e-6])
    def stdize(F, R):
        X = np.concatenate([F,R],1).astype(np.float32)
        return np.nan_to_num((X-raw_mean)/raw_std, nan=0, posinf=0, neginf=0).astype(np.float32)
    Xtr_s = torch.from_numpy(stdize(F_all[~is_va], R_all[~is_va])).to(DEV)
    Xva_s = torch.from_numpy(stdize(F_all[is_va], R_all[is_va])).to(DEV)
    Xte_s = torch.from_numpy(stdize(F_te, R_te)).to(DEV)
    Atr = torch.from_numpy(pf["asset_id"].to_numpy(np.int64)[~is_va]).to(DEV)
    Ava = torch.from_numpy(pf["asset_id"].to_numpy(np.int64)[is_va]).to(DEV)
    Ate = torch.from_numpy(te["asset_id"].to_numpy(np.int64)).to(DEV)
    Ytr = torch.from_numpy(y32[~is_va]).to(DEV); Wtr = torch.from_numpy(w32[~is_va]).to(DEV)
    n_feat = len(feats)+len(ratios)
    bs = 16384; n_tr = len(Xtr_s)

    # 多架构实验
    configs = [
        ("ratio_nn_big", dict(emb_dim=32, hidden=(1024,512,256), dropout=0.3, in_drop=0.2)),
        ("ratio_nn_big_in0", dict(emb_dim=32, hidden=(1024,512,256), dropout=0.3, in_drop=0.0)),
        ("ratio_nn_emb64", dict(emb_dim=64, hidden=(1024,512,256), dropout=0.3, in_drop=0.2)),
    ]
    seeds = list(range(2026, 2046))  # 20 seed

    oofs = {}; te_preds = {}
    for name, kw in configs:
        print(f"\n=== {name} ({seeds[0]}-{seeds[-1]}, {len(seeds)} seeds) ===", flush=True); t0=time.time()
        acc_va = []; acc_te = []
        for sd in seeds:
            torch.manual_seed(sd); m = MLP(n_feat, **kw).to(DEV)
            opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-5)
            m.train(); perm = torch.randperm(n_tr, device=DEV)
            for i in range(0, n_tr, bs):
                idx = perm[i:i+bs]; opt.zero_grad()
                loss = (Wtr[idx]*(m(Xtr_s[idx],Atr[idx])-Ytr[idx])**2).mean(); loss.backward(); opt.step()
            m.eval()
            with torch.no_grad():
                pv = [m(Xva_s[i:i+16384],Ava[i:i+16384]).cpu().numpy() for i in range(0,len(Xva_s),16384)]
                pt = [m(Xte_s[i:i+16384],Ate[i:i+16384]).cpu().numpy() for i in range(0,len(Xte_s),16384)]
                acc_va.append(np.concatenate(pv)); acc_te.append(np.concatenate(pt))
        oofs[name] = np.mean(acc_va, 0); te_preds[name] = np.mean(acc_te, 0)
        print(f"  holdout R²={wr2(yv,oofs[name],wv):+.5f} ({time.time()-t0:.0f}s)", flush=True)
        # 增量保存
        pickle.dump({"oofs":oofs,"te_preds":te_preds,"yv":yv,"wv":wv,
                     "row_id":te["row_id"].to_numpy()}, open(RUN/"ratio_nn_big_partial.pkl","wb"))

    for k,p in te_preds.items():
        p = np.where(np.isfinite(p), p, 0.0)
        pd.DataFrame({"row_id":te["row_id"],"target":p}).to_csv(SUB/f"{k}.csv", index=False)
    pickle.dump({"oofs":oofs,"te_preds":te_preds,"yv":yv,"wv":wv,"row_id":te["row_id"].to_numpy()},
                open(RUN/"ratio_nn_big_oof.pkl","wb"))
    print(f"\n[done] {len(oofs)} models saved.", flush=True)


if __name__ == "__main__":
    main()
