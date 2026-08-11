"""NN 优化训练：cosine LR + 大batch + 多epoch + 早停，试图突破 epoch0 过拟合峰值(0.0011)。

之前 NN 在 epoch0 就峰值(过拟合)。这里用 cosine LR schedule + 大batch(更稳梯度) + 适度正则，
看能否让 NN 训练更多 epoch 达到更高 holdout R²。若 NN 能到 0.002+，集成可冲更高。
"""
from __future__ import annotations
import sys, time, math
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn

STRAT="/mnt/iscsi/hd/xxz/public_release_lightgbm_baseline/examples/lightgbm_baseline"
if STRAT not in sys.path: sys.path.insert(0,STRAT)
from data_utils import manifest_files, feature_columns_from_path  # noqa: E402
DATA_ROOT=Path("/mnt/iscsi/hd/xxz/public_release_20260630/data"); DEV="cuda:0"; N_ASSET=15


def wr2(y,p,w):
    d=float(np.sum(w*y*y)); return 0.0 if d<=0 else 1-float(np.sum(w*(y-p)**2)/d)


class MLP(nn.Module):
    def __init__(self,n_feat,emb_dim=16,hidden=(512,256,128),dropout=0.4):
        super().__init__(); self.emb=nn.Embedding(N_ASSET,emb_dim)
        layers=[]; d=n_feat+emb_dim
        for h in hidden: layers+=[nn.Linear(d,h),nn.GELU(),nn.BatchNorm1d(h),nn.Dropout(dropout)]; d=h
        layers+=[nn.Linear(d,1)]; self.net=nn.Sequential(*layers)
    def forward(self,x,a): return self.net(torch.cat([x,self.emb(a)],1)).squeeze(-1)


def main():
    paths=manifest_files(DATA_ROOT,"train"); feats=feature_columns_from_path(paths[0])
    pf=pd.read_parquet(paths,columns=["time_id","asset_id","weight","target"]+feats)
    pf[feats]=np.nan_to_num(pf[feats].to_numpy(np.float32))
    times=np.sort(pf["time_id"].unique()); ho=set(times[-max(1,int(len(times)*0.15)):].tolist())
    is_va=pf["time_id"].isin(ho).to_numpy(); tr,va=pf[~is_va].reset_index(drop=True),pf[is_va].reset_index(drop=True)
    mean=tr[feats].to_numpy(np.float32).mean(0); std=tr[feats].to_numpy(np.float32).std(0)+1e-6
    def prep(df):
        x=np.nan_to_num((df[feats].to_numpy(np.float32)-mean)/std,nan=0,posinf=0,neginf=0).astype(np.float32)
        return torch.from_numpy(x).to(DEV), torch.from_numpy(df["asset_id"].to_numpy(np.int64)).to(DEV)
    Xt,At=prep(tr); Xv,Av=prep(va)
    ytr=pd.to_numeric(tr["target"],errors="coerce").fillna(0).to_numpy(np.float32)
    wtr=pd.to_numeric(tr["weight"],errors="coerce").fillna(0).clip(lower=0).to_numpy(np.float32)
    yv=pd.to_numeric(va["target"],errors="coerce").fillna(0).to_numpy(np.float64)
    wv=pd.to_numeric(va["weight"],errors="coerce").fillna(0).clip(lower=0).to_numpy(np.float64)
    Ytr=torch.from_numpy(ytr).to(DEV); Wtr=torch.from_numpy(wtr).to(DEV)
    torch.manual_seed(2026); m=MLP(len(feats)).to(DEV)
    bs=32768; n_tr=len(Xt); n_batch=(n_tr+bs-1)//bs
    epochs=8; lr0=1e-3; opt=torch.optim.AdamW(m.parameters(),lr=lr0,weight_decay=1e-4)
    best=-9
    for ep in range(epochs):
        # cosine LR per epoch 区间
        lr=lr0*0.5*(1+math.cos(math.pi*ep/epochs))
        for g in opt.param_groups: g["lr"]=lr
        m.train(); perm=torch.randperm(n_tr,device=DEV); t0=time.time(); tot=0
        for i in range(0,n_tr,bs):
            idx=perm[i:i+bs]; opt.zero_grad(); loss=(Wtr[idx]*(m(Xt[idx],At[idx])-Ytr[idx])**2).mean()
            loss.backward(); opt.step(); tot+=loss.item()*len(idx)
        m.eval()
        with torch.no_grad():
            ps=[]
            for i in range(0,len(Xv),16384): ps.append(m(Xv[i:i+16384],Av[i:i+16384]).cpu().numpy())
            r2=wr2(yv,np.concatenate(ps),wv)
        print(f"ep{ep} lr={lr:.5f} loss={tot/n_tr:.6f} holdout={r2:+.5f} ({time.time()-t0:.0f}s) best={max(best,r2):+.5f}",flush=True)
        best=max(best,r2)
    print(f"\nBEST holdout={best:+.5f} (原NN0.0011, v2 0.0007)",flush=True)


if __name__=="__main__":
    main()
