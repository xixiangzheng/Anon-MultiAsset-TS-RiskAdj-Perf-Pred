"""Mixup NN：训练时混合样本对(强力正则化)，解决 NN epoch0 过拟合。

mixup: x_mix=λ*x1+(1-λ)*x2, y_mix=λ*y1+(1-λ)*y2, λ~Beta(0.4,0.4)。
强制 NN 在"样本间"泛化，可能让 NN 训更多 epoch 达到更高 R²(突破0.0011)。
同时测 target noise injection(对 y 加高斯噪声)作为备选正则。
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn

STRAT="/mnt/iscsi/hd/xxz/public_release_lightgbm_baseline/examples/lightgbm_baseline"
if STRAT not in sys.path: sys.path.insert(0,STRAT)
from data_utils import manifest_files, feature_columns_from_path  # noqa: E402
DATA_ROOT=Path("/mnt/iscsi/hd/xxz/public_release_20260630/data"); DEV="cuda:0"; N_ASSET=15


def wr2(y,p,w):
    d=float(np.sum(w*y*y)); return 0.0 if d<=0 else 1-float(np.sum(w*(y-p)**2)/d)


class MLP(nn.Module):
    def __init__(self,n_feat,emb_dim=8,hidden=(512,256,128),dropout=0.3):
        super().__init__(); self.emb=nn.Embedding(N_ASSET,emb_dim)
        layers=[]; d=n_feat+emb_dim
        for h in hidden: layers+=[nn.Linear(d,h),nn.GELU(),nn.BatchNorm1d(h),nn.Dropout(dropout)]; d=h
        layers+=[nn.Linear(d,1)]; self.net=nn.Sequential(*layers)
    def forward(self,x,a): return self.net(torch.cat([x,self.emb(a)],1)).squeeze(-1)


def main():
    paths=manifest_files(DATA_ROOT,"train")[:3]; feats=feature_columns_from_path(paths[0])
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
    bs=16384; n_tr=len(Xt); alpha=0.4  # mixup alpha

    for mode in ["baseline","mixup","noise"]:
        torch.manual_seed(2026)
        m=MLP(len(feats)).to(DEV); opt=torch.optim.Adam(m.parameters(),lr=1e-3,weight_decay=1e-5)
        best=-9; bad=0
        for ep in range(20):
            m.train(); perm=torch.randperm(n_tr,device=DEV); t0=time.time()
            for i in range(0,n_tr,bs):
                idx=perm[i:i+bs]; opt.zero_grad()
                x=Xt[idx]; a=At[idx]; y=Ytr[idx]; w=Wtr[idx]
                if mode=="mixup":
                    lam=np.random.beta(alpha,alpha)
                    perm2=torch.randperm(len(idx),device=DEV)
                    x=lam*x+(1-lam)*x[perm2]; y=lam*y+(1-lam)*y[perm2]
                    w=lam*w+(1-lam)*w[perm2]  # 混合权重也混合
                elif mode=="noise":
                    y=y+0.05*torch.randn_like(y)*y.std()  # target 加噪
                loss=(w*(m(x,a)-y)**2).mean(); loss.backward(); opt.step()
            m.eval()
            with torch.no_grad():
                ps=[]
                for j in range(0,len(Xv),16384): ps.append(m(Xv[j:j+16384],Av[j:j+16384]).cpu().numpy())
                r2=wr2(yv,np.concatenate(ps),wv)
            if r2>best: best=r2; bad=0
            else:
                bad+=1
                if bad>=5: break
        print(f"{mode:10s}: best holdout={best:+.5f}",flush=True)


if __name__=="__main__":
    main()
