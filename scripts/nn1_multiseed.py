"""nn1 多 seed 稳定化：训练 plain MLP 10 个 seed，平均 holdout(看R²是否提升) + 平均 test 出 nn1_10s。

nn1 最多样(corr 0.62)但单 seed 弱(0.00064)；多 seed 平均降噪后应更强、更稳，对集成更有价值。
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn

STRAT = "/mnt/iscsi/hd/xxz/public_release_lightgbm_baseline/examples/lightgbm_baseline"
if STRAT not in sys.path: sys.path.insert(0, STRAT)
from data_utils import manifest_files, feature_columns_from_path  # noqa: E402

DATA_ROOT = Path("/mnt/iscsi/hd/xxz/public_release_20260630/data")
DEV="cuda:0"; N_ASSET=15; SEEDS=list(range(2026,2036))  # 10 seeds


class MLP(nn.Module):
    def __init__(self,n_feat,emb_dim=8,hidden=(512,256,128),dropout=0.3):
        super().__init__(); self.emb=nn.Embedding(N_ASSET,emb_dim)
        layers=[]; d=n_feat+emb_dim
        for h in hidden: layers+=[nn.Linear(d,h),nn.GELU(),nn.BatchNorm1d(h),nn.Dropout(dropout)]; d=h
        layers+=[nn.Linear(d,1)]; self.net=nn.Sequential(*layers)
    def forward(self,x,a): return self.net(torch.cat([x,self.emb(a)],1)).squeeze(-1)


def wr2(y,p,w):
    d=float(np.sum(w*y*y)); return 0.0 if d<=0 else 1-float(np.sum(w*(y-p)**2)/d)


def main():
    paths=manifest_files(DATA_ROOT,"train"); feats=feature_columns_from_path(paths[0])
    pf=pd.read_parquet(paths,columns=["time_id","asset_id","weight","target"]+feats)
    pf[feats]=np.nan_to_num(pf[feats].to_numpy(np.float32))
    te=pd.read_parquet(manifest_files(DATA_ROOT,"test"),columns=["row_id","asset_id"]+feats)
    te[feats]=np.nan_to_num(te[feats].to_numpy(np.float32)); te=te.sort_values("row_id").reset_index(drop=True)
    times=np.sort(pf["time_id"].unique()); ho=set(times[-max(1,int(len(times)*0.15)):].tolist())
    is_va=pf["time_id"].isin(ho).to_numpy(); tr_df,va_df=pf[~is_va].reset_index(drop=True),pf[is_va].reset_index(drop=True)
    mean=tr_df[feats].to_numpy(np.float32).mean(0); std=tr_df[feats].to_numpy(np.float32).std(0)+1e-6
    def prep(df):
        x=np.nan_to_num((df[feats].to_numpy(np.float32)-mean)/std,nan=0,posinf=0,neginf=0).astype(np.float32)
        return torch.from_numpy(x).to(DEV), torch.from_numpy(df["asset_id"].to_numpy(np.int64)).to(DEV)
    Xt,At=prep(tr_df); Xv,Av=prep(va_df); Xe,Ae=prep(te)
    ytr=pd.to_numeric(tr_df["target"],errors="coerce").fillna(0).to_numpy(np.float32)
    wtr=pd.to_numeric(tr_df["weight"],errors="coerce").fillna(0).clip(lower=0).to_numpy(np.float32)
    yv=pd.to_numeric(va_df["target"],errors="coerce").fillna(0).to_numpy(np.float64)
    wv=pd.to_numeric(va_df["weight"],errors="coerce").fillna(0).clip(lower=0).to_numpy(np.float64)
    Ytr=torch.from_numpy(ytr).to(DEV); Wtr=torch.from_numpy(wtr).to(DEV)
    bs=16384; n_tr=len(Xt)
    hv=[]; tv=[]
    for i,sd in enumerate(SEEDS):
        torch.manual_seed(sd); m=MLP(len(feats)).to(DEV)
        opt=torch.optim.Adam(m.parameters(),lr=1e-3,weight_decay=1e-5)
        m.train(); perm=torch.randperm(n_tr,device=DEV)
        for j in range(0,n_tr,bs):
            idx=perm[j:j+bs]; opt.zero_grad(); loss=(Wtr[idx]*(m(Xt[idx],At[idx])-Ytr[idx])**2).mean(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            hv.append(torch.cat([m(Xv[k:k+16384],Av[k:k+16384]) for k in range(0,len(Xv),16384)]).cpu().numpy())
            tv.append(torch.cat([m(Xe[k:k+16384],Ae[k:k+16384]) for k in range(0,len(Xe),16384)]).cpu().numpy())
        # 增量 holdout R²
        inc=np.mean(hv,axis=0)
        print(f"seed {sd} ({i+1}/{len(SEEDS)}): 增量 holdout R²={wr2(yv,inc,wv):+.5f}", flush=True)
    final_h=np.mean(hv,axis=0); final_t=np.mean(tv,axis=0)
    print(f"\n10-seed nn1 holdout R²={wr2(yv,final_h,wv):+.5f} (单seed~0.00064)", flush=True)
    final_t=np.where(np.isfinite(final_t),final_t,0.0)
    pd.DataFrame({"row_id":te["row_id"],"target":final_t}).to_csv("/mnt/iscsi/hd/xxz/submissions/nn1_10s.csv",index=False)
    print(f"wrote nn1_10s.csv mean={final_t.mean():+.4f}", flush=True)


if __name__=="__main__":
    main()
