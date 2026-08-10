"""NN 多变体扫描：测多种架构的 holdout R² + 互相相关性，挑"多样+够强"的进集成。

每变体: 1 epoch × 1 seed，训练于 train-减holdout，预测 holdout(R²) + test。打印 R² 与相关性矩阵。
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


def wr2(y, p, w):
    d = float(np.sum(w*y*y)); return 0.0 if d<=0 else 1-float(np.sum(w*(y-p)**2)/d)


class MLP(nn.Module):
    def __init__(self, n_feat, emb_dim=8, hidden=(512,256,128), dropout=0.3, in_drop=0.0, act="gelu"):
        super().__init__(); self.emb=nn.Embedding(N_ASSET,emb_dim); self.in_drop=in_drop
        Act = {"gelu":nn.GELU,"relu":nn.ReLU}[act]
        layers=[]; d=n_feat+emb_dim
        for h in hidden: layers+=[nn.Linear(d,h),Act(),nn.BatchNorm1d(h),nn.Dropout(dropout)]; d=h
        layers+=[nn.Linear(d,1)]; self.net=nn.Sequential(*layers)
    def forward(self,x,a):
        if self.training and self.in_drop>0:
            x=x*(torch.rand(x.shape[0],x.shape[1],device=x.device)>self.in_drop).float()/(1-self.in_drop)
        return self.net(torch.cat([x,self.emb(a)],1)).squeeze(-1)


VARIANTS = [
    ("emb32",      dict(emb_dim=32, hidden=(512,256,128))),
    ("narrow",     dict(emb_dim=8,  hidden=(128,64,32))),
    ("wide1k",     dict(emb_dim=8,  hidden=(1024,))),
    ("indrop07",   dict(emb_dim=8,  hidden=(512,256,128), in_drop=0.7)),
    ("deep4",      dict(emb_dim=8,  hidden=(256,256,256,256))),
    ("relu",       dict(emb_dim=8,  hidden=(512,256,128), act="relu")),
]


def main():
    paths = manifest_files(DATA_ROOT, "train")
    feats = feature_columns_from_path(paths[0])
    pf = pd.read_parquet(paths, columns=["time_id","asset_id","weight","target"]+feats)
    pf[feats]=np.nan_to_num(pf[feats].to_numpy(np.float32))
    te = pd.read_parquet(manifest_files(DATA_ROOT,"test"), columns=["row_id","asset_id"]+feats)
    te[feats]=np.nan_to_num(te[feats].to_numpy(np.float32)); te=te.sort_values("row_id").reset_index(drop=True)
    print(f"loaded train {len(pf):,} test {len(te):,}", flush=True)
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

    holdout_preds={}; test_preds={}
    print(f"\n{'variant':10s} {'holdout_R2':>10s}", flush=True)
    for name,cfg in VARIANTS:
        torch.manual_seed(2026)
        m=MLP(len(feats),**cfg).to(DEV)
        opt=torch.optim.Adam(m.parameters(),lr=1e-3,weight_decay=1e-5)
        m.train(); perm=torch.randperm(n_tr,device=DEV)
        for i in range(0,n_tr,bs):
            idx=perm[i:i+bs]; opt.zero_grad(); loss=(Wtr[idx]*(m(Xt[idx],At[idx])-Ytr[idx])**2).mean(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            pv=torch.cat([m(Xv[i:i+16384],Av[i:i+16384]) for i in range(0,len(Xv),16384)]).cpu().numpy()
            pe=torch.cat([m(Xe[i:i+16384],Ae[i:i+16384]) for i in range(0,len(Xe),16384)]).cpu().numpy()
        holdout_preds[name]=pv; test_preds[name]=pe
        print(f"{name:10s} {wr2(yv,pv,wv):+10.5f}", flush=True)
    # 相关性(变体间)
    keys=list(holdout_preds)
    print("\n变体间相关性(holdout):", flush=True)
    for a in keys:
        print(f"  {a}: "+" ".join(f"{np.corrcoef(holdout_preds[a],holdout_preds[b])[0,1]:.2f}" for b in keys), flush=True)
    # 保存所有变体 test 预测
    for name in keys:
        pd.DataFrame({"row_id":te["row_id"],"target":np.where(np.isfinite(test_preds[name]),test_preds[name],0)}).to_csv(f"/mnt/iscsi/hd/xxz/submissions/nnvar_{name}.csv",index=False)
    print("\n已保存 nnvar_*.csv", flush=True)
    # 保存 holdout 预测供后续权重优化
    import pickle
    pickle.dump({"yv":yv,"wv":wv,"preds":holdout_preds}, open("/mnt/iscsi/hd/xxz/runs/nnvar_holdout.pkl","wb"))
    print("holdout preds saved to runs/nnvar_holdout.pkl", flush=True)


if __name__=="__main__":
    main()
