"""Deep & Cross Network (DCN)：显式特征交叉，全323特征，proper训练。

DCN 的 Cross 层显式建模特征交互(x_i×x_j的高阶组合)，比 MLP 更强、比 Transformer 快(O(n)而非O(n²))。
- Cross: 4层(捕捉4阶交互)
- Deep: 3层MLP(512/256/128)
- 合并 → 输出
proper训练(AdamW+cosine+grad clip+多epoch)。holdout 验证。
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


class CrossLayer(nn.Module):
    def __init__(self, d):
        super().__init__(); self.w = nn.Linear(d, 1, bias=False); self.b = nn.Parameter(torch.zeros(d))
    def forward(self, x0, xl):
        return x0 * self.w(xl) + self.b + xl  # x_0 ⊙ (W x_l + b) + x_l


class DCN(nn.Module):
    def __init__(self, n_feat, emb_dim=8, n_cross=4, deep=(512,256,128), dropout=0.3):
        super().__init__()
        d = n_feat + emb_dim
        self.emb = nn.Embedding(N_ASSET, emb_dim)
        self.cross = nn.ModuleList([CrossLayer(d) for _ in range(n_cross)])
        layers = []; din = d
        for h in deep: layers += [nn.Linear(din,h), nn.GELU(), nn.BatchNorm1d(h), nn.Dropout(dropout)]; din = h
        self.deep = nn.Sequential(*layers)
        self.head = nn.Linear(d + deep[-1], 1)

    def forward(self, x, asset):
        x0 = torch.cat([x, self.emb(asset)], 1)  # [B, n_feat+emb]
        xl = x0
        for layer in self.cross: xl = layer(x0, xl)
        d = self.deep(x0)
        return self.head(torch.cat([xl, d], 1)).squeeze(-1)


def main():
    paths=manifest_files(DATA_ROOT,"train")[:3]; feats=feature_columns_from_path(paths[0])
    pf=pd.read_parquet(paths,columns=["time_id","asset_id","weight","target"]+feats)
    pf[feats]=np.nan_to_num(pf[feats].to_numpy(np.float32))
    print(f"loaded {len(pf):,} rows, {len(feats)} feats",flush=True)
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

    torch.manual_seed(2026)
    m=DCN(len(feats)).to(DEV)
    print(f"DCN params: {sum(p.numel() for p in m.parameters()):,}",flush=True)
    bs=16384; n_tr=len(Xt); epochs=20; warmup=2; lr0=1e-3
    opt=torch.optim.AdamW(m.parameters(),lr=lr0,weight_decay=1e-4)
    def lr_at(ep):
        if ep<warmup: return lr0*(ep+1)/warmup
        return lr0*0.5*(1+math.cos(math.pi*(ep-warmup)/(epochs-warmup)))
    best=-9; bad=0
    for ep in range(epochs):
        lr=lr_at(ep); 
        for g in opt.param_groups: g["lr"]=lr
        m.train(); perm=torch.randperm(n_tr,device=DEV); t0=time.time(); tot=0; cnt=0
        for i in range(0,n_tr,bs):
            idx=perm[i:i+bs]; opt.zero_grad()
            loss=(Wtr[idx]*(m(Xt[idx],At[idx])-Ytr[idx])**2).mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
            tot+=loss.item()*len(idx); cnt+=len(idx)
        m.eval()
        with torch.no_grad():
            ps=[]
            for j in range(0,len(Xv),16384): ps.append(m(Xv[j:j+16384],Av[j:j+16384]).cpu().numpy())
            r2=wr2(yv,np.concatenate(ps),wv)
        print(f"ep{ep} lr={lr:.5f} loss={tot/cnt:.6f} holdout={r2:+.5f} ({time.time()-t0:.0f}s) best={max(best,r2):+.5f}",flush=True)
        if r2>best: best=r2; bad=0
        else:
            bad+=1
            if bad>=5: print("early stop"); break
    print(f"\nBEST holdout={best:+.5f} (GBDT 0.00170, MLP 0.0011, FT-Transformer top50 0.00148)",flush=True)


if __name__=="__main__":
    main()
