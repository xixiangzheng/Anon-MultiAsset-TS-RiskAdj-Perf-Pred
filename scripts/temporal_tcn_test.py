"""时序模型测试：因果卷积(TCN)按每个 asset 的时间序列建模。

突破"逐行独立"范式：用特征的时间序列形态预测。严格因果(只用过去+当前)。
每个 asset 的 [n_feat, T] 序列 → CausalConv1d(3层,感受野~13) → 每步预测 target。
holdout 上看是否突破逐行模型的天花板(0.0017)。
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F

STRAT="/mnt/iscsi/hd/xxz/public_release_lightgbm_baseline/examples/lightgbm_baseline"
if STRAT not in sys.path: sys.path.insert(0,STRAT)
from data_utils import manifest_files, feature_columns_from_path  # noqa: E402
DATA_ROOT=Path("/mnt/iscsi/hd/xxz/public_release_20260630/data"); DEV="cuda:0"; N_ASSET=15


def wr2(y,p,w):
    d=float(np.sum(w*y*y)); return 0.0 if d<=0 else 1-float(np.sum(w*(y-p)**2)/d)


class CausalConv1d(nn.Module):
    def __init__(self,ci,co,k):
        super().__init__(); self.conv=nn.Conv1d(ci,co,k,padding=k-1)  # left-causal: 右侧裁掉
    def forward(self,x):
        return self.conv(x)[..., :x.shape[-1]]  # 裁掉右侧 padding, 保持长度


class TCN(nn.Module):
    def __init__(self,n_feat,emb_dim=8,ch=128,k=5):
        super().__init__(); self.emb=nn.Embedding(N_ASSET,emb_dim)
        self.net=nn.Sequential(
            CausalConv1d(n_feat,ch,k), nn.GELU(), nn.BatchNorm1d(ch), nn.Dropout(0.2),
            CausalConv1d(ch,ch,k), nn.GELU(), nn.BatchNorm1d(ch), nn.Dropout(0.2),
            CausalConv1d(ch,64,k), nn.GELU(), nn.BatchNorm1d(64), nn.Dropout(0.2))
        self.head=nn.Linear(64+emb_dim,1)
    def forward(self,x,asset):  # x [1,n_feat,T], asset int
        h=self.net(x)  # [1,64,T]
        h=h.transpose(1,2)  # [1,T,64]
        e=self.emb(asset).expand(h.shape[1],-1)  # [T,emb]
        return self.head(torch.cat([h.squeeze(0),e],1)).squeeze(-1)  # [T]


def main():
    sel=json.load(open("/mnt/iscsi/hd/xxz/runs/top100_features.json"))["sel_features"]  # top-100
    paths=manifest_files(DATA_ROOT,"train")[:2]  # 2分区
    pf=pd.read_parquet(paths,columns=["time_id","asset_id","weight","target"]+sel)
    pf[sel]=np.nan_to_num(pf[sel].to_numpy(np.float32))
    print(f"loaded {len(pf):,} rows, {len(sel)} features",flush=True)
    # 全局标准化
    mean=pf[sel].to_numpy(np.float32).mean(0); std=pf[sel].to_numpy(np.float32).std(0)+1e-6
    pf[sel]=np.nan_to_num((pf[sel].to_numpy(np.float32)-mean)/std,nan=0,posinf=0,neginf=0).astype(np.float32)
    # 全局 holdout(末15% time)
    times=np.sort(pf["time_id"].unique()); ho=set(times[-max(1,int(len(times)*0.15)):].tolist())
    pf["is_va"]=pf["time_id"].isin(ho)
    # 按 asset 分组, 时间排序
    assets_data={}
    for a,g in pf.sort_values("time_id").groupby("asset_id"):
        assets_data[int(a)]=(g[sel].to_numpy(np.float32), pd.to_numeric(g["target"],errors="coerce").fillna(0).to_numpy(np.float32),
                              pd.to_numeric(g["weight"],errors="coerce").fillna(0).clip(lower=0).to_numpy(np.float32),
                              g["is_va"].to_numpy())
    print(f"{len(assets_data)} assets, seq lengths: {[len(v[0]) for v in assets_data.values()]}",flush=True)
    # 转 GPU tensor per asset
    G={a:(torch.from_numpy(x).to(DEV),torch.from_numpy(y).to(DEV),torch.from_numpy(w).to(DEV),torch.from_numpy(v).to(DEV)) for a,(x,y,w,v) in assets_data.items()}

    torch.manual_seed(2026); m=TCN(len(sel)).to(DEV)
    opt=torch.optim.Adam(m.parameters(),lr=1e-3,weight_decay=1e-5)
    assets=list(G.keys()); best=-9
    for ep in range(15):
        m.train(); t0=time.time(); tot=0; cnt=0
        for a in assets:
            x,y,w,_=G[a]
            x4=x.unsqueeze(0).transpose(1,2)  # [1, n_feat, T]
            opt.zero_grad()
            p=m(x4,torch.tensor(a,device=DEV))
            loss=(w*(p-y)**2).mean(); loss.backward(); opt.step()
            tot+=loss.item()*len(y); cnt+=len(y)
        m.eval()
        ys=[];ps=[];ws=[]
        with torch.no_grad():
            for a in assets:
                x,y,w,v=G[a]; x4=x.unsqueeze(0).transpose(1,2)
                p=m(x4,torch.tensor(a,device=DEV))
                mask=v.bool()
                if mask.any(): ys.append(y[mask].cpu().numpy()); ps.append(p[mask].cpu().numpy()); ws.append(w[mask].cpu().numpy())
        r2=wr2(np.concatenate(ys),np.concatenate(ps),np.concatenate(ws))
        print(f"ep{ep} loss={tot/cnt:.6f} holdout={r2:+.5f} ({time.time()-t0:.0f}s) best={max(best,r2):+.5f}",flush=True)
        best=max(best,r2)
    print(f"\nBEST holdout={best:+.5f} (逐行LGBM 0.00170, NN 0.0011)",flush=True)


if __name__=="__main__":
    main()
