"""FT-Transformer 正经版：全323特征 + 自注意力 + proper训练。

突破之前小MLP的弱表现：
- 全部323特征tokenize → 323+CLS+asset=325 tokens → 3层自注意力
- pre-LN(norm_first)更稳, warmup+cosine LR, 梯度裁剪, 15 epoch early-stop
- PyTorch SDPA(FlashAttention)加速
holdout 上验证能否匹配/突破 GBDT 0.0017。
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


class FTTransformer(nn.Module):
    def __init__(self, n_feat, d=96, nhead=8, nlayer=3, ff=384, dropout=0.2):
        super().__init__()
        self.feat_proj = nn.Linear(1, d)  # 每个标量特征 → d维 (共享投影, 简单高效)
        self.cls = nn.Parameter(torch.randn(1,1,d)*0.02)
        self.asset_emb = nn.Embedding(N_ASSET, d)
        n_tok = n_feat + 2  # features + cls + asset
        self.pos = nn.Parameter(torch.randn(1, n_tok, d)*0.02)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=nhead, dim_feedforward=ff,
                                           dropout=dropout, batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=nlayer)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d,d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d,1))

    def forward(self, x, asset):  # x [B, n_feat], asset [B]
        B = x.shape[0]
        tok = self.feat_proj(x.unsqueeze(-1))  # [B, n_feat, d]  每标量→d维
        cls = self.cls.expand(B,-1,-1)
        ast = self.asset_emb(asset).unsqueeze(1)
        h = torch.cat([cls, ast, tok], 1)  # [B, n_feat+2, d]
        h = h + self.pos
        h = self.encoder(h)
        return self.head(h[:,0]).squeeze(-1)


def main():
    paths=manifest_files(DATA_ROOT,"train")[:2]; feats=feature_columns_from_path(paths[0])
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
    m=FTTransformer(len(feats)).to(DEV)
    print(f"params: {sum(p.numel() for p in m.parameters()):,}", flush=True)
    bs=8192; n_tr=len(Xt); epochs=15; warmup=2; lr0=5e-4
    opt=torch.optim.AdamW(m.parameters(),lr=lr0,weight_decay=1e-4)
    def lr_at(ep):
        if ep<warmup: return lr0*(ep+1)/warmup
        return lr0*0.5*(1+math.cos(math.pi*(ep-warmup)/(epochs-warmup)))
    best=-9; bad=0
    for ep in range(epochs):
        lr=lr_at(ep)
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
            for j in range(0,len(Xv),8192): ps.append(m(Xv[j:j+8192],Av[j:j+8192]).cpu().numpy())
            r2=wr2(yv,np.concatenate(ps),wv)
        print(f"ep{ep} lr={lr:.6f} loss={tot/cnt:.6f} holdout={r2:+.5f} ({time.time()-t0:.0f}s) best={max(best,r2):+.5f}",flush=True)
        if r2>best: best=r2; bad=0
        else:
            bad+=1
            if bad>=5: print("early stop"); break
    print(f"\nBEST holdout={best:+.5f} (GBDT 0.00170, 之前FT-Transformer top50 0.00148)",flush=True)


if __name__=="__main__":
    main()
