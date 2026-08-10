"""集成权重优化：5 模型(LGBM/CB/XGB/NN1/NN2) 在干净 holdout 上优化混合权重。

train 减末 15%(holdout) 训练各模型 → 预测 holdout(干净OOF) → 非负权重优化最大化加权R² → 套用到 test。
"""
from __future__ import annotations

import sys, time, json
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb, catboost as cb, xgboost as xgb
import torch, torch.nn as nn
from scipy.optimize import minimize

STRAT = "/mnt/iscsi/hd/xxz/public_release_lightgbm_baseline/examples/lightgbm_baseline"
if STRAT not in sys.path: sys.path.insert(0, STRAT)
from data_utils import manifest_files, feature_columns_from_path  # noqa: E402

DATA_ROOT = Path("/mnt/iscsi/hd/xxz/public_release_20260630/data")
DEV = "cuda:0"; N_ASSET = 15
TUNED = dict(objective="regression", metric="None", learning_rate=0.010326965981106629,
             num_leaves=79, min_data_in_leaf=556, feature_fraction=0.5974233229491067,
             bagging_fraction=0.9445362670741704, bagging_freq=1, lambda_l1=0.03778653953330111,
             lambda_l2=2.9757802078489703, max_bin=127, verbosity=-1, num_threads=32, seed=2026,
             bagging_seed=2026, feature_fraction_seed=2026, data_random_seed=2026)


def wr2(y, p, w):
    d = float(np.sum(w * y * y)); return 0.0 if d <= 0 else 1 - float(np.sum(w * (y - p) ** 2) / d)


class MLP(nn.Module):
    def __init__(self, n_feat, emb_dim=8, hidden=(512,256,128), dropout=0.3, in_drop=0.0):
        super().__init__(); self.emb = nn.Embedding(N_ASSET, emb_dim); self.in_drop = in_drop
        layers=[]; d=n_feat+emb_dim
        for h in hidden: layers += [nn.Linear(d,h), nn.GELU(), nn.BatchNorm1d(h), nn.Dropout(dropout)]; d=h
        layers += [nn.Linear(d,1)]; self.net = nn.Sequential(*layers)
    def forward(self, x, a):
        if self.training and self.in_drop>0:
            x = x*(torch.rand(x.shape[0],x.shape[1],device=x.device)>self.in_drop).float()/(1-self.in_drop)
        return self.net(torch.cat([x,self.emb(a)],1)).squeeze(-1)


def main():
    paths = manifest_files(DATA_ROOT, "train")
    feats = feature_columns_from_path(paths[0])
    pf = pd.read_parquet(paths, columns=["time_id","asset_id","weight","target"]+feats)
    pf[feats] = np.nan_to_num(pf[feats].to_numpy(np.float32))
    print(f"loaded {len(pf):,}", flush=True)
    times = np.sort(pf["time_id"].unique())
    ho = set(times[-max(1,int(len(times)*0.15)):].tolist())
    is_va = pf["time_id"].isin(ho).to_numpy()
    tr_df, va_df = pf[~is_va].reset_index(drop=True), pf[is_va].reset_index(drop=True)
    print(f"train {len(tr_df):,} / holdout {len(va_df):,}", flush=True)
    yv = pd.to_numeric(va_df["target"],errors="coerce").fillna(0).to_numpy(np.float64)
    wv = pd.to_numeric(va_df["weight"],errors="coerce").fillna(0).clip(lower=0).to_numpy(np.float64)

    mean = tr_df[feats].to_numpy(np.float32).mean(0); std = tr_df[feats].to_numpy(np.float32).std(0)+1e-6
    def feat_std(df): return np.nan_to_num((df[feats].to_numpy(np.float32)-mean)/std, nan=0,posinf=0,neginf=0).astype(np.float32)
    Xtr_raw = tr_df[feats].to_numpy(np.float32); Xva_raw = va_df[feats].to_numpy(np.float32)
    atr = tr_df["asset_id"].to_numpy(np.float32); ava = va_df["asset_id"].to_numpy(np.float32)
    ytr = pd.to_numeric(tr_df["target"],errors="coerce").fillna(0).to_numpy(np.float32)
    wtr = pd.to_numeric(tr_df["weight"],errors="coerce").fillna(0).clip(lower=0).to_numpy(np.float32)

    oofs = {}
    # LGBM (CPU)
    print("train LGBM...", flush=True); t0=time.time()
    Xtr_lgb = np.column_stack([atr, Xtr_raw]); Xva_lgb = np.column_stack([ava, Xva_raw])
    dtr = lgb.Dataset(Xtr_lgb, label=ytr, weight=wtr, categorical_feature=[0], free_raw_data=False)
    ml = lgb.train(TUNED, dtr, num_boost_round=448)
    oofs["lgb"] = ml.predict(Xva_lgb)
    print(f"  LGBM R²={wr2(yv,oofs['lgb'],wv):+.5f} ({time.time()-t0:.0f}s)", flush=True)
    # CatBoost (GPU)
    print("train CatBoost...", flush=True); t0=time.time()
    dftr = tr_df[["asset_id"]+feats].copy(); dftr["asset_id"]=dftr["asset_id"].astype(np.int32)
    dfva = va_df[["asset_id"]+feats].copy(); dfva["asset_id"]=dfva["asset_id"].astype(np.int32)
    cbp = dict(loss_function="RMSE",learning_rate=0.05,depth=8,l2_leaf_reg=3.0,iterations=161,random_seed=2026,task_type="GPU",devices="0",verbose=False)
    mc = cb.train(cb.Pool(dftr,label=ytr,weight=wtr,cat_features=["asset_id"]), cbp)
    oofs["cb"] = mc.predict(dfva)
    print(f"  CB R²={wr2(yv,oofs['cb'],wv):+.5f} ({time.time()-t0:.0f}s)", flush=True)
    # XGBoost (GPU)
    print("train XGBoost...", flush=True); t0=time.time()
    fcols=["asset_id"]+feats
    dxtr=xgb.DMatrix(tr_df[fcols].astype(np.float32).to_numpy(),label=ytr,weight=wtr,feature_names=fcols)
    dxva=xgb.DMatrix(va_df[fcols].astype(np.float32).to_numpy(),feature_names=fcols)
    mx=xgb.train(dict(tree_method="hist",device="cuda",objective="reg:squarederror",learning_rate=0.03,max_depth=8,min_child_weight=5,subsample=0.8,colsample_bytree=0.8,reg_lambda=3.0,seed=2026,verbosity=0),dxtr,num_boost_round=91)
    oofs["xgb"]=mx.predict(dxva)
    print(f"  XGB R²={wr2(yv,oofs['xgb'],wv):+.5f} ({time.time()-t0:.0f}s)", flush=True)
    # NN1, NN2 (GPU)
    Xtr_s=torch.from_numpy(feat_std(tr_df)).to(DEV); Xva_s=torch.from_numpy(feat_std(va_df)).to(DEV)
    Atr=torch.from_numpy(tr_df["asset_id"].to_numpy(np.int64)).to(DEV); Ava=torch.from_numpy(va_df["asset_id"].to_numpy(np.int64)).to(DEV)
    Ytr=torch.from_numpy(ytr).to(DEV); Wtr=torch.from_numpy(wtr).to(DEV)
    bs=16384; n_tr=len(Xtr_s)
    for name,in_drop,seeds,kw in [("nn1",0.0,list(range(2026,2036)),{}),("nn2",0.5,[2027],{}),("nn3_emb32",0.0,[2028],{"emb_dim":32}),("nn4_deep4",0.0,[2029],{"hidden":(256,256,256,256)})]:
        print(f"train {name} ({len(seeds)} seeds)...", flush=True); t0=time.time()
        acc=[]
        for sd in seeds:
            torch.manual_seed(sd); m=MLP(len(feats),in_drop=in_drop,**kw).to(DEV)
            opt=torch.optim.Adam(m.parameters(),lr=1e-3,weight_decay=1e-5); m.train(); perm=torch.randperm(n_tr,device=DEV)
            for i in range(0,n_tr,bs):
                idx=perm[i:i+bs]; opt.zero_grad(); loss=(Wtr[idx]*(m(Xtr_s[idx],Atr[idx])-Ytr[idx])**2).mean(); loss.backward(); opt.step()
            m.eval()
            with torch.no_grad():
                ps=[]
                for i in range(0,len(Xva_s),16384): ps.append(m(Xva_s[i:i+16384],Ava[i:i+16384]).cpu().numpy())
                acc.append(np.concatenate(ps))
        oofs[name]=np.mean(acc,axis=0)
        print(f"  {name} R²={wr2(yv,oofs[name],wv):+.5f} ({time.time()-t0:.0f}s)", flush=True)

    keys=list(oofs); P=np.array([oofs[k] for k in keys])  # [n_models, n_holdout]
    print("\n相关性:", {k:round(float(np.corrcoef(oofs[keys[0]],oofs[k])[0,1]),2) for k in keys}, flush=True)
    # 非负权重优化(归一化约束)
    def neg_wr2(w):
        return -wr2(yv, w@P, wv)
    cons=({"type":"eq","fun":lambda w:w.sum()-1})
    bnds=[(0,1)]*len(keys)
    best=None
    for _ in range(40):
        w0=np.random.dirichlet(np.ones(len(keys)))
        r=minimize(neg_wr2,w0,method="SLSQP",bounds=bnds,constraints=cons)
        if best is None or r.fun<best.fun: best=r
    w=best.x; w=np.maximum(w,0); w=w/w.sum()
    print("\n最优权重:", {k:round(float(wi),3) for k,wi in zip(keys,w)}, flush=True)
    print(f"集成 holdout R² = {wr2(yv, w@P, wv):+.5f}", flush=True)
    for k in keys: print(f"  单{k}: {wr2(yv,oofs[k],wv):+.5f}", flush=True)

    # 套用到 test 预测(已存的 submissions)
    S=Path("/mnt/iscsi/hd/xxz/submissions")
    tmap={"lgb":"lgbm_full_submission","cb":"cb_submission","xgb":"xgb_submission","nn1":"nn1_10s","nn2":"nn2_submission","nn3_emb32":"nnvar_emb32","nn4_deep4":"nnvar_deep4"}
    base=pd.read_csv(S/f"{tmap[keys[0]]}.csv").sort_values("row_id").reset_index(drop=True)
    T=np.array([pd.read_csv(S/f"{tmap[k]}.csv").sort_values("row_id")["target"].to_numpy() for k in keys])
    # NN 去均值(它们偏负)，用 lgb 均值对齐
    for i,k in enumerate(keys):
        if k.startswith("nn"): T[i]=T[i]-T[i].mean()+T[0].mean()  # T[0]=lgb
    pred=w@T; pred=np.where(np.isfinite(pred),pred,0.0)
    out=pd.DataFrame({"row_id":base["row_id"],"target":pred})
    out.to_csv(S/"ensemble_opt.csv",index=False)
    print(f"\nwrote ensemble_opt.csv weights={[round(float(x),3) for x in w]} mean={pred.mean():+.4f}", flush=True)
    json.dump({"weights":{k:float(wi) for k,wi in zip(keys,w)},"holdout_r2":float(wr2(yv,w@P,wv))},
              open("/mnt/iscsi/hd/xxz/runs/ensemble_opt_weights.json","w"), indent=2)


if __name__=="__main__":
    main()
