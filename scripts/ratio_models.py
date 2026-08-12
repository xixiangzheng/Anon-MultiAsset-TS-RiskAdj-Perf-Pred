"""Ratio 特征 + 多模型训练：用大规模 ratio 扫描找 top-50 交互，喂给 LGBM/CB/XGB/NN 4 个新模型。

目的：在 ratio_lgbm(+0.00021) 突破基础上，把 ratio 推广到 CB/XGB/NN（更多样），扩大集成上限。
- Step1: 搜索 top-5 分母 + top-50 ratio 交互（用前2分区数据，省时）
- Step2: 在 holdout(末15%) 上训练各模型 → 输出 holdout R²
- Step3: 全量训练 → test 预测 → submissions/ratio_{lgb,cb,xgb,nn}.csv
- Step4: 与现有 cb/nn 一起做集成权重优化 → submissions/ens_ratio4.csv
"""
from __future__ import annotations
import sys, time, json, pickle, gc
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb, catboost as cb, xgboost as xgb
import torch, torch.nn as nn
from scipy.optimize import minimize

STRAT = "/mnt/iscsi/hd/xxz/public_release_lightgbm_baseline/examples/lightgbm_baseline"
if STRAT not in sys.path: sys.path.insert(0, STRAT)
from data_utils import manifest_files, feature_columns_from_path  # noqa: E402
DATA_ROOT = Path("/mnt/iscsi/hd/xxz/public_release_20260630/data")
DEV = "cuda:4"; N_ASSET = 15  # GPU 4
GPU_CB = "1"   # GPU 1 for CatBoost
GPU_XGB = "2"  # GPU 2 for XGBoost
SUB = Path("/mnt/iscsi/hd/xxz/submissions"); RUN = Path("/mnt/iscsi/hd/xxz/runs")

TUNED = dict(objective="regression", metric="None", learning_rate=0.010326965981106629,
             num_leaves=79, min_data_in_leaf=556, feature_fraction=0.5974233229491067,
             bagging_fraction=0.9445362670741704, bagging_freq=1, lambda_l1=0.03778653953330111,
             lambda_l2=2.9757802078489703, max_bin=127, verbosity=-1, num_threads=48, seed=2026,
             bagging_seed=2026, feature_fraction_seed=2026, data_random_seed=2026)


def wr2(y, p, w):
    d = float(np.sum(w*y*y)); return 0.0 if d<=0 else 1-float(np.sum(w*(y-p)**2)/d)


def feval_wr2(preds, ds):
    y = ds.get_label(); w = ds.get_weight()
    if w is None: w = np.ones_like(y)
    return ("wr2", float(wr2(y, preds, w)), True)


def wcorr(x,y,w):
    wm=(w*x).sum()/w.sum(); ym=(w*y).sum()/w.sum()
    num=(w*(x-wm)*(y-ym)).sum(); den=np.sqrt((w*(x-wm)**2).sum()*(w*(y-ym)**2).sum())
    return abs(num/den) if den>0 else 0


class MLP(nn.Module):
    def __init__(self, n_feat, emb_dim=8, hidden=(512,256,128), dropout=0.3, in_drop=0.0):
        super().__init__(); self.emb=nn.Embedding(N_ASSET,emb_dim); self.in_drop=in_drop
        layers=[]; d=n_feat+emb_dim
        for h in hidden: layers+=[nn.Linear(d,h),nn.GELU(),nn.BatchNorm1d(h),nn.Dropout(dropout)]; d=h
        layers+=[nn.Linear(d,1)]; self.net=nn.Sequential(*layers)
    def forward(self,x,a):
        if self.training and self.in_drop>0:
            x=x*(torch.rand(x.shape[0],x.shape[1],device=x.device)>self.in_drop).float()/(1-self.in_drop)
        return self.net(torch.cat([x,self.emb(a)],1)).squeeze(-1)


def search_ratios(F, y, w, feats):
    """搜索 top-5 分母 → top-50 ratio 交互。返回 [(numer, denom), ...]"""
    sc=np.array([wcorr(F[:,i],y,w) for i in range(len(feats))])
    top50_idx=np.argsort(-sc)[:50]
    print(f"[search] top-5 single: {[(feats[i],round(float(sc[i]),4)) for i in np.argsort(-sc)[:5]]}", flush=True)
    print("[search] scanning 323 denominators...", flush=True); t0=time.time()
    denom_scores=[]
    for d in range(len(feats)):
        fd=F[:,d]
        if fd.min()<0: continue
        fd_s=np.clip(fd,1e-8,np.percentile(fd,99))
        imps=[]
        for ni in top50_idx[:10]:
            r=F[:,ni]/fd_s
            r=np.clip(r,np.percentile(r,1),np.percentile(r,99))
            r=np.nan_to_num(r,nan=0,posinf=0,neginf=0)
            imps.append(wcorr(r,y,w)-sc[ni])
        denom_scores.append((d,float(np.mean(imps))))
    denom_scores.sort(key=lambda x:-x[1])
    best_d=[d for d,_ in denom_scores[:5]]
    print(f"[search] done {time.time()-t0:.0f}s. Top-5 分母: {[(feats[d],round(s,5)) for d,s in denom_scores[:5]]}", flush=True)
    interactions=[]
    for d in best_d:
        fd=np.clip(F[:,d],1e-8,np.percentile(F[:,d],99))
        for ni in top50_idx:
            r=F[:,ni]/fd
            r=np.clip(r,np.percentile(r,1),np.percentile(r,99))
            r=np.nan_to_num(r,nan=0,posinf=0,neginf=0)
            interactions.append((ni,d,wcorr(r,y,w)))
    interactions.sort(key=lambda x:-x[2])
    top_inter=interactions[:50]
    print(f"[search] top-10 ratio:", flush=True)
    for ni,di,c in top_inter[:10]: print(f"  {c:.4f}  {feats[ni]}/{feats[di]}", flush=True)
    return [(ni,di) for ni,di,_ in top_inter]


def add_ratio_cols(F, ratios):
    """F: ndarray [N, n_feat]; ratios: [(numer_idx, denom_idx), ...] → ndarray [N, K]"""
    cols=[]
    for ni,di in ratios:
        fd=np.clip(F[:,di],1e-8,np.percentile(F[:,di],99))
        r=F[:,ni]/fd
        r=np.clip(r,np.percentile(r,1),np.percentile(r,99))
        r=np.nan_to_num(r,nan=0,posinf=0,neginf=0)
        cols.append(r.astype(np.float32))
    return np.column_stack(cols)


def main():
    paths=manifest_files(DATA_ROOT,"train"); feats=feature_columns_from_path(paths[0])
    if (RUN/"ratio_top50.json").exists():
        rs = json.loads((RUN/"ratio_top50.json").read_text())["ratios"]
        ratios = [(feats.index(r[2]), feats.index(r[3])) for r in rs]
        print(f"[search] loaded {len(ratios)} ratios from cache (skip search)", flush=True)
    else:
        # 先用前2分区搜索 ratio
        pf_s=pd.read_parquet(paths[:2],columns=["weight","target"]+feats)
        pf_s[feats]=np.nan_to_num(pf_s[feats].to_numpy(np.float32))
        ys=pd.to_numeric(pf_s["target"],errors="coerce").fillna(0).to_numpy(np.float64)
        ws=pd.to_numeric(pf_s["weight"],errors="coerce").fillna(0).clip(lower=0).to_numpy(np.float64)
        Fs=pf_s[feats].to_numpy(np.float64)
        print(f"[search] {len(pf_s):,} rows for ratio search", flush=True)
        ratios=search_ratios(Fs,ys,ws,feats)
        del Fs,ys,ws,pf_s; gc.collect()
        json.dump({"ratios":[[int(ni),int(di),feats[ni],feats[di]] for ni,di in ratios]},
                  open(RUN/"ratio_top50.json","w"),indent=2)
        print(f"[search] saved {len(ratios)} ratios to runs/ratio_top50.json", flush=True)

    # 全量加载 + 加 ratio
    pf=pd.read_parquet(paths,columns=["time_id","asset_id","weight","target"]+feats)
    pf[feats]=np.nan_to_num(pf[feats].to_numpy(np.float32))
    F_all=pf[feats].to_numpy(np.float32)
    R_all=add_ratio_cols(F_all, ratios)  # [N, 50]
    print(f"[data] train {len(pf):,} + {R_all.shape[1]} ratio cols", flush=True)

    te=pd.read_parquet(manifest_files(DATA_ROOT,"test"),columns=["row_id","asset_id"]+feats)
    te[feats]=np.nan_to_num(te[feats].to_numpy(np.float32)); te=te.sort_values("row_id").reset_index(drop=True)
    F_te=te[feats].to_numpy(np.float32); R_te=add_ratio_cols(F_te, ratios)

    times=np.sort(pf["time_id"].unique()); ho=set(times[-max(1,int(len(times)*0.15)):].tolist())
    is_va=pf["time_id"].isin(ho).to_numpy()
    y32=pd.to_numeric(pf["target"],errors="coerce").fillna(0).to_numpy(np.float32)
    w32=pd.to_numeric(pf["weight"],errors="coerce").fillna(0).clip(lower=0).to_numpy(np.float32)
    a32=pf["asset_id"].to_numpy(np.float32)
    yv=y32[is_va].astype(np.float64); wv=w32[is_va].astype(np.float64)
    ava=a32[is_va]
    n=len(pf); n_te=len(te)
    ratio_cols=[f"r{i}" for i in range(R_all.shape[1])]
    all_cols=feats+ratio_cols
    print(f"[data] {len(all_cols)} total features ({len(feats)} raw + {len(ratio_cols)} ratio)", flush=True)

    oofs={}; te_preds={}
    # 增量加载已存模型（断点续跑）
    oof_pkl = RUN/"ratio4_oof_partial.pkl"
    if oof_pkl.exists():
        prev = pickle.load(open(oof_pkl, "rb"))
        oofs = prev.get("oofs", {}); te_preds = prev.get("te_preds", {})
        print(f"[resume] loaded {len(oofs)} prev models: {list(oofs.keys())}", flush=True)
    def save_partial():
        pickle.dump({"oofs":oofs,"te_preds":te_preds,"yv":yv,"wv":wv,
                     "row_id":te["row_id"].to_numpy(),"tids_holdout":pf.loc[is_va,"time_id"].to_numpy()},
                    open(oof_pkl,"wb"))

    # --- ratio_lgb (tuned 参数 + ratio) ---
    if "ratio_lgb" not in oofs:
        print("\n=== ratio_lgb (tuned + ratio) ===", flush=True); t0=time.time()
        Xtr=np.column_stack([a32, F_all, R_all]); Xte=np.column_stack([te["asset_id"].to_numpy(np.float32), F_te, R_te])
        Xva=Xtr[is_va]; Xtr=Xtr[~is_va]; ytr=y32[~is_va]; wtr=w32[~is_va]
        dtr=lgb.Dataset(Xtr,label=ytr,weight=wtr,categorical_feature=[0],free_raw_data=False)
        dva=lgb.Dataset(Xva,label=yv.astype(np.float32),weight=wv.astype(np.float32),reference=dtr,free_raw_data=False)
        m=lgb.train(TUNED,dtr,num_boost_round=1500,valid_sets=[dva],valid_names=["va"],feval=feval_wr2,
                    callbacks=[lgb.early_stopping(60,verbose=False),lgb.log_evaluation(0)])
        bi=m.best_iteration or 1500
        oofs["ratio_lgb"]=m.predict(Xva)
        # 全量 3 seed（data_random_seed 不可改，但 seed/bagging/feat 可改 → 已有多样性）
        Xf=np.concatenate([Xtr,Xva]); yf=np.concatenate([ytr,yv.astype(np.float32)]); wf=np.concatenate([wtr,wv.astype(np.float32)])
        dtr_full=lgb.Dataset(Xf,label=yf,weight=wf,categorical_feature=[0],free_raw_data=False)
        preds=[]
        for s in [2026,2027,2028]:
            p=dict(TUNED); p.update(seed=s,bagging_seed=s,feature_fraction_seed=s)
            mm=lgb.train(p,dtr_full,num_boost_round=bi,feval=feval_wr2); preds.append(mm.predict(Xte))
        te_preds["ratio_lgb"]=np.mean(preds,0)
        print(f"  holdout R²={wr2(yv,oofs['ratio_lgb'],wv):+.5f} iter={bi} ({time.time()-t0:.0f}s)", flush=True)
        save_partial()
        del dtr,dva,dtr_full,m; gc.collect()
    else: print("[skip] ratio_lgb already done", flush=True)

    # --- ratio_cb (GPU) ---
    if "ratio_cb" not in oofs:
        print("\n=== ratio_cb (CatBoost + ratio) ===", flush=True); t0=time.time()
        cols_df=["asset_id"]+feats+ratio_cols
        pdf=pf[["asset_id"]+feats].copy(); pdf[ratio_cols]=R_all
        pdf["asset_id"]=pdf["asset_id"].astype(np.int32)
        tedf=te[["asset_id"]+feats].copy(); tedf[ratio_cols]=R_te
        tedf["asset_id"]=tedf["asset_id"].astype(np.int32)
        tr_df=pdf[~is_va].reset_index(drop=True); va_df=pdf[is_va].reset_index(drop=True)
        trpool=cb.Pool(tr_df,label=y32[~is_va],weight=w32[~is_va],cat_features=["asset_id"])
        vapool=cb.Pool(va_df,label=yv.astype(np.float32),weight=wv.astype(np.float32),cat_features=["asset_id"])
        cbp=dict(loss_function="RMSE",learning_rate=0.05,depth=8,l2_leaf_reg=3.0,iterations=800,random_seed=2026,
                 task_type="GPU",devices=GPU_CB,verbose=False,early_stopping_rounds=50,use_best_model=True)
        mc=cb.train(trpool,cbp,eval_set=vapool,verbose=False)
        bi_cb=mc.tree_count_
        oofs["ratio_cb"]=mc.predict(vapool)
        fullpool=cb.Pool(pdf,label=y32,weight=w32,cat_features=["asset_id"])
        preds=[]
        for s in [2026,2027,2028]:
            p=dict(cbp); p["random_seed"]=s; p["iterations"]=bi_cb; p["use_best_model"]=False
            mm=cb.train(fullpool,p,verbose=False); preds.append(mm.predict(tedf))
        te_preds["ratio_cb"]=np.mean(preds,0)
        print(f"  holdout R²={wr2(yv,oofs['ratio_cb'],wv):+.5f} trees={bi_cb} ({time.time()-t0:.0f}s)", flush=True)
        save_partial()
        del trpool,vapool,fullpool,mc,pdf,tr_df,va_df; gc.collect()
    else: print("[skip] ratio_cb already done", flush=True)

    # --- ratio_xgb (GPU) ---
    if "ratio_xgb" not in oofs:
        print("\n=== ratio_xgb (XGBoost + ratio) ===", flush=True); t0=time.time()
        Xxgb_tr=np.column_stack([a32[~is_va], F_all[~is_va], R_all[~is_va]]).astype(np.float32)
        Xxgb_va=np.column_stack([a32[is_va], F_all[is_va], R_all[is_va]]).astype(np.float32)
        Xxgb_te=np.column_stack([te["asset_id"].to_numpy(np.float32), F_te, R_te]).astype(np.float32)
        fcols=["asset_id"]+feats+ratio_cols
        dxtr=xgb.DMatrix(Xxgb_tr,label=y32[~is_va],weight=w32[~is_va],feature_names=fcols)
        dxva=xgb.DMatrix(Xxgb_va,label=y32[is_va],weight=w32[is_va],feature_names=fcols)
        dxte=xgb.DMatrix(Xxgb_te,feature_names=fcols)
        xp=dict(tree_method="hist",device="cuda:"+GPU_XGB,objective="reg:squarederror",learning_rate=0.03,max_depth=8,
                min_child_weight=5,subsample=0.8,colsample_bytree=0.8,reg_lambda=3.0,seed=2026,verbosity=0)
        mx=xgb.train(xp,dxtr,num_boost_round=500,evals=[(dxva,"va")],early_stopping_rounds=30,verbose_eval=False)
        bi_xgb=mx.best_iteration+1
        oofs["ratio_xgb"]=mx.predict(dxva)
        dxfull=xgb.DMatrix(np.concatenate([Xxgb_tr,Xxgb_va]),label=np.concatenate([y32[~is_va],yv.astype(np.float32)]),
                           weight=np.concatenate([w32[~is_va],wv.astype(np.float32)]),feature_names=fcols)
        preds=[]
        for s in [2026,2027,2028]:
            p=dict(xp); p["seed"]=s
            mm=xgb.train(p,dxfull,num_boost_round=bi_xgb); preds.append(mm.predict(dxte))
        te_preds["ratio_xgb"]=np.mean(preds,0)
        print(f"  holdout R²={wr2(yv,oofs['ratio_xgb'],wv):+.5f} iter={bi_xgb} ({time.time()-t0:.0f}s)", flush=True)
        save_partial()
        del dxtr,dxva,dxte,dxfull,mx; gc.collect()
    else: print("[skip] ratio_xgb already done", flush=True)

    # --- ratio_nn (GPU) ---
    if "ratio_nn" not in oofs:
        print("\n=== ratio_nn (MLP 10-seed + ratio) ===", flush=True); t0=time.time()
        raw_mean=F_all[~is_va].mean(0); raw_std=F_all[~is_va].std(0)+1e-6
        raw_mean=np.concatenate([raw_mean, R_all[~is_va].mean(0)])  # +ratio mean
        raw_std=np.concatenate([raw_std, R_all[~is_va].std(0)+1e-6])
        def stdize(F, R):
            X=np.concatenate([F,R],1).astype(np.float32)
            return np.nan_to_num((X-raw_mean)/raw_std,nan=0,posinf=0,neginf=0).astype(np.float32)
        Xtr_s=torch.from_numpy(stdize(F_all[~is_va], R_all[~is_va])).to(DEV)
        Xva_s=torch.from_numpy(stdize(F_all[is_va], R_all[is_va])).to(DEV)
        Xte_s=torch.from_numpy(stdize(F_te, R_te)).to(DEV)
        Atr=torch.from_numpy(pf["asset_id"].to_numpy(np.int64)[~is_va]).to(DEV)
        Ava=torch.from_numpy(pf["asset_id"].to_numpy(np.int64)[is_va]).to(DEV)
        Ate=torch.from_numpy(te["asset_id"].to_numpy(np.int64)).to(DEV)
        Ytr=torch.from_numpy(y32[~is_va]).to(DEV); Wtr=torch.from_numpy(w32[~is_va]).to(DEV)
        bs=16384; n_tr=len(Xtr_s); n_feat=len(feats)+len(ratio_cols)
        seeds=list(range(2026,2036)); acc_va=[]; acc_te=[]
        for sd in seeds:
            torch.manual_seed(sd); m=MLP(n_feat,emb_dim=8).to(DEV)
            opt=torch.optim.Adam(m.parameters(),lr=1e-3,weight_decay=1e-5); m.train(); perm=torch.randperm(n_tr,device=DEV)
            for i in range(0,n_tr,bs):
                idx=perm[i:i+bs]; opt.zero_grad(); loss=(Wtr[idx]*(m(Xtr_s[idx],Atr[idx])-Ytr[idx])**2).mean(); loss.backward(); opt.step()
            m.eval()
            with torch.no_grad():
                pv=[m(Xva_s[i:i+16384],Ava[i:i+16384]).cpu().numpy() for i in range(0,len(Xva_s),16384)]
                pt=[m(Xte_s[i:i+16384],Ate[i:i+16384]).cpu().numpy() for i in range(0,len(Xte_s),16384)]
                acc_va.append(np.concatenate(pv)); acc_te.append(np.concatenate(pt))
            print(f"    seed {sd} done", flush=True)
        oofs["ratio_nn"]=np.mean(acc_va,0); te_preds["ratio_nn"]=np.mean(acc_te,0)
        print(f"  holdout R²={wr2(yv,oofs['ratio_nn'],wv):+.5f} ({time.time()-t0:.0f}s)", flush=True)
        save_partial()
        del Xtr_s,Xva_s,Xte_s,Atr,Ava,Ate,Ytr,Wtr; gc.collect(); torch.cuda.empty_cache()
    else: print("[skip] ratio_nn already done", flush=True)

    # --- 保存最终 OOF ---
    pickle.dump({"keys":list(oofs.keys()),"oofs":oofs,"yv":yv,"wv":wv,"te_preds":te_preds,"row_id":te["row_id"].to_numpy(),
                 "tids_holdout":pf.loc[is_va,"time_id"].to_numpy()},
                open(RUN/"ratio4_oof.pkl","wb"))
    # 写 submissions
    for k,p in te_preds.items():
        p=np.where(np.isfinite(p),p,0.0)
        pd.DataFrame({"row_id":te["row_id"],"target":p}).to_csv(SUB/f"ratio4_{k}.csv",index=False)
    print("\n[done] all ratio4 models trained + saved.", flush=True)


if __name__=="__main__":
    main()
