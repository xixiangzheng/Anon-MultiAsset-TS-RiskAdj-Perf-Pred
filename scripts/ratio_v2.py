"""Ratio v2 训练：用更强参数 + ratio 特征，冲击 holdout > 0.002。

1. ratio_cb_tuned: tuned CB 参数 (lr=0.0287, depth=7, l2=4.28, bag=0.65, ~432树) + ratio
2. ratio_nn_emb32: emb32 MLP 10-seed + ratio（更强 NN 架构）
3. ratio_cb_deep: depth=10 + ratio（更深 CB）
"""
from __future__ import annotations
import sys, time, json, pickle, gc
from pathlib import Path
import numpy as np, pandas as pd
import catboost as cb
import torch, torch.nn as nn

STRAT = "/mnt/iscsi/hd/xxz/public_release_lightgbm_baseline/examples/lightgbm_baseline"
if STRAT not in sys.path: sys.path.insert(0, STRAT)
from data_utils import manifest_files, feature_columns_from_path  # noqa: E402
DATA_ROOT = Path("/mnt/iscsi/hd/xxz/public_release_20260630/data")
RUN = Path("/mnt/iscsi/hd/xxz/runs"); SUB = Path("/mnt/iscsi/hd/xxz/submissions")
DEV = "cuda:4"; N_ASSET = 15
GPU_CB = "1"  # CB 用 GPU 1


def wr2(y, p, w):
    d = float(np.sum(w*y*y)); return 0.0 if d<=0 else 1-float(np.sum(w*(y-p)**2)/d)


class MLP(nn.Module):
    def __init__(self, n_feat, emb_dim=32, hidden=(512,256,128), dropout=0.3, in_drop=0.0):
        super().__init__(); self.emb=nn.Embedding(N_ASSET,emb_dim); self.in_drop=in_drop
        layers=[]; d=n_feat+emb_dim
        for h in hidden: layers+=[nn.Linear(d,h),nn.GELU(),nn.BatchNorm1d(h),nn.Dropout(dropout)]; d=h
        layers+=[nn.Linear(d,1)]; self.net=nn.Sequential(*layers)
    def forward(self,x,a):
        if self.training and self.in_drop>0:
            x=x*(torch.rand(x.shape[0],x.shape[1],device=x.device)>self.in_drop).float()/(1-self.in_drop)
        return self.net(torch.cat([x,self.emb(a)],1)).squeeze(-1)


def add_ratio_cols(F, ratios):
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
    print(f"loaded {len(ratios)} ratios", flush=True)

    pf = pd.read_parquet(paths, columns=["time_id","asset_id","weight","target"]+feats)
    pf[feats] = np.nan_to_num(pf[feats].to_numpy(np.float32))
    F_all = pf[feats].to_numpy(np.float32)
    R_all = add_ratio_cols(F_all, ratios)
    te = pd.read_parquet(manifest_files(DATA_ROOT,"test"),columns=["row_id","asset_id"]+feats)
    te[feats] = np.nan_to_num(te[feats].to_numpy(np.float32)); te = te.sort_values("row_id").reset_index(drop=True)
    F_te = te[feats].to_numpy(np.float32); R_te = add_ratio_cols(F_te, ratios)

    times = np.sort(pf["time_id"].unique()); ho = set(times[-max(1,int(len(times)*0.15)):].tolist())
    is_va = pf["time_id"].isin(ho).to_numpy()
    y32 = pd.to_numeric(pf["target"],errors="coerce").fillna(0).to_numpy(np.float32)
    w32 = pd.to_numeric(pf["weight"],errors="coerce").fillna(0).clip(lower=0).to_numpy(np.float32)
    yv = y32[is_va].astype(np.float64); wv = w32[is_va].astype(np.float64)
    ratio_cols = [f"r{i}" for i in range(R_all.shape[1])]
    print(f"data: train {len(pf):,} holdout {is_va.sum():,}", flush=True)

    # 准备共享 pkl 续跑
    oof_pkl = RUN/"ratio_v2_partial.pkl"
    if oof_pkl.exists():
        prev = pickle.load(open(oof_pkl,"rb")); oofs = prev.get("oofs",{}); te_preds = prev.get("te_preds",{})
        print(f"[resume] {list(oofs.keys())}", flush=True)
    else:
        oofs = {}; te_preds = {}
    def save_partial():
        pickle.dump({"oofs":oofs,"te_preds":te_preds,"yv":yv,"wv":wv,
                     "row_id":te["row_id"].to_numpy()}, open(oof_pkl,"wb"))

    pdf = pf[["asset_id"]+feats].copy(); pdf[ratio_cols] = R_all; pdf["asset_id"] = pdf["asset_id"].astype(np.int32)
    tedf = te[["asset_id"]+feats].copy(); tedf[ratio_cols] = R_te; tedf["asset_id"] = tedf["asset_id"].astype(np.int32)

    # === 1. ratio_cb_tuned（tuned CB 参数 + ratio）===
    if "ratio_cb_tuned" not in oofs:
        print("\n=== ratio_cb_tuned (lr=0.0287, depth=7, l2=4.28, bag=0.65) + ratio ===", flush=True); t0=time.time()
        tr_df = pdf[~is_va].reset_index(drop=True); va_df = pdf[is_va].reset_index(drop=True)
        trpool = cb.Pool(tr_df, label=y32[~is_va], weight=w32[~is_va], cat_features=["asset_id"])
        vapool = cb.Pool(va_df, label=yv.astype(np.float32), weight=wv.astype(np.float32), cat_features=["asset_id"])
        cbp = dict(loss_function="RMSE", learning_rate=0.0287, depth=7, l2_leaf_reg=4.28,
                   bagging_temperature=0.65, iterations=1200, random_seed=2026,
                   task_type="GPU", devices=GPU_CB, verbose=False,
                   early_stopping_rounds=50, use_best_model=True)
        mc = cb.train(trpool, cbp, eval_set=vapool, verbose=False)
        bi = mc.tree_count_
        oofs["ratio_cb_tuned"] = mc.predict(vapool)
        fullpool = cb.Pool(pdf, label=y32, weight=w32, cat_features=["asset_id"])
        preds = []
        for s in [2026,2027,2028]:
            p = dict(cbp); p["random_seed"] = s; p["iterations"] = bi; p["use_best_model"] = False
            mm = cb.train(fullpool, p, verbose=False); preds.append(mm.predict(tedf))
        te_preds["ratio_cb_tuned"] = np.mean(preds, 0)
        print(f"  holdout R²={wr2(yv,oofs['ratio_cb_tuned'],wv):+.5f} trees={bi} ({time.time()-t0:.0f}s)", flush=True)
        save_partial(); del trpool,vapool,fullpool,mc; gc.collect()
    else: print("[skip] ratio_cb_tuned", flush=True)

    # === 2. ratio_cb_deep（depth=10 + ratio）===
    if "ratio_cb_deep" not in oofs:
        print("\n=== ratio_cb_deep (depth=10) + ratio ===", flush=True); t0=time.time()
        tr_df = pdf[~is_va].reset_index(drop=True); va_df = pdf[is_va].reset_index(drop=True)
        trpool = cb.Pool(tr_df, label=y32[~is_va], weight=w32[~is_va], cat_features=["asset_id"])
        vapool = cb.Pool(va_df, label=yv.astype(np.float32), weight=wv.astype(np.float32), cat_features=["asset_id"])
        cbp = dict(loss_function="RMSE", learning_rate=0.05, depth=10, l2_leaf_reg=3.0,
                   iterations=800, random_seed=2026, task_type="GPU", devices=GPU_CB,
                   verbose=False, early_stopping_rounds=50, use_best_model=True)
        mc = cb.train(trpool, cbp, eval_set=vapool, verbose=False)
        bi = mc.tree_count_
        oofs["ratio_cb_deep"] = mc.predict(vapool)
        fullpool = cb.Pool(pdf, label=y32, weight=w32, cat_features=["asset_id"])
        preds = []
        for s in [2026,2027,2028]:
            p = dict(cbp); p["random_seed"] = s; p["iterations"] = bi; p["use_best_model"] = False
            mm = cb.train(fullpool, p, verbose=False); preds.append(mm.predict(tedf))
        te_preds["ratio_cb_deep"] = np.mean(preds, 0)
        print(f"  holdout R²={wr2(yv,oofs['ratio_cb_deep'],wv):+.5f} trees={bi} ({time.time()-t0:.0f}s)", flush=True)
        save_partial(); del trpool,vapool,fullpool,mc; gc.collect()
    else: print("[skip] ratio_cb_deep", flush=True)

    # === 3. ratio_nn_emb32（emb32 MLP + ratio + 10 seed）===
    if "ratio_nn_emb32" not in oofs:
        print("\n=== ratio_nn_emb32 (emb32 MLP 10-seed + ratio) ===", flush=True); t0=time.time()
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
        bs = 16384; n_tr = len(Xtr_s); n_feat = len(feats)+len(ratio_cols)
        seeds = list(range(2026,2036)); acc_va = []; acc_te = []
        for sd in seeds:
            torch.manual_seed(sd); m = MLP(n_feat, emb_dim=32).to(DEV)
            opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5)
            m.train(); perm = torch.randperm(n_tr, device=DEV)
            for i in range(0, n_tr, bs):
                idx = perm[i:i+bs]; opt.zero_grad()
                loss = (Wtr[idx]*(m(Xtr_s[idx],Atr[idx])-Ytr[idx])**2).mean(); loss.backward(); opt.step()
            m.eval()
            with torch.no_grad():
                pv = [m(Xva_s[i:i+16384],Ava[i:i+16384]).cpu().numpy() for i in range(0,len(Xva_s),16384)]
                pt = [m(Xte_s[i:i+16384],Ate[i:i+16384]).cpu().numpy() for i in range(0,len(Xte_s),16384)]
                acc_va.append(np.concatenate(pv)); acc_te.append(np.concatenate(pt))
            print(f"    seed {sd} done", flush=True)
        oofs["ratio_nn_emb32"] = np.mean(acc_va,0); te_preds["ratio_nn_emb32"] = np.mean(acc_te,0)
        print(f"  holdout R²={wr2(yv,oofs['ratio_nn_emb32'],wv):+.5f} ({time.time()-t0:.0f}s)", flush=True)
        save_partial()
        del Xtr_s,Xva_s,Xte_s,Atr,Ava,Ate,Ytr,Wtr; gc.collect(); torch.cuda.empty_cache()
    else: print("[skip] ratio_nn_emb32", flush=True)

    # 写 submissions + 最终 pkl
    for k,p in te_preds.items():
        p = np.where(np.isfinite(p), p, 0.0)
        pd.DataFrame({"row_id":te["row_id"],"target":p}).to_csv(SUB/f"v2_{k}.csv", index=False)
    pickle.dump({"oofs":oofs,"te_preds":te_preds,"yv":yv,"wv":wv,"row_id":te["row_id"].to_numpy()},
                open(RUN/"ratio_v2_oof.pkl","wb"))
    print(f"\n[done] {len(oofs)} v2 models saved.", flush=True)


if __name__ == "__main__":
    main()
