"""私榜交付重训（版本B）：train(13.2M) + 回补(3.2M) = 16.4M 行。

关键设计：
- 回补段是最新时段（最接近私榜未来数据），必须并入训练
- purged holdout：从回补段尾部取 15%（时序最晚，分布最接近私榜）
- 组件：ratio_lgb(÷157 all) + nn_emb32 + cb（与版本A同配方，数据不同）
- 产出保存到 src/final_submission/model_v2/
"""
from __future__ import annotations
import sys, time, json, pickle, gc
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
import catboost as cb
import torch, torch.nn as nn

STRAT = "/mnt/iscsi/hd/xxz/public_release_lightgbm_baseline/examples/lightgbm_baseline"
if STRAT not in sys.path: sys.path.insert(0, STRAT)
from data_utils import manifest_files, feature_columns_from_path  # noqa: E402
DATA_ROOT = Path("/mnt/iscsi/hd/xxz/public_release_20260630/data")
BACKFILL = Path("/mnt/iscsi/hd/xxz/public_release_20260823/public_release_20260823/data")
MODEL_DIR = Path("/mnt/iscsi/hd/xxz/src/final_submission/model_v2")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DEV = "cuda:4"
N_ASSET = 15
RATIO_DENOM = "feature_157"

TUNED = dict(objective="regression", metric="None", learning_rate=0.010326965981106629,
             num_leaves=79, min_data_in_leaf=556, feature_fraction=0.5974233229491067,
             bagging_fraction=0.9445362670741704, bagging_freq=1, lambda_l1=0.03778653953330111,
             lambda_l2=2.9757802078489703, max_bin=127, verbosity=-1, num_threads=48, seed=2026,
             bagging_seed=2026, feature_fraction_seed=2026, data_random_seed=2026)

CB_PARAMS = dict(loss_function="RMSE", learning_rate=0.05, depth=8, l2_leaf_reg=3.0,
                 random_seed=2026, verbose=False, early_stopping_rounds=50,
                 use_best_model=True, task_type="GPU", devices="1", gpu_ram_part=0.45)


def wr2(y, p, w):
    d = float(np.sum(w*y*y)); return 0.0 if d<=0 else 1-float(np.sum(w*(y-p)**2)/d)


def feval_wr2(preds, ds):
    y = ds.get_label(); w = ds.get_weight()
    if w is None: w = np.ones_like(y)
    return ("wr2", float(wr2(y, preds, w)), True)


class MLP(nn.Module):
    def __init__(self, n_feat, emb_dim=32, hidden=(512,256,128), dropout=0.3):
        super().__init__(); self.emb = nn.Embedding(N_ASSET, emb_dim)
        layers=[]; d=n_feat+emb_dim
        for h in hidden: layers += [nn.Linear(d,h), nn.GELU(), nn.BatchNorm1d(h), nn.Dropout(dropout)]; d=h
        layers += [nn.Linear(d,1)]; self.net = nn.Sequential(*layers)
    def forward(self, x, a):
        return self.net(torch.cat([x, self.emb(a)], 1)).squeeze(-1)


def main():
    paths = manifest_files(DATA_ROOT, "train"); feats = feature_columns_from_path(paths[0])
    denom_idx = feats.index(RATIO_DENOM)
    cols = ["time_id","asset_id","weight","target"]+feats
    pf = pd.read_parquet(paths, columns=cols)
    bf = pd.read_parquet(sorted(BACKFILL.glob("train/*.parquet")), columns=cols)
    pf = pd.concat([pf, bf], ignore_index=True)
    del bf; gc.collect()
    pf[feats] = np.nan_to_num(pf[feats].to_numpy(np.float32))
    F = pf[feats].to_numpy(np.float32)
    y32 = pd.to_numeric(pf["target"],errors="coerce").fillna(0).to_numpy(np.float32)
    w32 = pd.to_numeric(pf["weight"],errors="coerce").fillna(0).clip(lower=0).to_numpy(np.float32)
    a32 = pf["asset_id"].to_numpy(np.float32)
    times = np.sort(pf["time_id"].unique())
    # holdout = 全部时段的最后 15%（含回补段尾部 —— 最接近私榜）
    ho = set(times[-max(1,int(len(times)*0.15)):].tolist())
    is_va = pf["time_id"].isin(ho).to_numpy()
    yv = y32[is_va].astype(np.float64); wv = w32[is_va].astype(np.float64)
    print(f"data: {len(pf):,} rows (train+backfill), holdout {is_va.sum():,} (最新时段)", flush=True)
    print(f"time range: {times[0]}..{times[-1]}, holdout from {times[-max(1,int(len(times)*0.15))]}", flush=True)

    # ratio 特征
    hi_pct = float(np.percentile(F[:, denom_idx], 99))
    fd = np.clip(F[:, denom_idx], 1e-8, hi_pct)
    R = np.nan_to_num(F / fd[:, None], nan=0, posinf=0, neginf=0).astype(np.float32)
    lo = np.percentile(R, 1, axis=0); hi = np.percentile(R, 99, axis=0)
    R = np.clip(R, lo, hi)
    X_full = np.column_stack([a32, F, R])
    print(f"features: 1 + {F.shape[1]} + {R.shape[1]} ratio = {X_full.shape[1]}", flush=True)

    oof_pkl = MODEL_DIR / "partial.pkl"
    oofs = {}
    if oof_pkl.exists():
        oofs = pickle.load(open(oof_pkl, "rb")); print(f"[resume] {list(oofs.keys())}", flush=True)
    def save_partial():
        pickle.dump(oofs, open(oof_pkl, "wb"))

    # ============ 1. ratio_lgb ============
    if "ratio_lgb" not in oofs:
        print("\n=== ratio_lgb v2 (train+backfill) ===", flush=True); t0 = time.time()
        Xtr, Xva = X_full[~is_va], X_full[is_va]
        ytr, wtr = y32[~is_va], w32[~is_va]
        dtr = lgb.Dataset(Xtr, label=ytr, weight=wtr, categorical_feature=[0], free_raw_data=False)
        dva = lgb.Dataset(Xva, label=yv.astype(np.float32), weight=wv.astype(np.float32), reference=dtr, free_raw_data=False)
        m = lgb.train(TUNED, dtr, num_boost_round=1500, valid_sets=[dva], valid_names=["va"], feval=feval_wr2,
                      callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)])
        bi = m.best_iteration or 1500
        oofs["ratio_lgb"] = m.predict(Xva)
        r2 = wr2(yv, oofs["ratio_lgb"], wv)
        print(f"  holdout R²={r2:+.5f} iter={bi} ({time.time()-t0:.0f}s)", flush=True)
        Xf = X_full; dtr_full = lgb.Dataset(Xf, label=y32, weight=w32, categorical_feature=[0], free_raw_data=False)
        for s in [2026, 2027, 2028]:
            p = dict(TUNED); p.update(seed=s, bagging_seed=s, feature_fraction_seed=s)
            mm = lgb.train(p, dtr_full, num_boost_round=bi, feval=feval_wr2)
            mm.save_model(str(MODEL_DIR / f"ratio_lgb_seed{s}.txt"), num_iteration=bi)
            print(f"  saved ratio_lgb_seed{s}.txt", flush=True)
        oofs["lgb_best_iter"] = bi; oofs["lgb_r2"] = float(r2)
        save_partial(); del dtr, dva, dtr_full, m; gc.collect()
    else: print("[skip] ratio_lgb", flush=True)

    # ============ 2. catboost ============
    if "cb" not in oofs:
        print("\n=== catboost v2 (train+backfill) ===", flush=True); t0 = time.time()
        pdf = pf[["asset_id"]+feats].copy(); pdf["asset_id"] = pdf["asset_id"].astype(np.int32)
        tr_df = pdf[~is_va].reset_index(drop=True); va_df = pdf[is_va].reset_index(drop=True)
        trpool = cb.Pool(tr_df, label=y32[~is_va], weight=w32[~is_va], cat_features=["asset_id"])
        vapool = cb.Pool(va_df, label=yv.astype(np.float32), weight=wv.astype(np.float32), cat_features=["asset_id"])
        mc = cb.train(trpool, CB_PARAMS, eval_set=vapool, verbose=False)
        bi_cb = mc.tree_count_
        oofs["cb"] = mc.predict(vapool)
        r2 = wr2(yv, oofs["cb"], wv)
        print(f"  holdout R²={r2:+.5f} trees={bi_cb} ({time.time()-t0:.0f}s)", flush=True)
        fullpool = cb.Pool(pdf, label=y32, weight=w32, cat_features=["asset_id"])
        for s in [2026, 2027, 2028]:
            p = dict(CB_PARAMS); p["random_seed"] = s; p["iterations"] = bi_cb; p["use_best_model"] = False
            mm = cb.train(fullpool, p, verbose=False)
            mm.save_model(str(MODEL_DIR / f"cb_seed{s}.cbm"))
            print(f"  saved cb_seed{s}.cbm ({bi_cb} trees)", flush=True)
        oofs["cb_trees"] = bi_cb; oofs["cb_r2"] = float(r2)
        save_partial(); del trpool, vapool, fullpool, mc, pdf, tr_df, va_df; gc.collect()
    else: print("[skip] cb", flush=True)

    # ============ 3. nn_emb32 ============
    if "nn" not in oofs:
        print("\n=== nn_emb32 v2 (train+backfill) ===", flush=True); t0 = time.time()
        mean = F[~is_va].mean(0); std = F[~is_va].std(0)+1e-6
        def stdize(Fx):
            return np.nan_to_num((Fx-mean)/std, nan=0, posinf=0, neginf=0).astype(np.float32)
        Xtr_s = torch.from_numpy(stdize(F[~is_va])).to(DEV)
        Xva_s = torch.from_numpy(stdize(F[is_va])).to(DEV)
        Atr = torch.from_numpy(pf["asset_id"].to_numpy(np.int64)[~is_va]).to(DEV)
        Ava = torch.from_numpy(pf["asset_id"].to_numpy(np.int64)[is_va]).to(DEV)
        Ytr = torch.from_numpy(y32[~is_va]).to(DEV); Wtr = torch.from_numpy(w32[~is_va]).to(DEV)
        bs = 16384; n_tr = len(Xtr_s); n_feat = len(feats)
        seeds = list(range(2026, 2036)); acc_va = []
        for sd in seeds:
            torch.manual_seed(sd); mlp = MLP(n_feat).to(DEV)
            opt = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-5)
            mlp.train(); perm = torch.randperm(n_tr, device=DEV)
            for i in range(0, n_tr, bs):
                idx = perm[i:i+bs]; opt.zero_grad()
                loss = (Wtr[idx]*(mlp(Xtr_s[idx],Atr[idx])-Ytr[idx])**2).mean(); loss.backward(); opt.step()
            mlp.eval()
            with torch.no_grad():
                pv = [mlp(Xva_s[i:i+16384],Ava[i:i+16384]).cpu().numpy() for i in range(0,len(Xva_s),16384)]
                acc_va.append(np.concatenate(pv))
            torch.save(mlp.state_dict(), MODEL_DIR / f"nn_emb32_seed{sd}.pt")
            print(f"  saved nn_emb32_seed{sd}.pt", flush=True)
        oofs["nn"] = np.mean(acc_va, 0)
        r2 = wr2(yv, oofs["nn"], wv)
        print(f"  holdout R²={r2:+.5f} ({time.time()-t0:.0f}s)", flush=True)
        oofs["nn_r2"] = float(r2); oofs["nn_mean"] = mean.tolist(); oofs["nn_std"] = std.tolist()
        save_partial()
        del Xtr_s, Xva_s, Atr, Ava, Ytr, Wtr; gc.collect(); torch.cuda.empty_cache()
    else: print("[skip] nn", flush=True)

    # ============ 4. 配比优化（holdout 上 SLSQP）============
    from scipy.optimize import minimize
    P3 = np.array([oofs["ratio_lgb"], oofs["cb"], oofs["nn"]])
    def neg(wv): return -wr2(yv, wv @ P3, wv)
    cons = ({"type": "eq", "fun": lambda x: x.sum() - 1}); bnds = [(0,1)]*3
    best = None; np.random.seed(2026)
    for _ in range(30):
        r = minimize(neg, np.random.dirichlet(np.ones(3)), method="SLSQP", bounds=bnds, constraints=cons)
        if best is None or r.fun < best.fun: best = r
    wv = np.maximum(best.x, 0); wv = wv/wv.sum()
    ens_r2 = wr2(yv, wv @ P3, wv)
    print(f"\n=== v2 集成 holdout R²={ens_r2:+.5f} ===")
    print(f"  weights: lgb={wv[0]:.3f} cb={wv[1]:.3f} nn={wv[2]:.3f}", flush=True)
    # 0.35/0.35/0.30 对比
    m_r2 = wr2(yv, np.array([0.35,0.35,0.30]) @ P3, wv)
    print(f"  manual 0.35/0.35/0.30: {m_r2:+.5f}", flush=True)

    report = {
        "data": "train(13.2M) + backfill(3.2M)", "holdout": "最新 15% 时段（含回补尾部）",
        "ratio_denom": RATIO_DENOM, "lgb_best_iter": int(oofs.get("lgb_best_iter", 0)),
        "lgb_r2": oofs.get("lgb_r2"), "cb_r2": oofs.get("cb_r2"), "nn_r2": oofs.get("nn_r2"),
        "ens_slsqp_r2": float(ens_r2), "ens_manual_r2": float(m_r2),
        "weights": {"lgb": float(wv[0]), "cb": float(wv[1]), "nn": float(wv[2])},
        "ratio_hi_pct": hi_pct, "ratio_lo": lo.tolist(), "ratio_hi": hi.tolist(),
        "nn_mean": oofs.get("nn_mean"), "nn_std": oofs.get("nn_std"),
    }
    (MODEL_DIR / "v2_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    np.savez_compressed(MODEL_DIR / "holdout_oof.npz",
                        lgb=oofs["ratio_lgb"], cb=oofs["cb"], nn=oofs["nn"], yv=yv, wv=wv)
    print(f"\n[done] v2 models + report saved to {MODEL_DIR}", flush=True)


if __name__ == "__main__":
    main()
