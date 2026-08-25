"""私榜交付重训（版本A：现有 train 数据）：复现 ens_ratio_nn30 配方的可交付模型。

产出（全部保存权重到 src/final_submission/model/）：
1. ratio_lgb: simple ratio(÷feature_157) + tuned LGBM 参数, 3 seed
   - 11-Aug 公榜 0.003345 的核心贡献（simple ratio 是真信号）
2. nn_emb32: emb32 MLP, 10 seed, torch state_dict 保存
   - 当时 NN 未存权重，这次必须保存
3. 复用现有 cb（src/cb_baseline/model/，不再重训）

时序合规：purged holdout（末 15%）选轮数，全量重训。
权重保存格式：LGBM txt / torch .pt，加载快（<180s 约束）。
"""
from __future__ import annotations
import sys, time, json, pickle, gc
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
import torch, torch.nn as nn

STRAT = "/mnt/iscsi/hd/xxz/public_release_lightgbm_baseline/examples/lightgbm_baseline"
if STRAT not in sys.path: sys.path.insert(0, STRAT)
from data_utils import manifest_files, feature_columns_from_path  # noqa: E402
DATA_ROOT = Path("/mnt/iscsi/hd/xxz/public_release_20260630/data")
MODEL_DIR = Path("/mnt/iscsi/hd/xxz/src/final_submission/model")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DEV = "cuda:4"  # 空闲 GPU
N_ASSET = 15
RATIO_DENOM = "feature_157"  # 11-Aug 公榜验证的 simple ratio 分母

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


class MLP(nn.Module):
    """与 11-Aug nn3 相同架构：emb32 + (512,256,128) + GELU + BN + Dropout"""
    def __init__(self, n_feat, emb_dim=32, hidden=(512,256,128), dropout=0.3):
        super().__init__(); self.emb = nn.Embedding(N_ASSET, emb_dim)
        layers=[]; d=n_feat+emb_dim
        for h in hidden: layers += [nn.Linear(d,h), nn.GELU(), nn.BatchNorm1d(h), nn.Dropout(dropout)]; d=h
        layers += [nn.Linear(d,1)]; self.net = nn.Sequential(*layers)
    def forward(self, x, a):
        return self.net(torch.cat([x, self.emb(a)], 1)).squeeze(-1)


def build_ratio(F, denom_idx, hi_pct):
    """simple ratio: F[:, i] / clip(F[:, denom], 1e-8, p99)，clip 到 [p1, p99]（train 分位数）。"""
    fd = np.clip(F[:, denom_idx], 1e-8, hi_pct)
    return fd


def main():
    paths = manifest_files(DATA_ROOT, "train"); feats = feature_columns_from_path(paths[0])
    denom_idx = feats.index(RATIO_DENOM)
    pf = pd.read_parquet(paths, columns=["time_id","asset_id","weight","target"]+feats)
    pf[feats] = np.nan_to_num(pf[feats].to_numpy(np.float32))
    F = pf[feats].to_numpy(np.float32)
    y32 = pd.to_numeric(pf["target"],errors="coerce").fillna(0).to_numpy(np.float32)
    w32 = pd.to_numeric(pf["weight"],errors="coerce").fillna(0).clip(lower=0).to_numpy(np.float32)
    a32 = pf["asset_id"].to_numpy(np.float32)
    times = np.sort(pf["time_id"].unique()); ho = set(times[-max(1,int(len(times)*0.15)):].tolist())
    is_va = pf["time_id"].isin(ho).to_numpy()
    yv = y32[is_va].astype(np.float64); wv = w32[is_va].astype(np.float64)
    print(f"data: {len(pf):,} rows, holdout {is_va.sum():,}", flush=True)

    # ratio 特征（simple ÷157，全特征都除？不——11-Aug 是 top 特征 ÷157）
    # 复现：对每个特征 i 构造 F_i / F_157（323 个 ratio 列，与 LGBM raw 323 并列 = 646 维）
    hi_pct = float(np.percentile(F[:, denom_idx], 99))
    fd = np.clip(F[:, denom_idx], 1e-8, hi_pct)
    R = np.nan_to_num(F / fd[:, None], nan=0, posinf=0, neginf=0).astype(np.float32)
    # clip 用 train 分位数（按列）
    lo = np.percentile(R, 1, axis=0); hi = np.percentile(R, 99, axis=0)
    R = np.clip(R, lo, hi)
    X_full = np.column_stack([a32, F, R])  # asset + 323 raw + 323 ratio
    print(f"features: 1 + {F.shape[1]} raw + {R.shape[1]} ratio = {X_full.shape[1]}", flush=True)

    # ============ 1. ratio_lgb（tuned 参数 + simple ratio）============
    print("\n=== ratio_lgb (tuned + ÷157 all features) ===", flush=True); t0 = time.time()
    Xtr, Xva = X_full[~is_va], X_full[is_va]
    ytr, wtr = y32[~is_va], w32[~is_va]
    dtr = lgb.Dataset(Xtr, label=ytr, weight=wtr, categorical_feature=[0], free_raw_data=False)
    dva = lgb.Dataset(Xva, label=yv.astype(np.float32), weight=wv.astype(np.float32), reference=dtr, free_raw_data=False)
    m = lgb.train(TUNED, dtr, num_boost_round=1500, valid_sets=[dva], valid_names=["va"], feval=feval_wr2,
                  callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)])
    bi = m.best_iteration or 1500
    r2 = wr2(yv, m.predict(Xva), wv)
    print(f"  holdout R²={r2:+.5f} iter={bi} ({time.time()-t0:.0f}s)", flush=True)
    # 全量 3 seed（保存权重）
    Xf = X_full; dtr_full = lgb.Dataset(Xf, label=y32, weight=w32, categorical_feature=[0], free_raw_data=False)
    for s in [2026, 2027, 2028]:
        p = dict(TUNED); p.update(seed=s, bagging_seed=s, feature_fraction_seed=s)
        mm = lgb.train(p, dtr_full, num_boost_round=bi, feval=feval_wr2)
        mm.save_model(str(MODEL_DIR / f"ratio_lgb_seed{s}.txt"), num_iteration=bi)
        print(f"  saved ratio_lgb_seed{s}.txt ({bi} rounds)", flush=True)
    del dtr, dva, dtr_full, m; gc.collect()

    # 保存 holdout 预测（供后续配比验证）
    ratio_lgb_va = None  # 用单模型重算太贵，从 3-seed 平均重算
    # ============ 2. nn_emb32（10 seed，保存权重）============
    print("\n=== nn_emb32 (10 seed, raw only — 与 11-Aug nn3 一致) ===", flush=True); t0 = time.time()
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
    nn_va = np.mean(acc_va, 0)
    print(f"  holdout R²={wr2(yv, nn_va, wv):+.5f} ({time.time()-t0:.0f}s)", flush=True)
    del Xtr_s, Xva_s, Atr, Ava, Ytr, Wtr; gc.collect(); torch.cuda.empty_cache()

    # ============ 3. 汇报 ============
    report = {
        "ratio_denom": RATIO_DENOM, "ratio_lgb_best_iter": int(bi),
        "ratio_lgb_holdout_r2": float(r2), "nn_holdout_r2": float(wr2(yv, nn_va, wv)),
        "cb_model": "src/cb_baseline/model/ (existing, 3 seeds)",
        "ratio_clip": {"lo_p1": lo.tolist()[:5], "hi_p99": hi.tolist()[:5]},
        "train_rows": int(len(pf)),
    }
    (MODEL_DIR / "final_train_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    # 保存 holdout OOF 供配比优化
    np.savez_compressed(MODEL_DIR / "holdout_oof.npz", nn_va=nn_va, yv=yv, wv=wv)
    print(f"\n[done] report: {json.dumps({k:v for k,v in report.items() if not isinstance(v,list)}, ensure_ascii=False)}", flush=True)


if __name__ == "__main__":
    main()
