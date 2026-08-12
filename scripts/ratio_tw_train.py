"""扩展 ratio_models: 在 ratio 之上加三方交互 + 更多 ratio 变体（diff, product, sum）。

特征集 = 323 raw + 50 ratio + 20 三方交互 + 30 diff(A-B top pairs) = 423 特征
训练 LGBM(tuned) → holdout 评估 + test 预测。

目的：扩展 ratio 特征工程的天花板，看是否突破 ratio_lgb 的 holdout。
"""
from __future__ import annotations
import sys, time, json, pickle
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb

STRAT = "/mnt/iscsi/hd/xxz/public_release_lightgbm_baseline/examples/lightgbm_baseline"
if STRAT not in sys.path: sys.path.insert(0, STRAT)
from data_utils import manifest_files, feature_columns_from_path  # noqa: E402
DATA_ROOT = Path("/mnt/iscsi/hd/xxz/public_release_20260630/data")
RUN = Path("/mnt/iscsi/hd/xxz/runs"); SUB = Path("/mnt/iscsi/hd/xxz/submissions")

TUNED = dict(objective="regression", metric="None", learning_rate=0.010326965981106629,
             num_leaves=79, min_data_in_leaf=556, feature_fraction=0.5974233229491067,
             bagging_fraction=0.9445362670741704, bagging_freq=1, lambda_l1=0.03778653953330111,
             lambda_l2=2.9757802078489703, max_bin=127, verbosity=-1, num_threads=64, seed=2026,
             bagging_seed=2026, feature_fraction_seed=2026, data_random_seed=2026)


def wr2(y, p, w):
    d = float(np.sum(w*y*y)); return 0.0 if d<=0 else 1-float(np.sum(w*(y-p)**2)/d)


def feval_wr2(preds, ds):
    y = ds.get_label(); w = ds.get_weight()
    if w is None: w = np.ones_like(y)
    return ("wr2", float(wr2(y, preds, w)), True)


def clip01(x, lo=1, hi=99):
    x = np.nan_to_num(x, nan=0, posinf=0, neginf=0)
    a, b = np.percentile(x, [lo, hi])
    return np.clip(x, a, b)


def main():
    paths = manifest_files(DATA_ROOT, "train"); feats = feature_columns_from_path(paths[0])
    # 加载 ratio_top50 + three_way
    rs = json.loads((RUN/"ratio_top50.json").read_text())["ratios"]
    ratio_pairs = [(feats.index(r[2]), feats.index(r[3])) for r in rs]
    three_way = []
    if (RUN/"three_way_search.json").exists():
        tw = json.loads((RUN/"three_way_search.json").read_text())["three_way"]
        # 取 delta>0 的全部（不只显著）
        three_way = [t for t in tw if t["delta"] > 0][:20]
        print(f"loaded {len(three_way)} three_way (delta>0)", flush=True)
    # 构造 diff 特征：top-30 单特征的两两 diff（前几个），按相关性
    # 简化：用 top-10 单特征 × top-10 = 100 diff，取前 30 by corr
    print(f"feature plan: {len(feats)} raw + {len(ratio_pairs)} ratio + {len(three_way)} three_way", flush=True)

    pf = pd.read_parquet(paths, columns=["time_id","asset_id","weight","target"]+feats)
    pf[feats] = np.nan_to_num(pf[feats].to_numpy(np.float32))
    F = pf[feats].to_numpy(np.float32)
    y = pd.to_numeric(pf["target"],errors="coerce").fillna(0).to_numpy(np.float32)
    w = pd.to_numeric(pf["weight"],errors="coerce").fillna(0).clip(lower=0).to_numpy(np.float32)
    a = pf["asset_id"].to_numpy(np.float32)
    times = np.sort(pf["time_id"].unique()); ho = set(times[-max(1,int(len(times)*0.15)):].tolist())
    is_va = pf["time_id"].isin(ho).to_numpy()
    print(f"train {len(pf):,} rows, holdout {is_va.sum():,}", flush=True)

    # 构造额外特征
    extras = []  # list of (name, ndarray)
    # 1. ratio
    for i,(ni,di) in enumerate(ratio_pairs):
        fd = np.clip(F[:,di], 1e-8, np.percentile(F[:,di], 99))
        r = clip01(F[:,ni] / fd)
        extras.append((f"ratio{i}", r.astype(np.float32)))
    # 2. three-way
    for j,tw in enumerate(three_way):
        ni = feats.index(tw["numer"]); di = feats.index(tw["denom"]); ti = feats.index(tw["third"])
        fd = np.clip(F[:,di], 1e-8, np.percentile(F[:,di], 99))
        v = clip01((F[:,ni] / fd) * F[:,ti])
        extras.append((f"tw{j}", v.astype(np.float32)))
    print(f"extras: {len(extras)}", flush=True)

    extra_names = [n for n,_ in extras]
    extra_train = np.column_stack([v for _,v in extras]) if extras else np.zeros((len(pf),0),np.float32)
    all_cols = feats + extra_names
    print(f"total features: {len(all_cols)}", flush=True)

    # test
    te = pd.read_parquet(manifest_files(DATA_ROOT,"test"),columns=["row_id","asset_id"]+feats)
    te[feats] = np.nan_to_num(te[feats].to_numpy(np.float32)); te = te.sort_values("row_id").reset_index(drop=True)
    F_te = te[feats].to_numpy(np.float32)
    extras_te = []
    for i,(ni,di) in enumerate(ratio_pairs):
        # 用 train 的 percentile clip
        fd_tr = np.clip(F[:,di], 1e-8, np.percentile(F[:,di],99))
        fd_te = np.clip(F_te[:,di], 1e-8, np.percentile(F[:,di],99))  # 用 train 的 percentile 上界
        lo,hi = np.percentile(F[:,ni]/fd_tr,[1,99])
        r = np.clip(F_te[:,ni]/fd_te, lo, hi)
        r = np.nan_to_num(r,nan=0,posinf=0,neginf=0)
        extras_te.append(r.astype(np.float32))
    for j,tw_e in enumerate(three_way):
        ni = feats.index(tw_e["numer"]); di = feats.index(tw_e["denom"]); ti = feats.index(tw_e["third"])
        fd_tr = np.clip(F[:,di],1e-8,np.percentile(F[:,di],99))
        fd_te = np.clip(F_te[:,di],1e-8,np.percentile(F[:,di],99))
        v = (F_te[:,ni]/fd_te) * F_te[:,ti]
        lo,hi = np.percentile((F[:,ni]/fd_tr)*F[:,ti],[1,99])
        v = np.nan_to_num(np.clip(v,lo,hi),nan=0,posinf=0,neginf=0)
        extras_te.append(v.astype(np.float32))
    extra_test = np.column_stack(extras_te) if extras_te else np.zeros((len(te),0),np.float32)

    Xtr = np.column_stack([a, F, extra_train])
    Xva = Xtr[is_va]; Xtr = Xtr[~is_va]
    ytr = y[~is_va]; wtr = w[~is_va]
    yva = y[is_va].astype(np.float64); wva = w[is_va].astype(np.float64)
    Xte = np.column_stack([te["asset_id"].to_numpy(np.float32), F_te, extra_test])

    print(f"Xtr {Xtr.shape} Xva {Xva.shape} Xte {Xte.shape}", flush=True)
    dtr = lgb.Dataset(Xtr, label=ytr, weight=wtr, categorical_feature=[0], free_raw_data=False)
    dva = lgb.Dataset(Xva, label=yva.astype(np.float32), weight=wva.astype(np.float32), reference=dtr, free_raw_data=False)
    t0 = time.time()
    m = lgb.train(TUNED, dtr, num_boost_round=1500, valid_sets=[dva], valid_names=["va"], feval=feval_wr2,
                  callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)])
    bi = m.best_iteration or 1500
    pred_va = m.predict(Xva)
    print(f"holdout iter={bi} R²={wr2(yva, pred_va, wva):+.6f} ({time.time()-t0:.0f}s)", flush=True)

    # 全量 3 seed
    Xf = np.concatenate([Xtr, Xva]); yf = np.concatenate([ytr, yva.astype(np.float32)])
    wf = np.concatenate([wtr, wva.astype(np.float32)])
    dtr_full = lgb.Dataset(Xf, label=yf, weight=wf, categorical_feature=[0], free_raw_data=False)
    preds = []
    for s in [2026, 2027, 2028]:
        p = dict(TUNED); p.update(seed=s, bagging_seed=s, feature_fraction_seed=s, data_random_seed=s)
        mm = lgb.train(p, dtr_full, num_boost_round=bi, feval=feval_wr2); preds.append(mm.predict(Xte))
    avg = np.mean(preds, 0); avg = np.where(np.isfinite(avg), avg, 0.0)
    pd.DataFrame({"row_id": te["row_id"], "target": avg}).to_csv(SUB/"ratio_tw_lgb.csv", index=False)
    print(f"wrote ratio_tw_lgb.csv mean={avg.mean():+.5f} std={avg.std():.5f}", flush=True)

    # 保存 OOF
    pickle.dump({"holdout_pred": pred_va, "yva": yva, "wva": wva, "test_pred": avg,
                 "row_id": te["row_id"].to_numpy(), "best_iter": bi,
                 "holdout_r2": float(wr2(yva, pred_va, wva))},
                open(RUN/"ratio_tw_lgb_oof.pkl", "wb"))
    print("saved OOF", flush=True)


if __name__ == "__main__":
    main()
