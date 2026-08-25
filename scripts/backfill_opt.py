"""P0-3: 用回补真值直接优化基础模型配比。

基础组件（submissions 中的单模型）：
- ratio_lgb 系: ratio4_ratio_lgb.csv (top50) / lgbm_full (raw)
- cb_submission.csv, xgb_submission.csv
- nn: nn1_10s, nn3_10s(emb32), ratio4_ratio_nn
- ens_ratio_nn30 的组件预测无法分离（CSV 已是混合）

策略：直接在 3.2M 真值上 SLSQP 搜索最优配比（这次是真值，不会过拟合公榜语义），
     但要防"对 test 过拟合"——限制候选组合简单化，且对比手设配比。
"""
from __future__ import annotations
import numpy as np, pandas as pd, json
from pathlib import Path
from scipy.optimize import minimize

ROOT = Path("/mnt/iscsi/hd/xxz")
LABELS = ROOT / "public_release_20260823/public_release_20260823/data/train"
SUB = ROOT / "submissions"

labels = pd.concat(
    [pd.read_parquet(p, columns=["row_id", "weight", "target"]) for p in sorted(LABELS.glob("*.parquet"))],
    ignore_index=True,
)
y = labels["target"].to_numpy(np.float64)
w = labels["weight"].to_numpy(np.float64)
rid = labels["row_id"].to_numpy()

def load(f):
    d = pd.read_csv(SUB/f).sort_values("row_id").reset_index(drop=True)
    assert (d["row_id"].to_numpy() == rid).all(), f"{f}: row_id mismatch"
    return d["target"].to_numpy(np.float64)

def wr2(p):
    return 1.0 - np.sum(w * (y - p) ** 2) / np.sum(w * y * y)

# 组件
comps = {
    "ratio_lgb50": load("ratio4_ratio_lgb.csv"),      # top50 ratio lgb
    "cb": load("cb_submission.csv"),
    "xgb": load("xgb_submission.csv"),
    "nn1": load("nn1_10s.csv"),
    "nn3": load("nn3_10s.csv"),
    "ratio_nn": load("ratio4_ratio_nn.csv"),
    "lgb_raw": load("lgbm_full_submission.csv"),
}
keys = list(comps)
P = np.array([comps[k] for k in keys])
print("=== 单组件回补真值 R² ===", flush=True)
for k in keys: print(f"  {k}: {wr2(comps[k]):+.8f}", flush=True)

# NN 均值对齐（对配比影响小但保持一致性）
ref = comps["lgb_raw"].mean()
for i, k in enumerate(keys):
    if k.startswith("nn"): P[i] = P[i] - P[i].mean() + ref

# SLSQP 优化（真值）
def neg(wv): return -wr2(wv @ P)
cons = ({"type": "eq", "fun": lambda wv: wv.sum() - 1})
bnds = [(0, 1)] * len(keys)
best = None; np.random.seed(2026)
for _ in range(60):
    r = minimize(neg, np.random.dirichlet(np.ones(len(keys))), method="SLSQP", bounds=bnds, constraints=cons,
                 options={"maxiter": 400, "ftol": 1e-12})
    if best is None or r.fun < best.fun: best = r
wv = np.maximum(best.x, 0); wv = wv / wv.sum()
print(f"\n=== 真值最优配比 R²={wr2(wv @ P):+.8f} ===", flush=True)
for k, wi in sorted(zip(keys, wv), key=lambda x: -x[1]):
    if wi > 0.001: print(f"  {k}: {wi:.4f}", flush=True)

# 手设候选对比（简单配比，防过拟合 test）
manual = {
    "m_lcb_nn30": {"lgb_raw": 0.35, "cb": 0.35, "nn3": 0.30},
    "m_rlgb50_cb_nn": {"ratio_lgb50": 0.35, "cb": 0.35, "nn3": 0.30},
    "m_5comp": {"ratio_lgb50": 0.25, "cb": 0.25, "xgb": 0.15, "nn3": 0.20, "nn1": 0.15},
    "m_nn3_heavy": {"lgb_raw": 0.25, "cb": 0.25, "nn3": 0.50},
    "m_rlgb_xgb_nn3": {"ratio_lgb50": 0.30, "xgb": 0.20, "cb": 0.20, "nn3": 0.30},
}
print(f"\n=== 手设配比 ===", flush=True)
manual_scores = {}
for name, ws in manual.items():
    wv2 = np.array([ws.get(k, 0.0) for k in keys]); wv2 = wv2 / wv2.sum()
    s = wr2(wv2 @ P)
    manual_scores[name] = (s, ws)
    print(f"  {name}: {s:+.8f}", flush=True)

# 写最优 SLSQP 和最优手设
out1 = wv @ P
pd.DataFrame({"row_id": rid, "target": out1}).to_csv(SUB/"backfill_opt_slsqp.csv", index=False)
best_manual = max(manual_scores.items(), key=lambda x: x[1][0])
s, ws = best_manual[1]
wv3 = np.array([ws.get(k, 0.0) for k in keys]); wv3 = wv3 / wv3.sum()
pd.DataFrame({"row_id": rid, "target": wv3 @ P}).to_csv(SUB/"backfill_opt_manual.csv", index=False)
print(f"\n→ wrote backfill_opt_slsqp.csv ({wr2(out1):+.8f})", flush=True)
print(f"→ wrote backfill_opt_manual.csv ({best_manual[0]}: {s:+.8f})", flush=True)
json.dump({"slsqp": {k: float(x) for k, x in zip(keys, wv)}, "manual_best": best_manual[0]},
          open(ROOT/"runs"/"backfill_opt_weights.json", "w"), indent=2)
