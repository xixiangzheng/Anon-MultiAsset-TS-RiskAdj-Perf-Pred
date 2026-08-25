"""组装私榜交付包：~/submit 结构 + inference_config.json + submission.zip。

用法：.venv/bin/python scripts/build_submission.py
产出：
  src/final_submission/submit/main.py          — 交付入口
  src/final_submission/submit/model/*.txt|cbm|pt — 权重
  src/final_submission/submit/model/inference_config.json — 推理配置
  src/final_submission/submit/requirements.txt
  src/final_submission/submit/submission.zip
"""
from __future__ import annotations
import json, shutil, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd

STRAT = "/mnt/iscsi/hd/xxz/public_release_lightgbm_baseline/examples/lightgbm_baseline"
if STRAT not in sys.path: sys.path.insert(0, STRAT)
from data_utils import manifest_files, feature_columns_from_path

ROOT = Path("/mnt/iscsi/hd/xxz")
MODEL_DIR = ROOT / "src/final_submission/model_v2"
SUBMIT = ROOT / "src/final_submission/submit"
CB_SRC = ROOT / "src/final_submission/model_v2"
DATA_ROOT = ROOT / "public_release_20260630/data"
RATIO_DENOM = "feature_157"


def main():
    # 清空重建
    if SUBMIT.exists():
        shutil.rmtree(SUBMIT)
    (SUBMIT / "model").mkdir(parents=True)

    # 1. main.py
    shutil.copy(ROOT / "src/final_submission/main.py", SUBMIT / "main.py")

    # 2. 模型权重
    n_lgb = 0
    for f in sorted(MODEL_DIR.glob("ratio_lgb_seed*.txt")):
        shutil.copy(f, SUBMIT / "model" / f.name); n_lgb += 1
    n_nn = 0
    for f in sorted(MODEL_DIR.glob("nn_emb32_seed*.pt")):
        shutil.copy(f, SUBMIT / "model" / f.name); n_nn += 1
    n_cb = 0
    for f in sorted(CB_SRC.glob("cb_seed*.cbm")):
        shutil.copy(f, SUBMIT / "model" / f.name); n_cb += 1
    print(f"models: lgb={n_lgb} cb={n_cb} nn={n_nn}")
    assert n_lgb > 0 and n_cb > 0 and n_nn > 0, "缺少模型权重（final_train.py 未完成？）"

    # 3. inference_config.json（ratio clip / nn 标准化参数从 train+backfill 重算）
    print("computing inference config from train+backfill data...")
    paths = manifest_files(DATA_ROOT, "train")
    BACKFILL = ROOT / "public_release_20260823/public_release_20260823/data"
    feats = feature_columns_from_path(paths[0])
    denom_idx = feats.index(RATIO_DENOM)
    # 采样：主 train 2 分区 + 回补 1 分区（覆盖新旧分布）
    parts = list(paths[:2]) + sorted(BACKFILL.glob("train/*.parquet"))[:1]
    pf = pd.read_parquet(parts, columns=["asset_id"] + feats)
    pf[feats] = np.nan_to_num(pf[feats].to_numpy(np.float32))
    F = pf[feats].to_numpy(np.float32)
    hi_pct = float(np.percentile(F[:, denom_idx], 99))
    fd = np.clip(F[:, denom_idx], 1e-8, hi_pct)
    R = np.nan_to_num(F / fd[:, None], nan=0, posinf=0, neginf=0).astype(np.float32)
    ratio_lo = np.percentile(R, 1, axis=0).tolist()
    ratio_hi = np.percentile(R, 99, axis=0).tolist()
    nn_mean = F.mean(0).tolist(); nn_std = (F.std(0) + 1e-6).tolist()
    raw_lo = np.percentile(F, 0.1, axis=0).tolist()   # 防极端 outlier（test 有 1e6 级值）
    raw_hi = np.percentile(F, 99.9, axis=0).tolist()

    report = json.loads((MODEL_DIR / "v2_report.json").read_text(encoding="utf-8"))
    wx = report["weights"]
    cfg = {
        "feature_cols": feats,
        "ratio_denom": RATIO_DENOM,
        "ratio_denom_idx": denom_idx,
        "ratio_hi_pct": hi_pct,
        "ratio_lo": ratio_lo, "ratio_hi": ratio_hi,
        "nn_mean": nn_mean, "nn_std": nn_std,
        "raw_lo": raw_lo, "raw_hi": raw_hi,
        "nn_seeds": list(range(2026, 2026 + n_nn)),
        "nn_hidden": [512, 256, 128], "nn_emb_dim": 32,
        "lgb_best_iter": int(report["lgb_best_iter"]),
        "weights": {"lgb": float(wx["lgb"]), "cb": float(wx["cb"]), "nn": float(wx["nn"])},
        "num_threads": 4,
    }
    (SUBMIT / "model" / "inference_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False))
    print(f"config: denom_idx={denom_idx} hi_pct={hi_pct:.4g} lgb_iter={cfg['lgb_best_iter']}")

    # 4. requirements.txt（只列真正需要的）
    reqs = """lightgbm==4.6.0
catboost==1.2.8
torch==2.13.0
numpy==2.2.6
pandas==2.2.3
"""
    (SUBMIT / "requirements.txt").write_text(reqs)

    # 5. zip
    subprocess.run(["zip", "-r", "submission.zip", "main.py", "requirements.txt", "model"],
                   cwd=SUBMIT, check=True)
    out = SUBMIT / "submission.zip"
    print(f"zip: {out} ({out.stat().st_size/1e6:.1f} MB)")
    subprocess.run(["unzip", "-l", str(out)])
    print("\n[done] submit dir:", SUBMIT)


if __name__ == "__main__":
    main()
