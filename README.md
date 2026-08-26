# Risk-Adjusted Performance Prediction for Anonymous Multi-Asset Time-Series Data

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**English Version** | **[中文版](./README_CN.md)**

> This repository contains our complete submission to the **2026 Quantitative Trading Research Competition**: predicting a risk-adjusted target on anonymous multi-asset time-series panel data, scored by Weighted Zero-Mean R² under a sequential Time-Series API (4 CPU cores / 12GB RAM / no GPU / ≤50ms per step / ≤180s model load).

## Results

The target is innovation-like and extremely weak-signal — every +0.0005 of R² is hard-won. Starting from the official LightGBM baseline, we progressed through 14 public-leaderboard submissions and a full label-backfill re-scoring of 91 candidate files:

| Stage | Weighted Zero-Mean R² | Strategy |
|---|---|---|
| Official baseline | 0.00313 | LightGBM (323 raw features) |
| Public LB best (submitted) | **0.00335** | ratio-LGBM + CatBoost + MLP ensemble |
| **Final delivery (holdout, clean OOF)** | **0.00386** | retrained ensemble on train + label-backfill (16.4M rows) |

**Final delivery package**: [`release/submission.zip`](release/submission.zip) — self-contained (16 model checkpoints + inference config), passes the official runner with `status=ok`, 2.3s init, 20.6ms mean per-step on 4 pinned cores, zero exceptions over 214,538 predict calls.

## Method

The final prediction is a three-family ensemble:

```
prediction = 0.358 × ratio_lgb(3 seeds) + 0.253 × catboost(3 seeds) + 0.389 × nn_emb32(10 seeds)
```

| Component | Features | Notes |
|---|---|---|
| **ratio_lgb** | raw 323 + ratio 323 | tuned LightGBM; simple ratio = every feature ÷ `feature_157` (the only feature engineering validated on the public LB) |
| **catboost** | raw 323 + asset_id | symmetric trees, decorrelated from LightGBM (corr 0.88) |
| **nn_emb32** | standardized raw 323 | MLP (512/256/128) with a learned asset embedding; 10-seed averaging |

Three decisions mattered most:

1. **Label-backfill data as the strongest lever.** When the organizer released ground-truth labels for the public test window (3.2M rows), merging them into training (16.4M rows total) doubled clean holdout R² (0.0017 → 0.0036) — the backfill segment is temporally closest to the future private-LB regime, worth more than every model trick combined.
2. **Simple beats clever.** A single hand-crafted ratio (÷ feature_157) gained +0.00003 on the public LB, while a systematically searched top-50 ratio family *lost* −0.00016 — the search overfit partition-specific noise. We verified this across three failed LB submissions.
3. **Inference robustness is the delivery lifeline.** The test set contains raw outliers up to 1e6; without clipping, the MLP extrapolates to outputs of ~465 and the score collapses to −204. All features are clipped to train p0.1/p99.9 at inference time.

A fuller analysis of what worked and what failed is in [LESSONS.md](LESSONS.md); the day-by-day experiment history is in [docs/](docs/) (Chinese).

## Repository Structure

```
├── release/submission.zip        # ★ final delivery (main.py + 16 checkpoints + config)
├── src/
│   ├── final_submission/         # delivery strategy (Model class, runner-compatible)
│   ├── lgbm_tuned/               # Optuna-tuned LightGBM
│   ├── cb_baseline/              # CatBoost (GPU training)
│   └── xgb_baseline/             # XGBoost (GPU training)
├── scripts/                      # 60+ experiment scripts
│   ├── final_train_v2.py         # ★ final training (train + backfill → model_v2)
│   ├── build_submission.py       # ★ delivery packaging (config quantiles, zip)
│   ├── backfill_score.py         # ★ re-score 91 historical submissions on backfill labels
│   └── ...                       # feature engineering / tuning / stacking / diagnostics
├── docs/                         # experiment logs & conclusions (Chinese)
├── public_release_20260630/      # official starter kit (Time-Series API runner)
└── public_release_20260823/      # official label-backfill kit (docs + manifest only)
```

## Reproduction

See [REPRODUCTION.md](REPRODUCTION.md) for the full pipeline. In short:

```bash
# 1. Train the final ensemble (requires train + label-backfill parquet, ~24GB, not committed)
python scripts/final_train_v2.py

# 2. Assemble the delivery package
python scripts/build_submission.py

# 3. Self-check with the official runner
cd public_release_20260630
python timeseries_api/run_timeseries_api.py \
  --data-root data --strategy-dir ../src/final_submission/submit \
  --output /tmp/selfcheck.csv --model-init-timeout-seconds 180
```

## Data

Competition parquet data (~20GB) is **not committed** per the rules. The schema: 13.2M train rows / 3.2M test rows / 15 anonymized assets / 323 anonymized features / 47 responders. See `public_release_20260630/data/manifest.json`.

## Key Findings

- The target behaves like an innovation process: single models cap at ~0.0018 holdout R², temporal models (TCN) score negative, and responders correlate with the target at only 0.056.
- Ensemble gains come from decorrelation, not strength: the weakest model family (MLP) earns the largest weight because it is the least correlated.
- Holdout optimization does not transfer to the public LB (holdout +6% → LB −5% in our worst case); only label-backfill re-scoring resolved model selection reliably.

## Authors

- **Xiangzheng Xi** (School of the Gifted Young, USTC) — strategy design, experiment pipeline, delivery engineering
- Developed in collaboration with an AI agent (autonomous experiment execution under human-set guardrails, see `.agent/`)

## License

[MIT](LICENSE) — competition data and official documentation PDFs are property of the organizer and excluded.
