# Risk-Adjusted Performance Prediction for Anonymous Multi-Asset Time-Series Data

# 基于匿名多标的时序数据的风险调整表现预测

> 2026 量化交易研究大赛参赛工作区。

预测匿名多标的时序数据的风险调整目标 `target`，评分指标为加权零均值 R²，评测采用 Time-Series API 顺序推理。

## 目录
- `public_release_20260630/` — 官方主包：数据（未跟踪）、文档、示例策略、本地评测工具 `timeseries_api/`
- `public_release_lightgbm_baseline/` — LightGBM 完整基线增补包（无数据、无权重，需先训练）
- `src/` — 自研策略代码
- `scripts/` — 实验脚本（特征工程、模型训练、集成优化）
- `docs/` — 实验结论与笔记
- `.agent/` — AI 协作控制面（`rules.md` 为硬规则）

## 数据
赛事 parquet 数据约 20GB，**不纳入 git**。见 `public_release_20260630/data/manifest.json`：训练 1322 万行 / 测试 321 万行 / 15 标的 / 323 特征 / 47 responder。

## 快速开始
```bash
cd public_release_20260630
# 跑通随机基线，验证环境
python examples/random_strategy/train.py --release-root data --model-dir examples/random_strategy/model
python timeseries_api/run_timeseries_api.py --data-root data --strategy-dir examples/random_strategy --output /tmp/random_submission.csv
```

详见 `public_release_20260630/docs/`。
