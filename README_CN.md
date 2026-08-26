# 基于匿名多标的时序数据的风险调整表现预测

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**[English Version](./README.md)** | **中文版**

> 本仓库是 **2026 量化交易研究大赛** 的完整参赛工作区：在匿名多标的时序 panel 数据上预测风险调整目标 `target`，指标为加权零均值 R²，采用 Time-Series API 顺序推理评测（4 核 CPU / 12GB 内存 / 无 GPU / 单步 ≤50ms / 模型加载 ≤180s）。

## 成绩

target 接近 innovation、信号极弱——每 +0.0005 R² 都来之不易。从官方 LightGBM baseline 出发，经过 14 次公榜提交和对 91 个候选文件的标签回补全量复算：

| 阶段 | 加权零均值 R² | 策略 |
|---|---|---|
| 官方 baseline | 0.00313 | LightGBM（323 原始特征）|
| 公榜最优（已提交）| **0.00335** | ratio-LGBM + CatBoost + MLP 集成 |
| **最终交付（holdout 干净 OOF）** | **0.00386** | train + 标签回补重训集成（16.4M 行）|

**最终交付包**：[`release/submission.zip`](release/submission.zip)——自包含（16 个模型权重 + 推理配置），官方 runner 自检 `status=ok`、init 2.3s、4 核绑核单步均值 20.6ms、214,538 次调用零异常。

## 方法

最终预测为三族集成：

```
prediction = 0.358 × ratio_lgb(3 seed) + 0.253 × catboost(3 seed) + 0.389 × nn_emb32(10 seed)
```

| 组件 | 特征 | 说明 |
|---|---|---|
| **ratio_lgb** | 原始 323 + ratio 323 | tuned LightGBM；simple ratio = 全特征 ÷ `feature_157`（唯一公榜验证通过的特征工程）|
| **catboost** | 原始 323 + asset_id | 对称树，与 LightGBM 去相关（corr 0.88）|
| **nn_emb32** | 标准化原始 323 | MLP（512/256/128）+ asset embedding，10-seed 平均 |

三个最关键的决策：

1. **标签回补数据是最强杠杆**。官方发布公开 test 窗口真值（3.2M 行）后，并入训练（共 16.4M 行）使干净 holdout R² 翻倍（0.0017 → 0.0036）——回补段在时段上最接近私榜未来 regime，其价值超过全部模型技巧之和。
2. **简单胜过聪明**。单个手工 ratio（÷ feature_157）公榜 +0.00003；而系统搜索的 top-50 ratio 族反而 −0.00016——搜索过拟合了 partition 特定噪声（经三次公榜失败验证）。
3. **推理鲁棒性是交付生命线**。test 含 1e6 级原始极端值，不加 clip 时 MLP 外推输出达 ~465、分数崩至 −204。推理时全部特征 clip 到 train p0.1/p99.9。

有效/无效技术的完整清单见 [LESSONS.md](LESSONS.md)；逐日实验史见 [docs/](docs/)。

## 仓库结构

```
├── release/submission.zip        # ★ 最终交付包（main.py + 16 权重 + config）
├── src/
│   ├── final_submission/         # 交付策略（Model 类，runner 兼容）
│   ├── lgbm_tuned/               # Optuna 调参 LightGBM
│   ├── cb_baseline/              # CatBoost（GPU 训练）
│   └── xgb_baseline/             # XGBoost（GPU 训练）
├── scripts/                      # 60+ 实验脚本
│   ├── final_train_v2.py         # ★ 最终训练（train+回补 → model_v2）
│   ├── build_submission.py       # ★ 交付打包（config 分位数、zip）
│   ├── backfill_score.py         # ★ 回补标签复算 91 个历史提交
│   └── ...                       # 特征工程/调参/stacking/诊断
├── docs/                         # 实验记录与结论
├── public_release_20260630/      # 官方主包（Time-Series API 评测工具）
└── public_release_20260823/      # 官方回补包（仅文档 + manifest）
```

## 复现

完整流程见 [REPRODUCTION.md](REPRODUCTION.md)。简要：

```bash
# 1. 训练最终集成（需 train + 回补 parquet，约 24GB，不入库）
python scripts/final_train_v2.py

# 2. 组装交付包
python scripts/build_submission.py

# 3. 官方 runner 自检
cd public_release_20260630
python timeseries_api/run_timeseries_api.py \
  --data-root data --strategy-dir ../src/final_submission/submit \
  --output /tmp/selfcheck.csv --model-init-timeout-seconds 180
```

## 数据

赛事 parquet 数据（约 20GB）按规则**不入库**。Schema：训练 1,322 万行 / 测试 321 万行 / 15 匿名标的 / 323 匿名特征 / 47 responder。见 `public_release_20260630/data/manifest.json`。

## 核心发现

- target 表现类似 innovation 过程：单模型 holdout ≤0.0018，时序模型（TCN）为负，responder 与 target 相关仅 0.056
- 集成增益来自去相关而非强度：最弱的模型族（MLP）因相关性最低获得最大权重
- holdout 优化不迁移到公榜（最差一次 holdout +6% → 公榜 −5%）；只有标签回补复算才可靠地解决了模型选择问题

## 作者

- **奚项正**（中国科学技术大学少年班学院）——策略设计、实验管线、交付工程
- 与 AI agent 协作完成（人类设定护栏下的自主实验执行，见 `.agent/`）

## 许可

[MIT](LICENSE)——赛事数据与官方文档 PDF 归主办方所有，不在此许可范围内。
