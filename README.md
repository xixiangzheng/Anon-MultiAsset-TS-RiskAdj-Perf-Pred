# Risk-Adjusted Performance Prediction for Anonymous Multi-Asset Time-Series Data

# 基于匿名多标的时序数据的风险调整表现预测

> 2026 量化交易研究大赛参赛项目 · 从 baseline 0.00313 到最终交付集成 holdout R² 0.003864

预测匿名多标的时序数据的风险调整目标 `target`，评分指标为加权零均值 R²（Weighted Zero-Mean R²），评测采用 Time-Series API 顺序推理（4 核 CPU / 12GB / 无 GPU / 单步 ≤50ms / 模型加载 ≤180s）。

## 🏆 最终成果

| 阶段 | 成绩 | 说明 |
|---|---|---|
| 官方 baseline | 0.00313 | LightGBM 公开基线（rank 71）|
| 公榜最优 | **0.00335** | `ens_ratio_nn30`：ratio_lgb + CatBoost + NN 三方集成 |
| **私榜交付（holdout）** | **0.00386** | v2 集成：train + 回补数据重训，超公榜最优 +15% |

**交付包**：[`release/submission.zip`](release/submission.zip)（23MB，含 16 个模型权重，md5 `efe9f3b8`）

## 📐 最终策略架构

```
prediction = 0.358 × ratio_lgb(3seed) + 0.253 × catboost(3seed) + 0.389 × nn_emb32(10seed)
```

| 组件 | 特征 | 说明 |
|---|---|---|
| **ratio_lgb** | raw 323 + ratio 323 | tuned LGBM；simple ratio = 全特征 ÷ feature_157（公榜验证的真信号）|
| **catboost** | raw 323 + asset_id | 对称树，与 LGBM 去相关（corr 0.88）|
| **nn_emb32** | 标准化 raw 323 | MLP(emb32, 512/256/128) + asset embedding，10-seed 平均 |

**关键工程决策**：
- **回补数据并入训练**（13.2M + 3.2M = 16.4M 行）：回补段是最接近私榜的未来分布，holdout 提升翻倍（0.0017 → 0.0036）
- **推理防爆**：test 存在 1e6 级极端值，NN 线性外推曾致输出爆炸（复算 -204）；修复 = raw clip 到 train p0.1/p99.9
- **合规**：全相对路径 / num_threads=4 / 无历史状态（负 row_id 行自然兼容）/ 纯 CPU

## 📁 仓库结构

```
├── release/submission.zip        # ★ 最终交付包（main.py + 16权重 + config）
├── src/
│   ├── final_submission/         # 交付策略（main.py 可直接评测导入）
│   ├── lgbm_tuned/               # tuned LGBM（Optuna 搜索）
│   ├── cb_baseline/              # CatBoost GPU 训练
│   └── xgb_baseline/             # XGBoost GPU 训练
├── scripts/                      # 全部实验脚本（62 个）
│   ├── final_train_v2.py         # ★ 最终训练（train+回补 → model_v2）
│   ├── build_submission.py       # ★ 交付包组装（config 分位数/打包）
│   ├── backfill_score.py         # ★ 回补真值复算 91 个历史提交
│   ├── ratio_models.py           # ratio 特征 × 4 模型族
│   ├── ensemble_weight_opt.py    # 多模型 SLSQP 权重优化
│   └── ...                       # 特征工程/调参/stacking/诊断
├── docs/                         # 实验记录与结论（中文）
│   ├── 私榜交付总结_26-Aug.md     # ★ 交付全流程 + bug 修复记录
│   ├── 探索日志.md                # 完整探索史（30+ 实验）
│   ├── 公榜提交记录.md            # 14 次公榜提交全表
│   └── ratio扩展探索_12-Aug.md    # ratio 特征工程完整教训
├── public_release_20260630/      # 官方主包（评测工具 timeseries_api/）
└── public_release_lightgbm_baseline/  # 官方 LGBM 基线包
```

## 🚀 复现流程

```bash
# 1. 环境依赖见 release zip 内 requirements.txt
pip install lightgbm catboost torch numpy pandas

# 2. 最终模型训练（需 train + 回补数据，~20GB，不入库）
python scripts/final_train_v2.py          # 产出 src/final_submission/model_v2/

# 3. 组装交付包
python scripts/build_submission.py        # 产出 submit/submission.zip

# 4. 本地评测自检（官方 Time-Series API）
cd public_release_20260630
python timeseries_api/run_timeseries_api.py \
  --data-root data --strategy-dir ../src/final_submission/submit \
  --output /tmp/selfcheck.csv --model-init-timeout-seconds 180
```

自检基准（4 核）：`status=ok` / `init 2.3s` / `mean_predict 20.6ms` / 214,538 步零异常。

## 📊 数据

赛事 parquet 约 20GB，**不纳入 git**（见 `public_release_20260630/data/manifest.json`）：
训练 1,322 万行 / 公开测试 321 万行 / 15 标的 / 323 匿名特征 / 47 responder。

## 🔬 核心研究发现

1. **target 接近 innovation（噪声墙）**：单模型 holdout ≤0.0018，时序模型（TCN）为负，responder 与 target 相关仅 0.056
2. **simple ratio 是真信号，复杂 ratio 是噪声**：全特征÷157 公榜 +0.00003；top-50 筛选的 ratio 是 partition-specific 过拟合（12-Aug 三次公榜失败验证）
3. **holdout 优化 ≠ 公榜提升**：holdout +6% 的集成公榜 -5%；公榜是唯一真理，回补真值复算是最终解
4. **集成增益来自去相关**：NN 与 GBDT corr 0.62-0.75，优化器奖励多样性 > 单模型强度
5. **回补数据是最强杠杆**：并入训练使 holdout 翻倍，且时段分布最接近私榜

完整实验史（30+ 实验、正负结论）见 `docs/`。

## 🤖 AI 协作

本项目由人机协作完成：策略设计/实验执行/交付工程由 AI agent 自主推进，关键决策（提交选择、方向取舍）由人类把关。`.agent/` 为协作控制面（硬规则：时序合规、数据不入库、仓库整洁）。
