# LightGBM Baseline 流水线验证

> 日期：2026-08-09
> 目的：端到端验证 lightgbm_baseline 的 训练→出模型→推理→提交 全链路（**非真实分数**）
> 机器/环境：同《环境与基线验证》；numpy 1.26.4 / pandas 2.2.3 / lightgbm 4.7.0 / pyarrow 25

## 1. 版本兼容性

源码（`public_release_lightgbm_baseline/examples/lightgbm_baseline/`）与现版本兼容：
- pandas 2.2.3 下 `.to_numpy(copy=True)` 非只读（不触发 linear 那种 CoW 崩溃）
- lightgbm 4.7 的 `lgb.train` / 自定义 `feval` / `lgb.early_stopping` / `lgb.log_evaluation` / `Booster` API 均稳定
- `metric: "None"` + `feval` 仅用自定义加权零均值 R²，正常

## 2. 快速验证策略（缩减数据根）

官方 smoke（`--max-train-rows 300000`）仍会对**全量 13M 行**做预处理与逐折 `_mask_times`，单次约 40-50 分钟。为快速端到端验证，构造缩减数据根 `/tmp/lgbm_tinydata`：

- `manifest.json` 的 `files.train` 仅指向 `train_partition_000.parquet`（符号链接到真实文件，~1.9M 行）
- 其余字段照搬主包 manifest

> ⚠️ 此根**仅用于流水线验证**，分布不均、分数不代表真实水平。

## 3. 训练（smoke，1 分区）

```
train.py --release-root /tmp/lgbm_tinydata --model-dir /tmp/lgbm_baseline_smoke \
  --num-threads 16 --max-train-rows 300000 --max-valid-rows 100000 --num-boost-round 100
```

耗时 ~16 min。purged K-fold（K=5，purge=30）× 4 预注册候选，结果：

| 候选 | mean_fold_score | mean_iterations |
|---|---|---|
| **leaves31_regular（胜出）** | **+0.00196** | 48 |
| leaves63_regular | +0.00192 | 36 |
| leaves31_strong | +0.00189 | 43 |
| leaves63_strong | +0.00184 | 43 |

报告关键值：`oof_raw=+0.00182`、`holdout_raw=+0.00081`、`fitted_oof_scale=0.904`、**`gates_passed=True`**、模型特征数 468（asset_id + 323 原始 + 48 历史×{lag1,diff1,rmean5}=144）。

产物（gitignored）：`model_seed202[678].txt`（各 ~210KB，3 种子集成）+ `lightgbm_report.json`。

**对比**：即便在缩减数据上，lightgbm 也呈正信号（oof/holdout 均 >0），明显优于 linear_window_strategy 的验证集负分。

## 4. 推理（Time-Series API，主包 test 数据）

`LIGHTGBM_BASELINE_MODEL_DIR=/tmp/lgbm_baseline_smoke`，耗时 16m40s：

| 指标 | 实测 | 限制 |
|---|---|---|
| status / rows | ok / 3,217,458 | ✅ |
| model_init（加载 3 个 booster） | 1.77s | ≤180s ✅ |
| 单步平均 | 3.73ms | ≤50ms ✅ |
| 单步最大 | 4.36ms | ≤50ms ✅ |
| 超时次数 | 0 | ✅ |

输出 `/tmp/lgbm_smoke_submission.csv`（96MB）：3,217,458 行，全有限浮点，0 空值，row_id 与官方模板对齐。

> 推理慢的主因：main.py 的 `_history_arrays` 对每行做 Python 循环（deque + vstack）+ 3 booster 集成 predict。单步计时仍合规。生产化时可向量化该循环并按规则固定 num_threads≤4。

## 5. 结论与下一步

- lightgbm_baseline 全链路（训练/门禁/多种子/顺序推理/提交格式）**已验证可用**。
- 真实基线分数需**全量训练**（13M 行、`--num-threads` 充分、700 轮 × 4 候选 × 5 折 + 3 种子最终拟合），README 估"数小时"，本机 128 核预计约 2-5 小时。
- 建议全量训练后台运行（setsid + 日志重定向，避免前台 ssh 超时/SIGPIPE），完成后用其 model 跑推理得真实公榜提交。

## 6. 全量训练结果（真实基线，2026-08-09）

`--num-threads 64`，load~110 争抢下耗时 **10h33m**（独占 64 物理核预计 ~3-4h）。

**结果**：`gates_passed=True`，胜出 `leaves63_regular`，mean_iterations=173。

| 指标 | 值 | 说明 |
|---|---|---|
| mean_fold_score | 0.002116 | 5 折均分 |
| oof_raw | 0.002093 | 全折外预测 |
| **holdout_raw** | **0.001778** | **最诚实的样本外估计（holdout 全程未碰）** |
| fitted_oof_scale | 0.9599 | 门禁 [0.75,1.25] 内 |
| gates | 全过 | OOF>0、holdout>0、scale∈范围 |

候选排名（mean_fold_score）：leaves63_regular 0.002116 > leaves63_strong 0.002113 > leaves31_strong 0.002082 > leaves31_regular 0.002044。最佳候选险胜（差距 3e-6）。

训练全量 13,227,692 行（含 holdout 1,958,574），3 种子集成（model_seed202[678].txt，各 ~1.3MB）。

**推理**（Time-Series API，主包 test）：16m15s
- model_init 2.26s，单步均 3.66ms / 最大 6.03ms，0 超时——全合规（限 50ms/步）
- 输出 `submissions/lgbm_full_submission.csv`（93MB，3,217,458 行，全有限，row_id 与官方模板完全对齐）

**公榜预期**：≈ holdout 0.0018 量级（取决于未来时段是否平稳；金融数据非平稳，可能上下浮动）。

**后续提升方向**（baseline 刻意保守，留有空间）：
1. 利用 47 个 responder 做多任务/去噪目标（baseline 完全未用，最大低垂果实）
2. 扩展因果历史特征（更长窗口、跨标的、波动率/比值类）
3. 认真调参 + LGBM/神经网络集成/stacking
