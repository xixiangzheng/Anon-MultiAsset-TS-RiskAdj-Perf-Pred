# 复现指南

本文档说明如何从零复现最终交付策略。

## 0. 前置条件

- **数据**（不入库，需从赛事方获取）：
  - 主公开包 `public_release_20260630/`（train 13.2M 行 + test 3.2M 行，~20GB）
  - 回补包 `public_release_20260823/`（test 的标签回填，3.2M 行，~4GB）
- **硬件**：64GB+ 内存（全量训练 16.4M×647 特征）；GPU 可选（CatBoost/NN 提速，LGBM 纯 CPU）
- **依赖**：见 `release/submission.zip` 内 requirements.txt（lightgbm 4.7.0 / catboost 1.2.10 / torch 2.13 / numpy 1.26.4 / pandas 2.2.3）

## 1. 最终模型训练

```bash
# v2 = train(13.2M) + 回补(3.2M)，holdout = 最后 15% 时段（回补尾部）
python scripts/final_train_v2.py
```

产出 `src/final_submission/model_v2/`：
- `ratio_lgb_seed{2026,2027,2028}.txt`（tuned LGBM + ratio 特征）
- `cb_seed{2026,2027,2028}.cbm`（CatBoost GPU 训练）
- `nn_emb32_seed{2026..2035}.pt`（10-seed MLP）
- `v2_report.json`（holdout 成绩 + SLSQP 最优配比）

预期 holdout（干净 OOF，未见最新时段）：

| 模型 | R² |
|---|---|
| ratio_lgb | 0.00362 |
| catboost | 0.00351 |
| nn_emb32 | 0.00343 |
| **集成** | **0.00386** |

> 注意：需要先配置脚本内 `DATA_ROOT` / `BACKFILL` 路径与 GPU 设备号。无 GPU 时 CatBoost 段需改 `task_type="CPU"`（训练时间从分钟级升至小时级）。

## 2. 组装交付包

```bash
python scripts/build_submission.py
```

产出 `src/final_submission/submit/submission.zip`：
- 从 train+backfill 采样重算推理 config（ratio clip 分位数 / NN 标准化参数 / raw clip p0.1-p99.9）
- 拷贝 16 个权重 + main.py + requirements.txt 并打包

## 3. 本地评测自检

```bash
cd public_release_20260630
python timeseries_api/run_timeseries_api.py \
  --data-root data \
  --strategy-dir ../src/final_submission/submit \
  --output /tmp/selfcheck.csv \
  --model-init-timeout-seconds 180
```

通过标准：`status=ok`、`messages=[]`、`init<180s`、`mean_predict<50ms`。
实测（4 核绑核）：init 2.3s / mean 20.6ms / 214,538 步零异常。

## 4.（可选）复算公榜 test 真值分数

```bash
python scripts/backfill_score.py    # 复算 submissions/*.csv
```

## 5. 关键实验复现入口

| 实验 | 脚本 | 结论 |
|---|---|---|
| ratio × 4 模型族 | `scripts/ratio_models.py` | 只有 CB 受益（后被证明是 leak）|
| 交互特征搜索 | `scripts/three_way_search.py` / `nonratio_search.py` | sum 相关高但模型无效 |
| 多模型权重优化 | `scripts/ensemble_weight_opt.py` / `fast_ensemble.py` | holdout 优化公榜失败 |
| 早段/晚段诊断 | 见 `docs/ratio扩展探索_12-Aug.md` | ratio_cb 提升 = 晚段 leak |
| 大 NN / stacking / ET | `ratio_nn_big.py` / `stacking_cv.py` / `et_rf.py` | 全部过拟合或过弱 |
