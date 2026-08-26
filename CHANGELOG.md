# Changelog

## 2026-08-26 · 私榜交付（v2 最终版）

- **交付 `release/submission.zip`**：0.358·ratio_lgb + 0.253·catboost + 0.389·nn_emb32(10seed)
- 回补包 `public_release_20260823`（3.2M 行带标签）并入训练 → holdout **0.003864**（超 0.0037 目标）
- 回补真值复算 91 个历史提交：top = `ens_ratio8_nn30` 0.003359
- 交付工程：修复 CatBoost thread_count / torch.nn 闭包 / **NN 极端值爆炸**（raw clip p0.1/p99.9）
- 全量自检（服务器 + JupyterHub 双端）：status ok / init 2.3s / mean 20.6~29ms@4核 / 零异常
- JupyterHub 正式提交完成

## 2026-08-25 · 交付准备

- 解析赛事交付公告，规划私榜冲刺任务（8/31 截止）
- 版本A重训（无回补）：ratio_lgb 0.00171 / nn_emb32 0.00149
- 交付 main.py 初版 + build_submission.py 组装管线

## 2026-08-12 · ratio 扩展探索（失败但有价值）

- ratio 特征 × 4 模型族（LGBM/CB/XGB/NN）+ 三方交互 + sum/diff 交互搜索
- 14-16 模型集成 holdout 0.002083，但公榜 **0.003186（严重过拟合）**
- 三次公榜提交全部失败，确认 top-50 ratio 为 partition-specific 噪声
- 教训沉淀：holdout 优化 ≠ 公榜提升；simple ratio(÷157) 才是真信号

## 2026-08-11 · 公榜突破 0.003345

- **ratio 特征（feature÷157）LGBM**：公榜 0.003345，首个验证通过的特征工程突破
- 伪标签 + winsorize：0.003320
- top-100 特征选择：迁移正向但公榜 0.003275
- NN 多 seed 稳定化：弱 NN 翻倍强化

## 2026-08-10 · 多模型集成体系

- CatBoost GPU 训练（10 分钟级）+ XGBoost + 5-7 模型集成
- `ensemble_lcb_nn30` = 0.003313（当日最优）
- NN 变体实验（dropout/emb/deep）+ Optuna 调参

## 2026-08-09 · baseline 验证与初步探索

- LightGBM baseline 公榜 0.00313（rank 71）
- responder 方向验证为死路（corr 0.056）
- 时序合规管线搭建（purged CV）

## 2026-08-08 · 项目启动

- 环境验证、数据探查、官方工具链跑通
