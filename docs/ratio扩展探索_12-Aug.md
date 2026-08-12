# Ratio 特征扩展 + 多模型集成探索（2026-08-12 自主）

> 起点：公榜 0.00334513（ens_ratio_nn30） → 目标 0.0037
> 结论：**holdout 从 0.00196 → 0.00208（+6%），公榜估计 0.0035-0.0036；0.0037 仍差 0.0001-0.0002**

## 关键突破

### 1. ratio_top50 系统化（feature_157 等最佳分母）
- 扫描 323 特征作为分母，找 top-5 最佳分母（feature_157 等）
- 构造 top-50 ratio 交互特征，加入训练

### 2. ★ ratio_cb 突破（+0.00019 holdout）
| 模型 | raw holdout | +ratio holdout | Δ |
|---|---|---|---|
| LGBM | 0.00170 | 0.00176 | +0.00006（微弱）|
| **CatBoost** | 0.00163 | **0.00182** | **+0.00019 ⭐** |
| XGBoost | 0.00158 | 0.00148 | -0.00010（有害）|
| NN(MLP) | 0.00154 | 0.00140 | -0.00014（有害）|

**洞察**：ratio 特征只对 **CatBoost**（对称树/oblivious）有效。LGBM/XGB（leaf-wise）和 NN 已在内部学习交互，显式 ratio 反而干扰。

### 3. 14-16 模型集成 holdout 0.002082
- 权重高度集中：ratio_nn_emb32(0.39) + ratio_cb(0.35) + ratio_cb_deep(0.19) + xgb(0.07)
- 手设权重 c6 holdout 0.002078（≈SLSQP 0.002082）→ **无 holdout 过拟合**

## 失败的方向（已穷尽）

| 方向 | 结果 | 原因 |
|---|---|---|
| ratio_xgb / ratio_nn | holdout 下降 | GBDT/NN 自学交互 |
| ratio_cb_tuned (tuned参数+ratio) | 0.00178（<ratio_cb 0.00182）| tuned参数为raw调，+ratio过拟合 |
| ratio_cb_deep (depth=10+ratio) | 0.00169 | 过拟合 |
| **sum交互 (A+B)** | 单特征相关 0.0359（>ratio 0.0338）但加入模型无效 | 模型已自学线性组合 |
| **三方交互 (A/B)*C** | 仅2个显著正向，加入无增益 | ratio 已捕获大部分 |
| **大 NN (1024,512,256)** | holdout 0.00119-0.00131（<小NN 0.00158）| 过拟合 |
| **ExtraTrees / RF** | holdout 0.00028 | 随机分割不适合弱信号 |
| **Stacking 元学习器** | 0.001904（<linear 0.002045）| 过拟合，5-fold CV确认 |

## 候选提交清单（submissions/，按 holdout 排序）

| 文件 | holdout R² | 说明 | 推荐度 |
|---|---|---|---|
| **mega2_ens.csv** | 0.002083 | 16模型 SLSQP 优化 | ★★★ 首选 |
| **mega_ens.csv** | 0.002082 | 14模型 SLSQP 优化 | ★★★ |
| robust_c6_cb_nn_balance.csv | 0.002078 | 手设权重（防过拟合）| ★★ 稳健备选 |
| final_ens.csv | 0.002043 | 11模型 | ★ |
| ens_ratio_nn30.csv | 公榜 0.00334513 | 已验证公榜（旧最优）| 参考锚点 |

## 0.0037 不可达的根因

1. **target 是 innovation-like 噪声墙**（用户之前已确认）
2. **特征工程饱和**：单特征相关性提升（0.031→0.036）不转化为模型 holdout 提升
3. **模型多样性饱和**：16 模型线性集成 holdout 仅比 7 模型高 +0.0001
4. **Stacking/大 NN 都过拟合**

**估计**：mega2_ens.csv 公榜 0.0035-0.0036（+0.0002 over 当前最优 0.00334513）。

## 技术产出（已保存）
- runs/ratio_top50.json — top-50 ratio 交互（最佳分母 feature_157）
- runs/nonratio_interactions.json — sum/diff/product/max 交互搜索（sum 最强）
- runs/three_way_search.json — 三方交互搜索
- runs/oof_all.pkl, ratio4_oof.pkl, ratio_v2_oof.pkl, ratio_sum_oof.pkl — 各模型 OOF
- submissions/mega2_ens.csv, mega_ens.csv, robust_*.csv — 集成候选

## ★ 最终重要发现：ratio 信号是 partition-specific（解释饱和）

最后一个实验 ratio_cb150 揭示了关键事实：
- 用 part0+1 搜 ratio → top ratio 相关 0.034，最佳分母 feature_157
- 用 part3+4 搜 ratio → top ratio 相关 **仅 0.021**，最佳分母 feature_159（完全不同）

**ratio 信号在不同分区不一致**，这解释了为什么：
1. ratio_cb (+0.00019) 有效但增益有限——ratio 部分是 partition-specific 的过拟合
2. 加更多 ratio（top-150）反而下降——更多 partition-specific 噪声
3. sum/diff 交互无效——它们也是 partition-specific 的

**这对公榜的影响**：
- ratio_cb 的 holdout 提升可能部分不迁移到公榜（公榜是新数据）
- 但 ratio_cb 全量训练（含所有 partition），有部分真实信号
- 保守估计：mega2_ens.csv 公榜 0.0034-0.0036（不是 0.0037+）

## 最终提交建议（按推荐度）

| 优先级 | 文件 | holdout | 预期公榜 | 风险 |
|---|---|---|---|---|
| 1 | **mega2_ens.csv** | 0.002083 | 0.0034-0.0036 | holdout 优化可能微过拟合 |
| 2 | robust_c6_cb_nn_balance.csv | 0.002078 | 0.0034-0.0035 | 手设权重，最稳健 |
| 3 | mega_ens.csv | 0.002082 | 0.0034-0.0036 | 同 mega2（14 模型）|
| 4 | final_ens.csv | 0.002043 | 0.0033-0.0035 | 11 模型，保守 |
| 参考 | ens_ratio_nn30.csv | ~0.00198 | **0.00334513** ⭐已验证 | 旧最优 |

**建议**：第 1 次提交 mega2_ens.csv；若不如 ens_ratio_nn30，第 2 次提交 robust_c6（更稳健）。

## 关于 0.0037 目标的诚实评估

**0.0037 在当前数据/方法下不可达**：
- holdout 从 ~0.00196 → 0.002083（+6%，已显著）
- 但 0.0037 需 holdout ~0.0022（还需 +6%）
- 所有合理方向已穷尽（特征工程、模型多样性、stacking、NN 扩展）
- 根因：target 是 innovation 噪声墙 + ratio 信号 partition-specific

**真实可达范围**：0.0034-0.0036（+0.0001~0.0003 over 当前公榜最优）

## ★★★ 重要教训：mega2_ens 公榜失败（2026-08-12 14:00）

**事实**：mega2_ens.csv holdout=0.002083（最高）但公榜 **0.003186**（比 ens_ratio_nn30 0.003345 **差 -0.00016**）！

### 根因诊断：ratio_cb 类模型晚段 leak

把 holdout 按 time_id 分早晚两段，发现：

| 模型 | 早段(似公榜) | 晚段(可能含leak) | Δ(晚-早) |
|---|---|---|---|
| **raw_cb** | 0.00176 | 0.00150 | -0.00025（正常）|
| **ratio_cb** | 0.00169 | 0.00194 | **+0.00025（leak!）** |
| **ratio_cb_deep** | 0.00157 | 0.00181 | **+0.00024（leak!）** |
| **nn3_emb32** | 0.00178 | 0.00130 | -0.00049（正常）|
| ratio_nn_emb32 | 0.00172 | 0.00143 | -0.00029 |
| **ratio_lgb** | 0.00180 | 0.00171 | -0.00010（稳健）|

**结论**：ratio_cb / ratio_cb_deep 在早段（似公榜）**不如 raw_cb**！ratio 提升 CB是晚段 leak 的假象。
ratio 特征用 part0+1 搜，holdout 含 part0+1 末段 → ratio_cb 在 holdout 上虚高。

### mega2 失败原因
权重 ratio_nn_emb32(0.39) + ratio_cb(0.35) + ratio_cb_deep(0.19) 全是晚段 leak模型，公榜必然差。

### 修正：用早段 holdout 优化权重（submissions/early_ens_*.csv）

只用**早段稳健模型**（lgb/cb/xgb/nn3_emb32/ratio_lgb），避免 ratio_cb 类：

| 候选 | 早段 R² | 全段 R² | 配方 |
|---|---|---|---|
| **early_ens_A_raw_only** | 0.002055 | 0.001971 | lgb:0.24, cb:0.19, xgb:0.13, nn3:0.44 |
| early_ens_H_rlgb_cb_xgb_nn | 0.002049 | 0.001985 | ratio_lgb:0.15, cb:0.22, xgb:0.17, nn3:0.46 |

### 提交建议（用户今天剩 4 次）

1. **优先**：（ratio_lgb+raw_cb+xgb+nn1+nn3，与 ens_ratio_nn30 同思路+xgb 多样性）
2. **次选**：（纯 raw，最稳健）
3. **参考**：ens_ratio_nn30.csv 已验证 0.003345（如果以上都不行，回到它）

### 教训总结
1. **holdout 全段优化会过拟合**：必须用早段（似公榜）验证
2. **特征工程 + 不同模型 ≠ 通用提升**：ratio 只对 LGBM 稳健，对 CB 是 leak
3. **公榜是唯一真理**：holdout 提升 +6% 可能公榜反而 -5%
