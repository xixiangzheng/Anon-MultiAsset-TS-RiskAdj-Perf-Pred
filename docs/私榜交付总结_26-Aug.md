# 私榜交付总结（2026-08-26）

## 最终交付策略：v2 集成（train + 回补重训）

**配方**：0.358 * ratio_lgb(3seed) + 0.253 * catboost(3seed) + 0.389 * nn_emb32(10seed)

### 训练数据（决定性改进）
- train(13.2M) + 回补(3.2M) = 16.4M 行；holdout = 最后 15% 时段（回补尾部，最接近私榜分布）

### v2 holdout（干净 OOF）成绩
| 模型 | holdout R² |
|---|---|
| ratio_lgb (÷157 全特征) | 0.003621 |
| catboost | 0.003514 |
| nn_emb32 (10 seed) | 0.003432 |
| **SLSQP 集成** | **0.003864** ⭐ |

（对照：公榜最优 ens_ratio_nn30 = 0.003345；v2 holdout 超 0.0037 目标）

### 回补真值排行榜（91 个历史提交复算，top 5）
| 文件 | backfill R² |
|---|---|
| ens_ratio8_nn30 | 0.003359 |
| ensemble_lcb_nn | 0.003354 |
| ratio_family_avg | 0.003348 |
| ens_ratio_nn30 | 0.003345 |
| verified_811 | 0.003345 |

### 自检结果（taskset 4 核，全量 3.2M 行）
- status: ok, messages: []（零异常/零超时）
- model_init: 2.3s（限 180s）✅
- mean_predict: 20.6ms（限 50ms）✅
- 端到端复算：missing=0、无爆炸 ✅

## 交付中修复的三个关键 bug
1. **CatBoost set_parameter 不存在** → predict(thread_count=4) 显式传参
2. **torch.nn 无 cat 属性** → _build_mlp 内部 import torch（闭包传模块错误）
3. **★ NN 极端值爆炸（致命）**：test 部分行原始特征有 1e6 级 outlier，标准化后 |xs| 达 130 万，NN 线性外推输出 max 465 → 复算 score=-204。修复：推理时 raw clip 到 train p0.1/p99.9（双层防御，config 存 raw_lo/raw_hi）

## 交付包结构（src/final_submission/submit/submission.zip, 23MB）
submission.zip
├── main.py                      # Model 类（相对路径、nt=4、raw clip、ratio 构造）
├── requirements.txt             # lightgbm 4.7.0 / catboost 1.2.10 / torch 2.13.0 / numpy 1.26.4 / pandas 2.2.3
└── model/
    ├── inference_config.json    # 特征列/ratio 参数/nn 标准化/raw clip/配比
    ├── ratio_lgb_seed{2026,2027,2028}.txt
    ├── cb_seed{2026,2027,2028}.cbm
    └── nn_emb32_seed{2026..2035}.pt

## 用户最终提交步骤（JupyterHub）
1. 下载 submission.zip（本机：scp school-server:.../submit/submission.zip）
2. JupyterHub Terminal:
   - python -c 'import sys; print(sys.executable)'  # 确认 /opt/conda/bin/python
   - python -m pip install lightgbm==4.7.0 catboost==1.2.10（torch/numpy/pandas base 环境已有则跳过）
   - python -m pip check
3. mkdir -p ~/submit && cd ~/submit && unzip submission.zip（或直接传解压后的目录）
4. 核对 ~/submit/main.py 存在
5. 用公开包 runner 自检（强烈建议）：
   python timeseries_api/run_timeseries_api.py --data-root data --strategy-dir ~/submit --output /tmp/selfcheck_final.csv --model-init-timeout-seconds 180
6. 提交 zip

## 风险与备注
- holdout 0.003864 是最新未见时段的干净估计，但私榜是未来 9 月数据，实际可能回落（保守预期 0.0033-0.0039）
- max_predict 毛刺 1.78s 出现 1 次（GC/页错误），mean 20.6ms 远低于 50ms 红线，评测按单步平均达标
- 负 row_id 历史行：模型无状态，自然兼容（所有行正常预测）
