# LightGBM Baseline（增补包用法）

完整可跑的单模型基线：读训练数据 → 冻结特征 → purged K-fold 选参 → holdout 体检 → 全量重训 → Time-Series API 顺序推理。

用于展示一套合理流程，不代表成绩上限。不要用测试集调参。**不附带预训练权重**，须先训练再推理。

> 本文档面向**独立增补包**：本目录只含策略代码与说明，数据与 `timeseries_api` 来自主包 `public_release_20260630`。

下文 `MAIN` = 主包根目录，`BASE` = 增补包根目录。

## 目录

| 文件 | 作用 |
| --- | --- |
| `train.py` | 训练入口（写出 `model/`） |
| `main.py` | Time-Series API 推理入口（`Model`，读取 `model/`） |
| `data_utils.py` | 按 manifest / 目录列举 parquet，按 `time_id` 采样，构造因果历史特征 |
| `preprocess.py` | 清洗 inf/nan，冻结可用 `feature_*` 列表 |
| `features.py` | 选历史特征列，拼模型输入表 |
| `validation.py` | purged K-fold / holdout / 加权 R² / scale 门禁 |

## 流程（每一步在做什么）

1. **划时间块**  
   按 `time_id` 排序后，尾部 15% 固定为 holdout；其余做 K=5 等块划分。  
   每个 valid 块两侧各空出 30 个 `time_id`（purge），减轻相邻时刻泄漏。

2. **冻结特征 schema**  
   只用最早一折的 train 段做健康检查：去掉常量/过稀特征，清洗非有限值。  
   之后全流程共用这份列集合，避免后面时段反向决定特征。

3. **构造输入**  
   在冻结列上，按与 target 的相关性取 Top-K 列，生成因果 `lag1` / `diff1` / `rmean`（默认窗口 5）。  
   模型输入 = 原始特征 + 这些历史特征 + `asset_id`。

4. **预注册候选 + CV 选型**  
   事先固定 4 组超参（树深 × 正则强弱），不做网格搜索。  
   每组在 5 折上 early stopping；按折均加权 R² 选优，轮数取折均 `best_iteration`。  
   并列时优先更强正则、更少轮数。

5. **holdout 只看一次**  
   用 development（非 holdout）按选定轮数重训，在 holdout 上算一次分数。  
   同时用 OOF 预测估一个 `fitted_oof_scale`（幅度诊断）；**推理不乘这个系数**。  
   门禁检查 OOF/holdout 是否明显异常、scale 是否离谱。

6. **全量重训**  
   门禁通过后，在**全部 train（含 holdout）**上用选定配置与轮数，按 3 个 seed 各训一份，写入 `model/`。

7. **顺序推理**  
   `main.py` 按递增 `time_id` 维护每个 `asset_id` 的历史，拼出与训练一致的特征，三种子预测取平均，直接提交。

## 训练（必做）

`--release-root` 指向主包的 `data/`（含 `manifest.json` 与 `train/`）；若存在 manifest，则按其中 `files.train` 读 parquet，否则读 `train/*.parquet`。

```bash
python "$BASE/examples/lightgbm_baseline/train.py" \
  --release-root "$MAIN/data" \
  --model-dir "$BASE/examples/lightgbm_baseline/model" \
  --num-threads 8
```

全量训练耗时较长（公开 train、`--num-threads 8` 约数小时）。烟测可加：

```bash
python "$BASE/examples/lightgbm_baseline/train.py" \
  --release-root "$MAIN/data" \
  --model-dir /tmp/lgbm_baseline_smoke \
  --num-threads 8 \
  --max-train-rows 300000 \
  --max-valid-rows 100000 \
  --num-boost-round 100
```

产物（写入 `--model-dir`）：

- `model_seed2026.txt` / `model_seed2027.txt` / `model_seed2028.txt`
- `lightgbm_report.json`（特征列表、候选 CV 结果、轮数、门禁等）

## Time-Series API 推理

须先完成训练。在主包中运行 runner；`--data-root` 为主包 `data/`，`--strategy-dir` 指向本策略目录（其下应已有 `model/`）。

```bash
cd "$MAIN"
python timeseries_api/run_timeseries_api.py \
  --data-root data \
  --strategy-dir "$BASE/examples/lightgbm_baseline" \
  --output /tmp/lgbm_baseline_submission.csv
```

也可用环境变量 `LIGHTGBM_BASELINE_MODEL_DIR` 指向其它模型目录。

## 几个刻意的设计

- **验证用时间块 + purge**，不用随机切分，避免偷看邻近未来。
- **候选预先注册**，不用测试集、不用 holdout 反复试参。
- **`prediction_scale` 恒为 1.0**：`fitted_oof_scale` 只写进报告做诊断，不校准提交。
- **最终模型吃满全部标注 train**（含 holdout），与「holdout 只评估一次」配套。

## 依赖

- Python 3.11+
- `numpy==1.24.3`
- `pandas==2.0.3`
- `pyarrow==11.0.0`
- `lightgbm==4.3.0`
