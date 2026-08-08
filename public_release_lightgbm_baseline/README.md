# LightGBM Baseline 增补包

本目录是对已发布主包 `public_release_20260630` 的**独立增补**：只提供 LightGBM 完整基线示例
（代码与教程）。**不包含**数据、Time-Series API runner，也**不附带预训练权重**。
请与主包一起使用，并先自行训练再推理。

## 目录

```text
examples/
  README.md
  lightgbm_baseline/     # 策略代码与 README（无 model/）
```

## 与主包如何配合

将本增补包与主包解压到同一层级（或记住各自绝对路径）。下文用：

- `MAIN`：主包根目录（含 `data/`、`timeseries_api/`）
- `BASE`：本增补包根目录

### 1. 训练（必做）

```bash
python "$BASE/examples/lightgbm_baseline/train.py" \
  --release-root "$MAIN/data" \
  --model-dir "$BASE/examples/lightgbm_baseline/model" \
  --num-threads 8
```

全量训练耗时较长（约数小时）。训练完成后才会生成 `model/`。

### 2. Time-Series API 推理

在主包中调用 runner，策略目录指向本增补包（其下需已有 `model/`）：

```bash
cd "$MAIN"
python timeseries_api/run_timeseries_api.py \
  --data-root data \
  --strategy-dir "$BASE/examples/lightgbm_baseline" \
  --output /tmp/lgbm_baseline_submission.csv
```

## 依赖

- Python 3.11+
- `numpy==1.24.3`
- `pandas==2.0.3`
- `pyarrow==11.0.0`
- `lightgbm==4.3.0`

策略流程与设计说明见 `examples/lightgbm_baseline/README.md`。
