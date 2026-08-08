# 示例代码

本增补包仅含 LightGBM 完整基线（**无预训练权重**），需配合主包 `public_release_20260630` 的 `data/` 与 `timeseries_api/` 使用。请先训练再推理。

设 `MAIN` = 主包根目录，`BASE` = 本增补包根目录：

```bash
python "$BASE/examples/lightgbm_baseline/train.py" \
  --release-root "$MAIN/data" \
  --model-dir "$BASE/examples/lightgbm_baseline/model" \
  --num-threads 8

cd "$MAIN"
python timeseries_api/run_timeseries_api.py \
  --data-root data \
  --strategy-dir "$BASE/examples/lightgbm_baseline" \
  --output /tmp/lgbm_baseline_submission.csv
```

详见增补包根目录 `README.md` 与 `lightgbm_baseline/README.md`。
