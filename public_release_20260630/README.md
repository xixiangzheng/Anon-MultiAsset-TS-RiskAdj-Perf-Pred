# 2026 量化交易研究大赛公开发布包

本目录是面向参赛者的公开 release。数据、文档和示例代码均已匿名化。

## 目录结构

```text
data/
  manifest.json
  train/train_partition_*.parquet
  test/test_partition_*.parquet
  sample_submission.csv
docs/
  competition_description.md
  data_description.md
examples/
  data_io/
  random_strategy/
  linear_window_strategy/
timeseries_api/
  runner.py
  run_timeseries_api.py
  example_main.py
  main.py
  README.md
```

## 本地 Time-Series API 验证

```bash
python timeseries_api/run_timeseries_api.py \
  --data-root data \
  --strategy-dir timeseries_api \
  --output /tmp/example_submission.csv
```
