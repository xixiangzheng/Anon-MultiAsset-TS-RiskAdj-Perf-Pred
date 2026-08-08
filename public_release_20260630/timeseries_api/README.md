# Time-Series API 本地验证

本目录提供一个轻量本地 runner，用于检查策略目录中的 `main.py` 是否能按
最终评测语义顺序推理。

```bash
python timeseries_api/run_timeseries_api.py \
  --data-root data \
  --strategy-dir timeseries_api \
  --output /tmp/example_submission.csv
```

runner 会生成两类输出：

- `--output` 指定的 CSV submission，只包含 `row_id,target` 两列；
- 标准输出中的 JSON 运行报告，包含 `status`、`rows`、`messages` 和 `timing`。

`timing` 字段用于本地检查推理耗时，结构如下：

```json
{
  "model_init_seconds": 0.0,
  "predict_total_seconds": 0.0,
  "predict_calls": 0,
  "predict_timeout_count": 0,
  "max_predict_seconds": 0.0,
  "mean_predict_seconds": 0.0,
  "total_seconds": 0.0,
  "aborted_after_timeout": false
}
```

可选超时参数：

```bash
python timeseries_api/run_timeseries_api.py \
  --data-root data \
  --strategy-dir timeseries_api \
  --output /tmp/example_submission.csv \
  --per-step-timeout-seconds 0.5 \
  --timeout-policy zero_step
```

`timeout-policy` 可取：

- `zero_step`：单个 time step 超时后，仅该 step 预测填 `0.0`；
- `zero_remaining`：单个 time step 超时后，该 step 及后续所有预测填 `0.0`。
