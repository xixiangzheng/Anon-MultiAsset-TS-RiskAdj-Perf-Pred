# Linear Window Strategy Demo

这是更接近真实参赛代码结构的示例，展示：

- 读取 `train/*.parquet`；
- 只使用训练阶段可见字段做预处理；
- 按 `time_id` 做时间序列 train/validation 切分；
- 使用 `sklearn.linear_model.Ridge` 训练一个简单线性模型；
- 保存预训练模型、标准化参数和特征列表；
- 在 `main.py` 中加载模型；
- 在 Time-Series API 推理过程中维护历史滑动窗口；
- 使用当前截面特征和历史 rolling 特征生成预测。

训练：

```bash
python examples/linear_window_strategy/train.py \
  --release-root data \
  --model-dir examples/linear_window_strategy/model \
  --window-size 5
```

本地顺序推理：

```bash
python timeseries_api/run_timeseries_api.py \
  --data-root data \
  --strategy-dir examples/linear_window_strategy \
  --output /tmp/linear_submission.csv
```

注意：示例只用于展示平台接口和工程结构，不代表推荐模型复杂度。
训练脚本会把训练期标准化参数、Ridge 截距和系数写入 `model/linear_model.json`；
`main.py` 只加载这个 JSON，不依赖 sklearn，以便推理侧保持轻量。
