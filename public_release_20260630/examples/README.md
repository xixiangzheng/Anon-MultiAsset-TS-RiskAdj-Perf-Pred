# 示例代码

本目录提供数据读取、随机基线和简单线性基线示例。

这些示例用于展示完整代码结构，不代表推荐建模上限。

## 运行策略示例

随机策略：

```bash
python examples/random_strategy/train.py \
  --release-root data \
  --model-dir examples/random_strategy/model

python timeseries_api/run_timeseries_api.py \
  --data-root data \
  --strategy-dir examples/random_strategy \
  --output /tmp/random_submission.csv
```

线性滑动窗口策略：

```bash
python examples/linear_window_strategy/train.py \
  --release-root data \
  --model-dir examples/linear_window_strategy/model \
  --window-size 5

python timeseries_api/run_timeseries_api.py \
  --data-root data \
  --strategy-dir examples/linear_window_strategy \
  --output /tmp/linear_submission.csv
```
