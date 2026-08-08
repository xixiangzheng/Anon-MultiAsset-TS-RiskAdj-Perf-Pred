# 2026 量化交易研究大赛 项目

本仓库是参赛工作区。代码与文档纳入 git；**赛事数据（parquet / 模型权重 / 大 CSV）一律不跟踪**，详见 `.gitignore`。

## AI 协作规则

修改代码前，先读 `.agent/README.md` 并按其顺序阅读；`.agent/rules.md` 为**硬规则**，必须遵守。

## 快速命令

本地 Time-Series API 顺序推理验证：

```bash
cd public_release_20260630
python timeseries_api/run_timeseries_api.py \
  --data-root data \
  --strategy-dir <策略目录> \
  --output /tmp/submission.csv
```
