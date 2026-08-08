# 硬规则（不可违反）

## 数据与版本控制
1. **赛事数据绝不入 git**：parquet、`sample_submission.csv`、训练出的 `model/` 已在 `.gitignore`。如需提交小数据集，用 `git add -f` 并确认体积。
2. 不得将任何数据文件复制进被跟踪的代码目录。

## 时序合规（赛题核心）
3. **严禁使用未来信息**：特征工程、标准化、切分中，当前 `time_id` 的预测只能用当前及历史可见字段。不得读取未来 `feature_*`、`responder_*`、`target`、`weight`。
4. 验证必须按 `time_id` 时间块切分（推荐 purge），禁止随机切分。

## 工程约束（评测环境：4 核 / 12GB / 无 GPU / 无外网）
5. 推理必须显式设置最大线程数（本地测试 ≤4，私榜 ≤8），如 `lightgbm.Booster().predict(x, num_threads=4)`。
6. `Model.predict(test)` 必须返回长度等于 `len(test)` 的一维有限浮点数数组；任何异常/超时对应预测会被置 0。
7. 模型加载 ≤180s，单步平均推理 ≤50ms。提交前必须用 `timeseries_api/runner.py` 本地跑通。

## 仓库整洁
8. 根目录只放 `AGENTS.md`、`README.md`、`.gitignore`、`.agent/`。`.py` 放 `src/` 或 `scripts/`。
9. 实验结论沉淀到 `docs/`，不要只留在 `runs/` 的日志里。
