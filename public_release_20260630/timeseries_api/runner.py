from __future__ import annotations

import importlib.util
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd


VISIBLE_INDEX_COLUMNS = ["row_id", "time_id", "asset_id"]
SUBMISSION_COLUMNS = ["row_id", "target"]


@dataclass(frozen=True)
class RunMessage:
    level: str
    code: str
    message: str
    count: int | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TimingStats:
    model_init_seconds: float = 0.0
    predict_total_seconds: float = 0.0
    predict_calls: int = 0
    predict_timeout_count: int = 0
    max_predict_seconds: float = 0.0
    mean_predict_seconds: float = 0.0
    total_seconds: float = 0.0
    aborted_after_timeout: bool = False

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RunResult:
    status: str
    rows: int
    output_path: str
    messages: list[RunMessage]
    timing: TimingStats

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "rows": self.rows,
            "output_path": self.output_path,
            "messages": [message.as_dict() for message in self.messages],
            "timing": self.timing.as_dict(),
        }


def manifest_files(data_root: str | Path, split: str) -> list[Path]:
    data_root = Path(data_root)
    manifest_path = data_root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest.get("files", {}).get(split, [])
        if files:
            return [data_root / str(rel) for rel in files]
    return sorted((data_root / split).glob("*.parquet"))


def visible_test_frame(frame: pd.DataFrame) -> pd.DataFrame:
    forbidden = {"weight", "target", "timestamp", "symbol"}
    visible_cols = [
        col
        for col in frame.columns
        if col not in forbidden and not str(col).startswith("responder_")
    ]
    missing = [col for col in VISIBLE_INDEX_COLUMNS if col not in visible_cols]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    return frame.loc[:, visible_cols].copy()


def iter_test_slices(data_root: str | Path, split: str = "test") -> Iterator[tuple[int, pd.DataFrame]]:
    carry = pd.DataFrame()
    for path in manifest_files(data_root, split):
        frame = visible_test_frame(pd.read_parquet(path))
        if not carry.empty:
            frame = pd.concat([carry, frame], ignore_index=True)
            carry = pd.DataFrame()
        if frame.empty:
            continue

        previous_group: tuple[int, pd.DataFrame] | None = None
        for time_id, current in frame.groupby("time_id", sort=False):
            if previous_group is not None:
                yield previous_group[0], previous_group[1].reset_index(drop=True)
            previous_group = (int(time_id), current.copy())
        if previous_group is not None:
            carry = previous_group[1]

    if not carry.empty:
        yield int(carry["time_id"].iloc[0]), carry.reset_index(drop=True)


def load_model(strategy_dir: str | Path) -> Any:
    strategy_dir = Path(strategy_dir)
    main_path = strategy_dir / "main.py"
    if not main_path.exists():
        raise ValueError(f"{main_path} does not exist")
    module_name = f"_participant_strategy_{strategy_dir.name}_{abs(hash(main_path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, main_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"could not import {main_path}")
    module = importlib.util.module_from_spec(spec)
    old_path = list(sys.path)
    sys.path.insert(0, str(strategy_dir))
    try:
        spec.loader.exec_module(module)
        model_class = getattr(module, "Model", None)
        if model_class is None:
            raise ValueError("main.py must define class Model")
        return model_class()
    finally:
        sys.path = old_path


def zero_predictions(test: pd.DataFrame) -> np.ndarray:
    return np.zeros(len(test), dtype=np.float64)


def coerce_prediction(raw_prediction: Any, expected_len: int) -> tuple[np.ndarray, int]:
    prediction = np.asarray(raw_prediction, dtype=np.float64)
    if prediction.ndim == 0:
        prediction = prediction.reshape(1)
    if prediction.ndim != 1:
        prediction = prediction.reshape(-1)
    if len(prediction) != expected_len:
        raise ValueError(f"prediction length {len(prediction)} != expected length {expected_len}")
    invalid = ~np.isfinite(prediction)
    invalid_count = int(invalid.sum())
    if invalid_count:
        prediction = prediction.copy()
        prediction[invalid] = 0.0
    return prediction, invalid_count


def zero_submission(data_root: str | Path, split: str) -> pd.DataFrame:
    frames = []
    for _, test in iter_test_slices(data_root, split=split):
        frames.append(pd.DataFrame({"row_id": test["row_id"].to_numpy(), "target": zero_predictions(test)}))
    if not frames:
        return pd.DataFrame(columns=SUBMISSION_COLUMNS)
    return pd.concat(frames, ignore_index=True).loc[:, SUBMISSION_COLUMNS]


def run_loaded_model(
    *,
    model: Any,
    data_root: str | Path,
    strategy_dir: str | Path,
    split: str,
    per_step_timeout_seconds: float | None,
    total_timeout_seconds: float | None,
    timeout_policy: str,
) -> tuple[pd.DataFrame, list[RunMessage], TimingStats]:
    rows: list[pd.DataFrame] = []
    messages: list[RunMessage] = []
    predict_seconds: list[float] = []
    predict_timeout_count = 0
    aborted_after_timeout = False
    run_start = time.perf_counter()
    old_path = list(sys.path)
    sys.path.insert(0, str(Path(strategy_dir)))
    try:
        for time_id, test in iter_test_slices(data_root, split=split):
            if aborted_after_timeout:
                prediction = zero_predictions(test)
                rows.append(pd.DataFrame({"row_id": test["row_id"].to_numpy(), "target": prediction}))
                continue

            elapsed_total = time.perf_counter() - run_start
            if total_timeout_seconds is not None and elapsed_total > total_timeout_seconds:
                prediction = zero_predictions(test)
                aborted_after_timeout = True
                messages.append(
                    RunMessage(
                        "warning",
                        "total_timeout",
                        f"time_id={time_id}: total inference time exceeded {total_timeout_seconds:.6f}s",
                        len(test),
                    )
                )
                rows.append(pd.DataFrame({"row_id": test["row_id"].to_numpy(), "target": prediction}))
                continue

            start = time.perf_counter()
            raw_prediction: Any = None
            error: Exception | None = None
            try:
                raw_prediction = model.predict(test.copy())
            except Exception as exc:
                error = exc
            elapsed = time.perf_counter() - start
            predict_seconds.append(elapsed)

            if per_step_timeout_seconds is not None and elapsed > per_step_timeout_seconds:
                prediction = zero_predictions(test)
                predict_timeout_count += 1
                messages.append(
                    RunMessage(
                        "warning",
                        "predict_timeout",
                        f"time_id={time_id}: predict exceeded {per_step_timeout_seconds:.6f}s",
                        len(test),
                    )
                )
                if timeout_policy == "zero_remaining":
                    aborted_after_timeout = True
                    messages.append(
                        RunMessage(
                            "warning",
                            "timeout_zero_remaining",
                            "remaining time_id predictions filled with 0.0 after timeout",
                            None,
                        )
                    )
            elif error is not None:
                prediction = zero_predictions(test)
                messages.append(RunMessage("warning", "predict_exception", f"time_id={time_id}: {error}", len(test)))
            else:
                try:
                    prediction, invalid_count = coerce_prediction(raw_prediction, len(test))
                except Exception as exc:
                    prediction = zero_predictions(test)
                    messages.append(RunMessage("warning", "predict_exception", f"time_id={time_id}: {exc}", len(test)))
                else:
                    if invalid_count:
                        messages.append(
                            RunMessage(
                                "warning",
                                "invalid_prediction_filled_zero",
                                f"time_id={time_id}: filled non-finite predictions with 0.0",
                                invalid_count,
                            )
                        )
            rows.append(pd.DataFrame({"row_id": test["row_id"].to_numpy(), "target": prediction}))
    finally:
        sys.path = old_path

    timing = TimingStats(
        predict_total_seconds=float(sum(predict_seconds)),
        predict_calls=len(predict_seconds),
        predict_timeout_count=predict_timeout_count,
        max_predict_seconds=float(max(predict_seconds) if predict_seconds else 0.0),
        mean_predict_seconds=float(sum(predict_seconds) / len(predict_seconds) if predict_seconds else 0.0),
        total_seconds=float(time.perf_counter() - run_start),
        aborted_after_timeout=aborted_after_timeout,
    )
    if not rows:
        return pd.DataFrame(columns=SUBMISSION_COLUMNS), messages, timing
    return pd.concat(rows, ignore_index=True).loc[:, SUBMISSION_COLUMNS], messages, timing


def run_strategy(
    *,
    data_root: str | Path,
    strategy_dir: str | Path,
    output_path: str | Path,
    split: str = "test",
    model_init_timeout_seconds: float | None = None,
    per_step_timeout_seconds: float | None = None,
    total_timeout_seconds: float | None = None,
    timeout_policy: str = "zero_step",
) -> RunResult:
    if timeout_policy not in {"zero_step", "zero_remaining"}:
        raise ValueError("timeout_policy must be 'zero_step' or 'zero_remaining'")
    run_start = time.perf_counter()
    model_init_seconds = 0.0
    try:
        init_start = time.perf_counter()
        model = load_model(strategy_dir)
        model_init_seconds = time.perf_counter() - init_start
        if model_init_timeout_seconds is not None and model_init_seconds > model_init_timeout_seconds:
            raise TimeoutError(f"model init exceeded {model_init_timeout_seconds:.6f}s")
    except Exception as exc:
        submission = zero_submission(data_root, split)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(output_path, index=False)
        timing = TimingStats(
            model_init_seconds=float(model_init_seconds),
            total_seconds=float(time.perf_counter() - run_start),
        )
        return RunResult(
            "error",
            int(len(submission)),
            str(output_path),
            [RunMessage("error", "model_init_failed", str(exc), None)],
            timing,
        )

    submission, messages, timing = run_loaded_model(
        model=model,
        data_root=data_root,
        strategy_dir=strategy_dir,
        split=split,
        per_step_timeout_seconds=per_step_timeout_seconds,
        total_timeout_seconds=total_timeout_seconds,
        timeout_policy=timeout_policy,
    )
    timing = TimingStats(
        model_init_seconds=float(model_init_seconds),
        predict_total_seconds=timing.predict_total_seconds,
        predict_calls=timing.predict_calls,
        predict_timeout_count=timing.predict_timeout_count,
        max_predict_seconds=timing.max_predict_seconds,
        mean_predict_seconds=timing.mean_predict_seconds,
        total_seconds=float(time.perf_counter() - run_start),
        aborted_after_timeout=timing.aborted_after_timeout,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    return RunResult("ok", int(len(submission)), str(output_path), messages, timing)
