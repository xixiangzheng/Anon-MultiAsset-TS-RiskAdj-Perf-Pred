from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runner import run_strategy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a strategy with the local Time-Series API demo.")
    parser.add_argument("--data-root", required=True, help="Directory containing manifest.json, train/, test/.")
    parser.add_argument("--strategy-dir", required=True, help="Directory containing participant main.py.")
    parser.add_argument("--output", required=True, help="CSV output path with row_id,target columns.")
    parser.add_argument("--split", choices=["test", "train"], default="test")
    parser.add_argument("--model-init-timeout-seconds", type=float, default=None)
    parser.add_argument("--per-step-timeout-seconds", type=float, default=None)
    parser.add_argument("--total-timeout-seconds", type=float, default=None)
    parser.add_argument("--timeout-policy", choices=["zero_step", "zero_remaining"], default="zero_step")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_strategy(
        data_root=args.data_root,
        strategy_dir=args.strategy_dir,
        output_path=args.output,
        split=args.split,
        model_init_timeout_seconds=args.model_init_timeout_seconds,
        per_step_timeout_seconds=args.per_step_timeout_seconds,
        total_timeout_seconds=args.total_timeout_seconds,
        timeout_policy=args.timeout_policy,
    )
    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
