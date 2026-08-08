from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

STRATEGY_DIR = Path(__file__).resolve().parent
if str(STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(STRATEGY_DIR))

from data_utils import (
    feature_columns_from_path,
    load_train_frame,
    manifest_files,
    sample_by_time,
)
from features import prepare_model_frame, select_history_features
from preprocess import apply_preprocess, fit_feature_schema
from validation import (
    evaluate_gates,
    fit_prediction_scale,
    make_validation_plan,
    weighted_zero_mean_r2,
)

# Shared non-searched LightGBM settings.
BASE_PARAMS = {
    "objective": "regression",
    "metric": "None",
    "learning_rate": 0.03,
    "bagging_freq": 1,
    "verbosity": -1,
}

# Pre-registered capacity × regularization candidates (frozen before training).
PARAM_CANDIDATES: tuple[dict, ...] = (
    {
        "name": "leaves31_regular",
        "num_leaves": 31,
        "min_data_in_leaf": 2000,
        "feature_fraction": 0.80,
        "bagging_fraction": 0.80,
        "lambda_l2": 10.0,
        "regularization_rank": 0,
        "logic": "较浅树 + 中等正则。",
    },
    {
        "name": "leaves63_regular",
        "num_leaves": 63,
        "min_data_in_leaf": 2000,
        "feature_fraction": 0.80,
        "bagging_fraction": 0.80,
        "lambda_l2": 10.0,
        "regularization_rank": 0,
        "logic": "较深树 + 中等正则。",
    },
    {
        "name": "leaves31_strong",
        "num_leaves": 31,
        "min_data_in_leaf": 5000,
        "feature_fraction": 1.00,
        "bagging_fraction": 0.80,
        "lambda_l2": 20.0,
        "regularization_rank": 1,
        "logic": "较浅树 + 更强叶样本量与 L2 正则。",
    },
    {
        "name": "leaves63_strong",
        "num_leaves": 63,
        "min_data_in_leaf": 5000,
        "feature_fraction": 1.00,
        "bagging_fraction": 0.80,
        "lambda_l2": 20.0,
        "regularization_rank": 1,
        "logic": "较深树 + 强正则约束。",
    },
)


def lgb_zero_mean_r2(preds: np.ndarray, dataset: lgb.Dataset) -> tuple[str, float, bool]:
    labels = dataset.get_label()
    weight = dataset.get_weight()
    if weight is None:
        weight = np.ones_like(labels)
    denominator = np.sum(weight * labels * labels)
    score = 0.0 if denominator <= 0 else 1.0 - np.sum(weight * (labels - preds) ** 2) / denominator
    return "weighted_zero_mean_r2", float(score), True


def _candidate_params(
    seed: int,
    candidate: dict,
    *,
    num_threads: int = -1,
    extra_overrides: dict | None = None,
) -> dict:
    params = {
        **BASE_PARAMS,
        "num_leaves": int(candidate["num_leaves"]),
        "min_data_in_leaf": int(candidate["min_data_in_leaf"]),
        "feature_fraction": float(candidate["feature_fraction"]),
        "bagging_fraction": float(candidate["bagging_fraction"]),
        "lambda_l2": float(candidate["lambda_l2"]),
        "num_threads": int(num_threads),
        "seed": int(seed),
        "bagging_seed": int(seed),
        "feature_fraction_seed": int(seed),
        "data_random_seed": int(seed),
    }
    if extra_overrides:
        params.update(extra_overrides)
    return params


def _xy(frame: pd.DataFrame, model_cols: list[str]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    x = frame.loc[:, model_cols]
    y = pd.to_numeric(frame["target"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    w = pd.to_numeric(frame["weight"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=np.float32)
    return x, y, w


def _train_es(
    x_train,
    y_train,
    w_train,
    x_valid,
    y_valid,
    w_valid,
    *,
    seed: int,
    candidate: dict,
    num_boost_round: int,
    early_stopping_rounds: int,
    num_threads: int,
    extra_overrides: dict | None,
) -> lgb.Booster:
    train_set = lgb.Dataset(x_train, label=y_train, weight=w_train, categorical_feature=["asset_id"], free_raw_data=False)
    valid_set = lgb.Dataset(
        x_valid,
        label=y_valid,
        weight=w_valid,
        categorical_feature=["asset_id"],
        reference=train_set,
        free_raw_data=False,
    )
    return lgb.train(
        _candidate_params(seed, candidate, num_threads=num_threads, extra_overrides=extra_overrides),
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[train_set, valid_set],
        valid_names=["train", "valid"],
        feval=lgb_zero_mean_r2,
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )


def _train_fixed(
    x_train,
    y_train,
    w_train,
    *,
    seed: int,
    candidate: dict,
    num_boost_round: int,
    num_threads: int,
    extra_overrides: dict | None,
) -> lgb.Booster:
    train_set = lgb.Dataset(x_train, label=y_train, weight=w_train, categorical_feature=["asset_id"], free_raw_data=False)
    return lgb.train(
        _candidate_params(seed, candidate, num_threads=num_threads, extra_overrides=extra_overrides),
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[train_set],
        valid_names=["train"],
        feval=lgb_zero_mean_r2,
        callbacks=[lgb.log_evaluation(period=0)],
    )


def _mask_times(frame: pd.DataFrame, time_ids: np.ndarray) -> pd.DataFrame:
    return frame.loc[frame["time_id"].isin(set(map(int, time_ids)))].copy()


def _select_winning_candidate(candidate_results: list[dict]) -> dict:
    """Higher mean fold score wins; ties -> stronger regularization, then fewer rounds."""
    ordered = sorted(
        candidate_results,
        key=lambda item: (
            -float(item["mean_fold_score"]),
            -int(item["regularization_rank"]),
            int(item["mean_iterations"]),
        ),
    )
    return ordered[0]


def _evaluate_candidate_cv(
    *,
    prepared: pd.DataFrame,
    model_cols: list[str],
    plan,
    candidate: dict,
    cv_seed: int,
    num_boost_round: int,
    early_stopping_rounds: int,
    max_train_rows: int,
    max_valid_rows: int,
    num_threads: int,
    extra_overrides: dict | None,
) -> dict:
    oof_pred = np.zeros(len(prepared), dtype=np.float64)
    oof_mask = np.zeros(len(prepared), dtype=bool)
    fold_best_iterations: list[int] = []
    fold_scores: list[dict] = []

    for fold in plan.folds:
        train_part = sample_by_time(_mask_times(prepared, fold.train_time_ids), max_train_rows, seed=cv_seed)
        valid_part = sample_by_time(_mask_times(prepared, fold.valid_time_ids), max_valid_rows, seed=cv_seed + 1)
        x_tr, y_tr, w_tr = _xy(train_part, model_cols)
        x_va, y_va, w_va = _xy(valid_part, model_cols)
        model = _train_es(
            x_tr,
            y_tr,
            w_tr,
            x_va,
            y_va,
            w_va,
            seed=cv_seed,
            candidate=candidate,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            num_threads=num_threads,
            extra_overrides=extra_overrides,
        )
        best_iteration = int(model.best_iteration or num_boost_round)
        fold_best_iterations.append(best_iteration)
        valid_mask = prepared["time_id"].isin(set(map(int, fold.valid_time_ids))).to_numpy()
        preds = model.predict(prepared.loc[valid_mask, model_cols], num_iteration=best_iteration)
        oof_pred[valid_mask] = preds
        oof_mask[valid_mask] = True
        fold_score = weighted_zero_mean_r2(
            prepared.loc[valid_mask, "target"].to_numpy(dtype=np.float64),
            preds,
            prepared.loc[valid_mask, "weight"].to_numpy(dtype=np.float64),
        )
        fold_scores.append(
            {
                "fold_id": fold.fold_id,
                "best_iteration": best_iteration,
                "valid_raw": fold_score,
                "train_rows": int(len(train_part)),
                "valid_rows": int(len(valid_part)),
            }
        )
        print(
            f"[cv] candidate={candidate['name']} fold={fold.fold_id} "
            f"best_iteration={best_iteration} valid_raw={fold_score:.6g}",
            flush=True,
        )

    mean_fold_score = float(np.mean([item["valid_raw"] for item in fold_scores]))
    mean_iterations = max(1, int(round(float(np.mean(fold_best_iterations)))))
    oof_frame = prepared.loc[oof_mask]
    oof_raw = weighted_zero_mean_r2(
        oof_frame["target"].to_numpy(dtype=np.float64),
        oof_pred[oof_mask],
        oof_frame["weight"].to_numpy(dtype=np.float64),
    )
    return {
        "name": candidate["name"],
        "logic": candidate.get("logic", ""),
        "regularization_rank": int(candidate.get("regularization_rank", 0)),
        "params": {
            "num_leaves": int(candidate["num_leaves"]),
            "min_data_in_leaf": int(candidate["min_data_in_leaf"]),
            "feature_fraction": float(candidate["feature_fraction"]),
            "bagging_fraction": float(candidate["bagging_fraction"]),
            "lambda_l2": float(candidate["lambda_l2"]),
        },
        "fold_scores": fold_scores,
        "fold_best_iterations": fold_best_iterations,
        "mean_fold_score": mean_fold_score,
        "mean_iterations": mean_iterations,
        "oof_raw": oof_raw,
        "oof_pred": oof_pred,
        "oof_mask": oof_mask,
    }


def run_baseline_training(
    train_frame: pd.DataFrame,
    *,
    output_dir: str | Path,
    feature_cols: list[str] | None = None,
    top_k_history: int = 48,
    rolling_windows: tuple[int, ...] = (5,),
    seeds: tuple[int, ...] = (2026, 2027, 2028),
    n_splits: int = 5,
    holdout_fraction: float = 0.15,
    purge_steps: int = 30,
    num_boost_round: int = 700,
    early_stopping_rounds: int = 80,
    num_threads: int = -1,
    max_train_rows: int = 0,
    max_valid_rows: int = 0,
    corr_sample_rows: int = 200_000,
    param_candidates: tuple[dict, ...] | list[dict] | None = None,
    param_overrides: dict | None = None,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = tuple(param_candidates) if param_candidates is not None else PARAM_CANDIDATES
    if not candidates:
        raise ValueError("param_candidates must be non-empty")

    if feature_cols is None:
        feature_cols = [col for col in train_frame.columns if str(col).startswith("feature_")]
    if not feature_cols:
        raise ValueError("no feature_* columns found")

    plan = make_validation_plan(
        train_frame["time_id"],
        n_splits=n_splits,
        holdout_fraction=holdout_fraction,
        purge_steps=purge_steps,
    )

    # Freeze schema on early development block to avoid using holdout / later folds.
    schema_source = _mask_times(train_frame, plan.folds[0].train_time_ids)
    if schema_source.empty:
        raise ValueError("first fold train is empty; cannot fit preprocess schema")
    schema = fit_feature_schema(schema_source, feature_cols)
    raw_features = list(schema.raw_features)
    cleaned = apply_preprocess(train_frame, schema)

    history_source = sample_by_time(schema_source, corr_sample_rows, seed=seeds[0])
    history_source = apply_preprocess(history_source, schema)
    history_features = select_history_features(
        history_source,
        raw_features,
        top_k=top_k_history,
        sample_rows=corr_sample_rows,
        seed=seeds[0],
    )

    prepared, model_cols = prepare_model_frame(
        cleaned,
        raw_features=raw_features,
        history_features=history_features,
        rolling_windows=rolling_windows,
    )
    prepared = prepared.reset_index(drop=True)

    cv_seed = int(seeds[0])
    candidate_results: list[dict] = []
    for candidate in candidates:
        print(f"[cv] start candidate={candidate['name']}", flush=True)
        result = _evaluate_candidate_cv(
            prepared=prepared,
            model_cols=model_cols,
            plan=plan,
            candidate=candidate,
            cv_seed=cv_seed,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            max_train_rows=max_train_rows,
            max_valid_rows=max_valid_rows,
            num_threads=num_threads,
            extra_overrides=param_overrides,
        )
        candidate_results.append(result)
        print(
            f"[cv] done candidate={candidate['name']} "
            f"mean_fold_score={result['mean_fold_score']:.6g} mean_iterations={result['mean_iterations']}",
            flush=True,
        )

    winner = _select_winning_candidate(candidate_results)
    winning_candidate = next(item for item in candidates if item["name"] == winner["name"])
    mean_iterations = int(winner["mean_iterations"])
    oof_mask = winner["oof_mask"]
    oof_pred = winner["oof_pred"]
    oof_frame = prepared.loc[oof_mask]
    fitted_oof_scale = fit_prediction_scale(
        oof_frame["target"].to_numpy(dtype=np.float64),
        oof_pred[oof_mask],
        oof_frame["weight"].to_numpy(dtype=np.float64),
    )
    oof_raw = float(winner["oof_raw"])

    holdout_part = _mask_times(prepared, plan.holdout_time_ids)
    development_part = sample_by_time(
        prepared.loc[prepared["time_id"].isin(set(map(int, plan.development_time_ids)))],
        max_train_rows,
        seed=cv_seed,
    )
    x_dev, y_dev, w_dev = _xy(development_part, model_cols)
    print(f"[holdout] train on development with mean_iterations={mean_iterations}", flush=True)
    holdout_model = _train_fixed(
        x_dev,
        y_dev,
        w_dev,
        seed=cv_seed,
        candidate=winning_candidate,
        num_boost_round=mean_iterations,
        num_threads=num_threads,
        extra_overrides=param_overrides,
    )
    holdout_pred = holdout_model.predict(holdout_part.loc[:, model_cols], num_iteration=mean_iterations)
    holdout_raw = weighted_zero_mean_r2(
        holdout_part["target"].to_numpy(dtype=np.float64),
        holdout_pred,
        holdout_part["weight"].to_numpy(dtype=np.float64),
    )

    gates = evaluate_gates(
        oof_raw_score=oof_raw,
        holdout_raw_score=holdout_raw,
        fitted_oof_scale=fitted_oof_scale,
    )

    # Final fit uses all labeled train rows, including holdout.
    final_train = sample_by_time(prepared, max_train_rows, seed=seeds[0])
    x_all, y_all, w_all = _xy(final_train, model_cols)
    model_files: list[str] = []
    best_iterations: list[int] = []
    for seed in seeds:
        print(f"[final] seed={seed} rounds={mean_iterations} candidate={winner['name']}", flush=True)
        booster = _train_fixed(
            x_all,
            y_all,
            w_all,
            seed=int(seed),
            candidate=winning_candidate,
            num_boost_round=mean_iterations,
            num_threads=num_threads,
            extra_overrides=param_overrides,
        )
        name = "model.txt" if len(seeds) == 1 else f"model_seed{seed}.txt"
        booster.save_model(str(output_dir / name))
        model_files.append(name)
        best_iterations.append(mean_iterations)

    report = {
        "strategy": "lightgbm_baseline",
        "schema_version": 1,
        "tuning_policy": "purged_kfold_pre_registered_candidates_no_test_tuning",
        "scale_policy": "diagnostic_only_never_apply",
        "rows": {
            "train_all": int(len(prepared)),
            "oof": int(oof_mask.sum()),
            "holdout": int(len(holdout_part)),
            "development": int(len(development_part)),
            "final_train_sample": int(len(final_train)),
            "final_train_includes_holdout": True,
        },
        "validation": {
            "cv_scheme": plan.cv_scheme,
            "n_splits": n_splits,
            "holdout_fraction": holdout_fraction,
            "purge_steps": purge_steps,
            "rounds_aggregation": "mean",
            "selection_metric": "mean_fold_score",
            "tie_break": ["stronger_regularization", "fewer_mean_iterations"],
            "candidates": [
                {
                    "name": item["name"],
                    "logic": item["logic"],
                    "params": item["params"],
                    "regularization_rank": item["regularization_rank"],
                    "mean_fold_score": item["mean_fold_score"],
                    "mean_iterations": item["mean_iterations"],
                    "oof_raw": item["oof_raw"],
                    "fold_best_iterations": item["fold_best_iterations"],
                    "fold_scores": item["fold_scores"],
                }
                for item in candidate_results
            ],
            "selected_candidate": winner["name"],
            "fold_scores": winner["fold_scores"],
            "fold_best_iterations": winner["fold_best_iterations"],
            "mean_iterations": mean_iterations,
            "mean_fold_score": winner["mean_fold_score"],
            "oof_raw": oof_raw,
            "holdout_raw": holdout_raw,
            "fitted_oof_scale": fitted_oof_scale,
            "gates": gates,
        },
        "features": {
            "selected_raw_features": raw_features,
            "history_features": history_features,
            "rolling_windows": list(rolling_windows),
            "model_feature_count": len(model_cols),
        },
        "seeds": list(map(int, seeds)),
        "model_files": model_files,
        "best_iteration": mean_iterations,
        "best_iterations": best_iterations,
        "prediction_scale": 1.0,
        "fitted_oof_scale": fitted_oof_scale,
        "gates_passed": gates["gates_passed"],
        "selected_candidate": winner["name"],
        "num_threads": int(num_threads),
        "lgbm_params": _candidate_params(
            seeds[0],
            winning_candidate,
            num_threads=num_threads,
            extra_overrides=param_overrides,
        ),
    }
    (output_dir / "lightgbm_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the LightGBM baseline (purged K-fold gated).")
    parser.add_argument("--release-root", required=True, help="Release root containing manifest.json and train/.")
    parser.add_argument(
        "--model-dir",
        "--output-dir",
        dest="output_dir",
        required=True,
        help="Directory to write model files and reports.",
    )
    parser.add_argument("--top-k-history", type=int, default=48)
    parser.add_argument("--max-train-rows", type=int, default=0, help="0 means use all rows.")
    parser.add_argument("--max-valid-rows", type=int, default=0, help="0 means use all rows.")
    parser.add_argument("--num-boost-round", type=int, default=700)
    parser.add_argument("--early-stopping-rounds", type=int, default=80)
    parser.add_argument(
        "--num-threads",
        type=int,
        default=-1,
        help="LightGBM num_threads; -1 uses all available cores.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    release_root = Path(args.release_root)
    train_paths = manifest_files(release_root, "train")
    feature_cols = feature_columns_from_path(train_paths[0])
    load_cols = ["row_id", "time_id", "asset_id", "weight", "target", *feature_cols]
    train_frame = load_train_frame(release_root, columns=load_cols)
    report = run_baseline_training(
        train_frame,
        output_dir=args.output_dir,
        feature_cols=feature_cols,
        top_k_history=args.top_k_history,
        max_train_rows=args.max_train_rows,
        max_valid_rows=args.max_valid_rows,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
        num_threads=args.num_threads,
    )
    print(
        json.dumps(
            {
                "gates_passed": report["gates_passed"],
                "selected_candidate": report["selected_candidate"],
                "mean_iterations": report["best_iteration"],
                "output_dir": args.output_dir,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
