#!/usr/bin/env python3
"""Recompute the paper's fixed XGBoost-versus-random-forest evaluation.

The required input is an existing OOF prediction table.  This program does not
fit a model, create predictions, construct folds, or generate resampling draws.
An optional path can replay the already-sealed target multiplicity matrix.
Only aggregate tables are written.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CANDIDATE_MODEL = "xgboost"
REFERENCE_MODEL = "random_forest"
MODEL_SEEDS = (20260624, 20260724, 20260801)
SPLIT_SEEDS = (20260624, 20260625, 20260626, 20260724, 20260801)
FOLDS = (0, 1, 2, 3, 4)

EXPECTED_STREAM_ROWS = 13_110
EXPECTED_PAIR_SUPPORT_PER_SEED = 4_370
EXPECTED_EQUAL_KEY_SUPPORT_PER_SEED = 4_190

TARGET_COUNT = 17
TARGET_REPLICATES = 50_000
TARGET_MULTIPLICITIES_SHA256 = (
    "2F6A8C820FBD67D2D5B1DDAD309B8C1648A104209BA59706A8EE1B26D5CED858"
)

REQUIRED_PREDICTION_COLUMNS = (
    "pair_id",
    "comparison_pair_key",
    "component_id",
    "true_delta",
    "prediction",
    "model",
    "model_seed",
    "split_seed",
    "fold",
)
JOIN_KEYS = ("split_seed", "fold", "model_seed", "pair_id")

FAMILIES = (
    {
        "comparison_id": "xgboost_vs_random_forest__pair_id__direction_accuracy",
        "estimand": "pair_id",
        "metric": "direction_accuracy",
        "contrast_direction": "candidate_minus_reference",
        "formal_point_estimate": 0.032265446224256311,
    },
    {
        "comparison_id": "xgboost_vs_random_forest__pair_id__delta_mae",
        "estimand": "pair_id",
        "metric": "delta_mae",
        "contrast_direction": "reference_minus_candidate",
        "formal_point_estimate": 0.13988297549150955,
    },
    {
        "comparison_id": "xgboost_vs_random_forest__pair_id__delta_rmse",
        "estimand": "pair_id",
        "metric": "delta_rmse",
        "contrast_direction": "reference_minus_candidate",
        "formal_point_estimate": 0.040420295983005193,
    },
    {
        "comparison_id": "xgboost_vs_random_forest__equal_comparison_key__direction_accuracy",
        "estimand": "equal_comparison_key",
        "metric": "direction_accuracy",
        "contrast_direction": "candidate_minus_reference",
        "formal_point_estimate": 0.025218774860779652,
    },
    {
        "comparison_id": "xgboost_vs_random_forest__equal_comparison_key__delta_mae",
        "estimand": "equal_comparison_key",
        "metric": "delta_mae",
        "contrast_direction": "reference_minus_candidate",
        "formal_point_estimate": 0.1370446977386639,
    },
    {
        "comparison_id": "xgboost_vs_random_forest__equal_comparison_key__delta_rmse",
        "estimand": "equal_comparison_key",
        "metric": "delta_rmse",
        "contrast_direction": "reference_minus_candidate",
        "formal_point_estimate": 0.038658720468472373,
    },
)

PAIR_VALUE_COLUMNS = {
    CANDIDATE_MODEL: {
        "direction_accuracy": "candidate_direction_value",
        "delta_mae": "candidate_absolute_loss",
        "delta_rmse": "candidate_squared_loss",
    },
    REFERENCE_MODEL: {
        "direction_accuracy": "reference_direction_value",
        "delta_mae": "reference_absolute_loss",
        "delta_rmse": "reference_squared_loss",
    },
}

EQUAL_KEY_VALUE_COLUMNS = {
    CANDIDATE_MODEL: {
        "direction_accuracy": "candidate_direction_mean",
        "delta_mae": "candidate_absolute_loss_mean",
        "delta_rmse": "candidate_squared_loss_mean",
    },
    REFERENCE_MODEL: {
        "direction_accuracy": "reference_direction_mean",
        "delta_mae": "reference_absolute_loss_mean",
        "delta_rmse": "reference_squared_loss_mean",
    },
}


class EvaluationError(RuntimeError):
    """Raised when an input cannot support the frozen evaluation."""


def f64(value: float | np.floating[Any]) -> str:
    return format(float(value), ".17g")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def require_columns(frame: pd.DataFrame, required: tuple[str, ...], label: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise EvaluationError(f"{label} is missing columns: {', '.join(missing)}")


def coerce_integer(frame: pd.DataFrame, column: str, label: str) -> None:
    try:
        numeric = pd.to_numeric(frame[column], errors="raise")
    except (TypeError, ValueError) as exc:
        raise EvaluationError(f"{label}.{column} must contain integers") from exc
    values = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise EvaluationError(f"{label}.{column} must contain finite integers")
    frame[column] = numeric.astype(np.int64)


def coerce_float(frame: pd.DataFrame, column: str, label: str) -> None:
    try:
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise EvaluationError(f"{label}.{column} must contain numeric values") from exc
    if not np.isfinite(values).all():
        raise EvaluationError(f"{label}.{column} contains a non-finite value")
    frame[column] = values


def load_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)
    require_columns(frame, REQUIRED_PREDICTION_COLUMNS, "predictions")

    selected = frame[frame["model"].isin((CANDIDATE_MODEL, REFERENCE_MODEL))].copy()
    for column in ("pair_id", "comparison_pair_key", "component_id", "model"):
        if selected[column].str.strip().eq("").any():
            raise EvaluationError(f"predictions.{column} contains a blank value")
    for column in ("model_seed", "split_seed", "fold"):
        coerce_integer(selected, column, "predictions")
    for column in ("true_delta", "prediction"):
        coerce_float(selected, column, "predictions")

    if (selected["true_delta"] == 0).any():
        raise EvaluationError("true_delta must be nonzero for every evaluated cliff row")

    for model in (CANDIDATE_MODEL, REFERENCE_MODEL):
        model_frame = selected[selected["model"] == model]
        if len(model_frame) != EXPECTED_STREAM_ROWS:
            raise EvaluationError(
                f"{model} has {len(model_frame)} rows; expected {EXPECTED_STREAM_ROWS}"
            )
        observed_seeds = tuple(sorted(model_frame["model_seed"].unique().tolist()))
        if observed_seeds != MODEL_SEEDS:
            raise EvaluationError(f"{model} model seeds differ from the frozen three-seed set")
        observed_splits = tuple(sorted(model_frame["split_seed"].unique().tolist()))
        if observed_splits != SPLIT_SEEDS:
            raise EvaluationError(f"{model} split seeds differ from the frozen five-seed set")
        observed_folds = tuple(sorted(model_frame["fold"].unique().tolist()))
        if observed_folds != FOLDS:
            raise EvaluationError(f"{model} folds differ from the frozen five-fold set")
        if model_frame.duplicated(list(JOIN_KEYS)).any():
            raise EvaluationError(f"{model} contains duplicate OOF join keys")

    expected_cells = set(
        itertools.product(
            (CANDIDATE_MODEL, REFERENCE_MODEL), SPLIT_SEEDS, FOLDS, MODEL_SEEDS
        )
    )
    observed_cells = set(
        selected[["model", "split_seed", "fold", "model_seed"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    if observed_cells != expected_cells:
        raise EvaluationError("the frozen 150 model/split/fold/seed cells are incomplete")
    return selected


def join_model_streams(predictions: pd.DataFrame) -> pd.DataFrame:
    projected = [*JOIN_KEYS, "comparison_pair_key", "component_id", "true_delta", "prediction"]
    candidate = predictions[predictions["model"] == CANDIDATE_MODEL][projected].copy()
    reference = predictions[predictions["model"] == REFERENCE_MODEL][projected].copy()
    joined = candidate.merge(
        reference,
        on=list(JOIN_KEYS),
        how="outer",
        suffixes=("_candidate", "_reference"),
        indicator=True,
        validate="one_to_one",
        sort=True,
    )
    if not joined["_merge"].eq("both").all() or len(joined) != EXPECTED_STREAM_ROWS:
        raise EvaluationError("XGBoost and random-forest OOF keys are not identical")

    for column in ("comparison_pair_key", "component_id"):
        if not joined[f"{column}_candidate"].equals(joined[f"{column}_reference"]):
            raise EvaluationError(f"model streams disagree on {column}")
    if not np.array_equal(
        joined["true_delta_candidate"].to_numpy(dtype=np.float64),
        joined["true_delta_reference"].to_numpy(dtype=np.float64),
    ):
        raise EvaluationError("model streams disagree on true_delta")

    result = pd.DataFrame(
        {
            "split_seed": joined["split_seed"].astype(np.int64),
            "fold": joined["fold"].astype(np.int64),
            "model_seed": joined["model_seed"].astype(np.int64),
            "pair_id": joined["pair_id"].astype(str),
            "comparison_pair_key": joined["comparison_pair_key_candidate"].astype(str),
            "component_id": joined["component_id_candidate"].astype(str),
            "true_delta": joined["true_delta_candidate"].astype(np.float64),
            "candidate_prediction": joined["prediction_candidate"].astype(np.float64),
            "reference_prediction": joined["prediction_reference"].astype(np.float64),
        }
    ).sort_values(list(JOIN_KEYS), kind="stable", ignore_index=True)

    if result.duplicated(["split_seed", "model_seed", "pair_id"]).any():
        raise EvaluationError("a pair occurs in more than one fold within a split/model seed")
    grouped = result.groupby(["split_seed", "fold", "pair_id"], sort=False, dropna=False)
    if not grouped.size().eq(len(MODEL_SEEDS)).all():
        raise EvaluationError("OOF membership differs across model seeds")
    for column in ("comparison_pair_key", "component_id", "true_delta"):
        if not grouped[column].nunique(dropna=False).eq(1).all():
            raise EvaluationError(f"{column} changes across model seeds")
    return result


def pair_losses(joined: pd.DataFrame) -> pd.DataFrame:
    result = joined.copy()
    truth = result["true_delta"].to_numpy(dtype=np.float64)
    for prefix in ("candidate", "reference"):
        prediction = result[f"{prefix}_prediction"].to_numpy(dtype=np.float64)
        result[f"{prefix}_direction_value"] = (
            (prediction != 0) & (np.sign(prediction) == np.sign(truth))
        ).astype(np.float64)
        result[f"{prefix}_absolute_loss"] = np.abs(prediction - truth)
        result[f"{prefix}_squared_loss"] = (prediction - truth) ** 2
    return result


def equal_key_losses(joined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups = joined.groupby(
        ["split_seed", "model_seed", "comparison_pair_key"], sort=True, dropna=False
    )
    for (split_seed, model_seed, comparison_key), group in groups:
        if group["component_id"].nunique(dropna=False) != 1:
            raise EvaluationError("an equal-comparison key spans more than one component")
        ordered = group.sort_values("pair_id", kind="stable")
        truth = ordered["true_delta"].to_numpy(dtype=np.float64)
        candidate = ordered["candidate_prediction"].to_numpy(dtype=np.float64)
        reference = ordered["reference_prediction"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "split_seed": int(split_seed),
                "model_seed": int(model_seed),
                "comparison_pair_key": str(comparison_key),
                "component_id": str(ordered["component_id"].iloc[0]),
                "candidate_direction_mean": float(
                    np.mean(
                        (candidate != 0) & (np.sign(candidate) == np.sign(truth)),
                        dtype=np.float64,
                    )
                ),
                "reference_direction_mean": float(
                    np.mean(
                        (reference != 0) & (np.sign(reference) == np.sign(truth)),
                        dtype=np.float64,
                    )
                ),
                "candidate_absolute_loss_mean": float(
                    np.mean(np.abs(candidate - truth), dtype=np.float64)
                ),
                "reference_absolute_loss_mean": float(
                    np.mean(np.abs(reference - truth), dtype=np.float64)
                ),
                "candidate_squared_loss_mean": float(
                    np.mean((candidate - truth) ** 2, dtype=np.float64)
                ),
                "reference_squared_loss_mean": float(
                    np.mean((reference - truth) ** 2, dtype=np.float64)
                ),
            }
        )
    result = pd.DataFrame(rows).sort_values(
        ["split_seed", "model_seed", "comparison_pair_key"],
        kind="stable",
        ignore_index=True,
    )
    expected_rows = EXPECTED_EQUAL_KEY_SUPPORT_PER_SEED * len(MODEL_SEEDS)
    if len(result) != expected_rows:
        raise EvaluationError(
            f"equal-key aggregation produced {len(result)} rows; expected {expected_rows}"
        )
    return result


def metric_value(
    frame: pd.DataFrame,
    value_column: str,
    model_seed: int,
    metric: str,
) -> tuple[float, int]:
    seed_frame = frame[frame["model_seed"] == model_seed]
    numerator = 0.0
    denominator = 0
    for _component, group in seed_frame.groupby("component_id", sort=True, dropna=False):
        values = group[value_column].to_numpy(dtype=np.float64)
        numerator += float(np.sum(values, dtype=np.float64))
        denominator += len(values)
    if denominator <= 0:
        raise EvaluationError("a model seed has zero evaluation support")
    mean_value = numerator / denominator
    if metric == "delta_rmse":
        if mean_value < -1e-15:
            raise EvaluationError("weighted mean squared error is negative")
        mean_value = float(np.sqrt(max(mean_value, 0.0)))
    return float(mean_value), int(denominator)


def point_tables(
    pair_frame: pd.DataFrame,
    equal_frame: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    mean_rows: list[dict[str, Any]] = []
    estimands = {
        "pair_id": (pair_frame, PAIR_VALUE_COLUMNS, EXPECTED_PAIR_SUPPORT_PER_SEED),
        "equal_comparison_key": (
            equal_frame,
            EQUAL_KEY_VALUE_COLUMNS,
            EXPECTED_EQUAL_KEY_SUPPORT_PER_SEED,
        ),
    }

    for family in FAMILIES:
        frame, value_columns, expected_support = estimands[family["estimand"]]
        seed_contrasts: list[float] = []
        for seed in MODEL_SEEDS:
            values: dict[str, float] = {}
            supports: dict[str, int] = {}
            for model in (CANDIDATE_MODEL, REFERENCE_MODEL):
                value, support = metric_value(
                    frame, value_columns[model][family["metric"]], seed, family["metric"]
                )
                if support != expected_support:
                    raise EvaluationError(
                        f"{family['estimand']} support for seed {seed} is {support}; "
                        f"expected {expected_support}"
                    )
                values[model] = value
                supports[model] = support
                metric_rows.append(
                    {
                        "model_seed": seed,
                        "model_id": model,
                        "estimand": family["estimand"],
                        "metric": family["metric"],
                        "metric_value": f64(value),
                        "support_count": support,
                    }
                )
            if supports[CANDIDATE_MODEL] != supports[REFERENCE_MODEL]:
                raise EvaluationError("candidate and reference supports are asymmetric")
            if family["contrast_direction"] == "candidate_minus_reference":
                contrast = values[CANDIDATE_MODEL] - values[REFERENCE_MODEL]
            else:
                contrast = values[REFERENCE_MODEL] - values[CANDIDATE_MODEL]
            seed_contrasts.append(float(contrast))
            contrast_rows.append(
                {
                    "comparison_id": family["comparison_id"],
                    "estimand": family["estimand"],
                    "metric": family["metric"],
                    "model_seed": seed,
                    "candidate_value": f64(values[CANDIDATE_MODEL]),
                    "reference_value": f64(values[REFERENCE_MODEL]),
                    "contrast_direction": family["contrast_direction"],
                    "seed_contrast": f64(contrast),
                    "candidate_support_count": supports[CANDIDATE_MODEL],
                    "reference_support_count": supports[REFERENCE_MODEL],
                    "status": "ESTIMABLE",
                }
            )
        mean_contrast = float(
            np.mean(np.asarray(seed_contrasts, dtype=np.float64), dtype=np.float64)
        )
        mean_rows.append(
            {
                "comparison_id": family["comparison_id"],
                "estimand": family["estimand"],
                "metric": family["metric"],
                "contrast_direction": family["contrast_direction"],
                "model_seed_count": len(MODEL_SEEDS),
                "mean_seed_contrast": f64(mean_contrast),
                "status": "ESTIMABLE",
            }
        )
    return metric_rows, contrast_rows, mean_rows


def load_target_map(path: Path, key_column: str, label: str) -> pd.DataFrame:
    required = (key_column, "comparison_pair_key", "component_id", "normalized_poi_id", "target_index")
    if key_column == "comparison_pair_key":
        required = ("comparison_pair_key", "component_id", "normalized_poi_id", "target_index")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)
    require_columns(frame, required, label)
    for column in required:
        if frame[column].str.strip().eq("").any():
            raise EvaluationError(f"{label}.{column} contains a blank value")
    coerce_integer(frame, "target_index", label)
    if frame.duplicated(key_column).any():
        raise EvaluationError(f"{label}.{key_column} is not unique")
    if not frame["target_index"].between(0, TARGET_COUNT - 1).all():
        raise EvaluationError(f"{label}.target_index falls outside 0..{TARGET_COUNT - 1}")
    return frame[list(required)].copy()


def require_exact_key_universe(
    mapping: pd.DataFrame,
    evaluated: pd.Series,
    key_column: str,
    label: str,
) -> None:
    mapped_keys = set(mapping[key_column].astype(str))
    evaluated_keys = set(evaluated.astype(str))
    if mapped_keys != evaluated_keys:
        raise EvaluationError(
            f"{label} key universe differs from the evaluated universe "
            f"(missing={len(evaluated_keys - mapped_keys)}, "
            f"extra={len(mapped_keys - evaluated_keys)})"
        )


def validated_target_universe(
    bound: pd.DataFrame,
    label: str,
) -> tuple[tuple[str, int], ...]:
    if not (
        bound.groupby("normalized_poi_id")["target_index"]
        .nunique()
        .eq(1)
        .all()
    ):
        raise EvaluationError(f"{label} maps one POI to more than one target index")
    if not (
        bound.groupby("target_index")["normalized_poi_id"]
        .nunique()
        .eq(1)
        .all()
    ):
        raise EvaluationError(f"{label} maps one target index to more than one POI")

    universe = bound[["normalized_poi_id", "target_index"]].drop_duplicates()
    if (
        len(universe) != TARGET_COUNT
        or universe["normalized_poi_id"].nunique() != TARGET_COUNT
        or universe["target_index"].nunique() != TARGET_COUNT
    ):
        raise EvaluationError(f"{label} does not contain exactly {TARGET_COUNT} POI/index pairs")
    ordered = universe.sort_values("normalized_poi_id", kind="stable")
    if ordered["target_index"].tolist() != list(range(TARGET_COUNT)):
        raise EvaluationError(f"{label} indices do not follow lexicographic POI order")
    return tuple(
        (str(row.normalized_poi_id), int(row.target_index))
        for row in ordered.itertuples(index=False)
    )


def bind_targets(
    pair_frame: pd.DataFrame,
    equal_frame: pd.DataFrame,
    pair_map_path: Path,
    key_map_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_map = load_target_map(pair_map_path, "pair_id", "pair target map")
    key_map = load_target_map(key_map_path, "comparison_pair_key", "key target map")
    require_exact_key_universe(
        pair_map,
        pair_frame["pair_id"],
        "pair_id",
        "pair target map",
    )
    require_exact_key_universe(
        key_map,
        equal_frame["comparison_pair_key"],
        "comparison_pair_key",
        "key target map",
    )

    pair_bound = pair_frame.merge(
        pair_map,
        on=["pair_id", "comparison_pair_key", "component_id"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if not pair_bound["_merge"].eq("both").all():
        raise EvaluationError("pair target map does not bind every evaluated pair")
    pair_bound = pair_bound.drop(columns="_merge")

    equal_bound = equal_frame.merge(
        key_map,
        on=["comparison_pair_key", "component_id"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if not equal_bound["_merge"].eq("both").all():
        raise EvaluationError("key target map does not bind every evaluated key")
    equal_bound = equal_bound.drop(columns="_merge")

    pair_consistency = pair_bound[
        ["comparison_pair_key", "normalized_poi_id", "target_index"]
    ].drop_duplicates()
    if pair_consistency.duplicated("comparison_pair_key").any():
        raise EvaluationError("members of an equal-comparison key map to different targets")
    key_consistency = equal_bound[
        ["comparison_pair_key", "normalized_poi_id", "target_index"]
    ].drop_duplicates()
    if key_consistency.duplicated("comparison_pair_key").any():
        raise EvaluationError("an evaluated equal-comparison key maps to different targets")
    check = pair_consistency.merge(
        key_consistency,
        on="comparison_pair_key",
        how="outer",
        suffixes=("_pair", "_key"),
        indicator=True,
        validate="one_to_one",
    )
    if not check["_merge"].eq("both").all():
        raise EvaluationError("pair and equal-key target universes differ")
    if not check["normalized_poi_id_pair"].equals(check["normalized_poi_id_key"]):
        raise EvaluationError("pair and equal-key target identifiers disagree")
    if not check["target_index_pair"].equals(check["target_index_key"]):
        raise EvaluationError("pair and equal-key target indices disagree")

    pair_universe = validated_target_universe(pair_bound, "pair-bound evaluation universe")
    key_universe = validated_target_universe(equal_bound, "equal-key-bound evaluation universe")
    if pair_universe != key_universe:
        raise EvaluationError("pair and equal-key evaluation universes have different POI/index maps")
    return pair_bound, equal_bound


def target_aggregates(
    pair_frame: pd.DataFrame,
    equal_frame: pd.DataFrame,
) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    for estimand, frame, columns in (
        ("pair_id", pair_frame, PAIR_VALUE_COLUMNS),
        ("equal_comparison_key", equal_frame, EQUAL_KEY_VALUE_COLUMNS),
    ):
        counts = np.zeros((len(MODEL_SEEDS), TARGET_COUNT), dtype=np.int64)
        sums = {
            f"{model}:{metric}": np.zeros(
                (len(MODEL_SEEDS), TARGET_COUNT), dtype=np.float64
            )
            for model in (CANDIDATE_MODEL, REFERENCE_MODEL)
            for metric in ("direction_accuracy", "delta_mae", "delta_rmse")
        }
        for seed_index, seed in enumerate(MODEL_SEEDS):
            seed_frame = frame[frame["model_seed"] == seed]
            indices = seed_frame["target_index"].to_numpy(dtype=np.int64)
            counts[seed_index] = np.bincount(indices, minlength=TARGET_COUNT).astype(
                np.int64
            )
            for model in (CANDIDATE_MODEL, REFERENCE_MODEL):
                for metric in ("direction_accuracy", "delta_mae", "delta_rmse"):
                    values = seed_frame[columns[model][metric]].to_numpy(dtype=np.float64)
                    sums[f"{model}:{metric}"][seed_index] = np.bincount(
                        indices,
                        weights=values,
                        minlength=TARGET_COUNT,
                    ).astype(np.float64)
        result[estimand] = {"counts": counts, **sums}
    return result


def values_from_weights(
    sums: np.ndarray,
    counts: np.ndarray,
    weights: np.ndarray,
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    numerator = weights @ sums
    denominator = weights @ counts
    values = np.full(len(weights), np.nan, dtype=np.float64)
    valid = denominator > 0
    values[valid] = numerator[valid] / denominator[valid]
    if metric == "delta_rmse":
        if np.any(values[valid] < -1e-15):
            raise EvaluationError("target-weighted mean squared error is negative")
        values[valid] = np.sqrt(np.maximum(values[valid], 0.0))
    return values, denominator


def replay_target_plan(
    multiplicities_path: Path,
    aggregates: dict[str, dict[str, np.ndarray]],
    mean_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if sha256_file(multiplicities_path) != TARGET_MULTIPLICITIES_SHA256:
        raise EvaluationError("target multiplicity matrix does not match the sealed plan")
    multiplicities = np.load(multiplicities_path, mmap_mode="r", allow_pickle=False)
    if multiplicities.shape != (TARGET_REPLICATES, TARGET_COUNT):
        raise EvaluationError(
            f"target multiplicity shape is {multiplicities.shape}; "
            f"expected {(TARGET_REPLICATES, TARGET_COUNT)}"
        )
    if multiplicities.dtype != np.dtype("uint16"):
        raise EvaluationError("target multiplicity dtype must be uint16")
    if not np.all(multiplicities.sum(axis=1, dtype=np.uint64) == TARGET_COUNT):
        raise EvaluationError("each target-plan row must contain exactly 17 tickets")

    point_lookup = {
        row["comparison_id"]: float(row["mean_seed_contrast"]) for row in mean_rows
    }
    for family in FAMILIES:
        observed = point_lookup[family["comparison_id"]]
        if not np.isclose(
            observed,
            float(family["formal_point_estimate"]),
            atol=1e-12,
            rtol=0.0,
        ):
            raise EvaluationError(
                "target-plan replay requires the frozen primary point estimates"
            )

    weights = np.asarray(multiplicities, dtype=np.float64)
    seed_contrasts = np.full(
        (TARGET_REPLICATES, len(FAMILIES), len(MODEL_SEEDS)),
        np.nan,
        dtype=np.float64,
    )
    candidate_support = np.zeros_like(seed_contrasts)
    reference_support = np.zeros_like(seed_contrasts)
    for family_index, family in enumerate(FAMILIES):
        arrays = aggregates[family["estimand"]]
        for seed_index, _seed in enumerate(MODEL_SEEDS):
            candidate, candidate_denominator = values_from_weights(
                arrays[f"{CANDIDATE_MODEL}:{family['metric']}"][seed_index],
                arrays["counts"][seed_index],
                weights,
                family["metric"],
            )
            reference, reference_denominator = values_from_weights(
                arrays[f"{REFERENCE_MODEL}:{family['metric']}"][seed_index],
                arrays["counts"][seed_index],
                weights,
                family["metric"],
            )
            candidate_support[:, family_index, seed_index] = candidate_denominator
            reference_support[:, family_index, seed_index] = reference_denominator
            valid = (
                (candidate_denominator > 0)
                & (reference_denominator > 0)
                & (candidate_denominator == reference_denominator)
                & np.isfinite(candidate)
                & np.isfinite(reference)
            )
            if family["contrast_direction"] == "candidate_minus_reference":
                seed_contrasts[valid, family_index, seed_index] = (
                    candidate[valid] - reference[valid]
                )
            else:
                seed_contrasts[valid, family_index, seed_index] = (
                    reference[valid] - candidate[valid]
                )

    family_values = np.mean(seed_contrasts, axis=2, dtype=np.float64)
    replicate_complete = (
        np.isfinite(seed_contrasts).all(axis=(1, 2))
        & np.isfinite(family_values).all(axis=1)
        & (candidate_support > 0).all(axis=(1, 2))
        & (reference_support > 0).all(axis=(1, 2))
        & (candidate_support == reference_support).all(axis=(1, 2))
    )
    complete_count = int(replicate_complete.sum())
    family_values[~replicate_complete, :] = np.nan

    alpha = 0.05
    family_count = len(FAMILIES)
    lower_probability = alpha / (2 * family_count)
    upper_probability = 1 - lower_probability
    rows: list[dict[str, Any]] = []
    for family_index, family in enumerate(FAMILIES):
        if complete_count == TARGET_REPLICATES:
            lower, upper = np.quantile(
                family_values[:, family_index],
                [lower_probability, upper_probability],
                method="linear",
            )
            lower_text = f64(lower)
            upper_text = f64(upper)
            better = str(bool(lower > 0)).lower()
            status = "ESTIMABLE"
        else:
            lower_text = ""
            upper_text = ""
            better = "false"
            status = "NONESTIMABLE_INCOMPLETE_REPLICATE_SET"
        rows.append(
            {
                "comparison_id": family["comparison_id"],
                "estimand": family["estimand"],
                "metric": family["metric"],
                "contrast_direction": family["contrast_direction"],
                "target_count": TARGET_COUNT,
                "point_estimate": f64(point_lookup[family["comparison_id"]]),
                "provided_plan_replicates": TARGET_REPLICATES,
                "estimable_replicates": complete_count,
                "nonestimable_replicates": TARGET_REPLICATES - complete_count,
                "bonferroni_family_size": family_count,
                "two_sided_alpha": f64(alpha),
                "lower_probability": f64(lower_probability),
                "upper_probability": f64(upper_probability),
                "quantile_method": "linear",
                "lower_bound": lower_text,
                "upper_bound": upper_text,
                "significant_better_positive": better,
                "status": status,
                "interpretation": "post_hoc_supported_target_unit_sensitivity",
            }
        )
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen XGBoost and random-forest OOF predictions."
    )
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pair-target-map", type=Path)
    parser.add_argument("--key-target-map", type=Path)
    parser.add_argument("--target-multiplicities", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target_arguments = (
        args.pair_target_map,
        args.key_target_map,
        args.target_multiplicities,
    )
    if any(value is not None for value in target_arguments) and not all(
        value is not None for value in target_arguments
    ):
        raise EvaluationError(
            "target-plan replay requires --pair-target-map, --key-target-map, "
            "and --target-multiplicities together"
        )

    predictions = load_predictions(args.predictions)
    joined = join_model_streams(predictions)
    pair_frame = pair_losses(joined)
    equal_frame = equal_key_losses(joined)
    metric_rows, contrast_rows, mean_rows = point_tables(pair_frame, equal_frame)

    target_rows: list[dict[str, Any]] | None = None
    if all(value is not None for value in target_arguments):
        pair_bound, equal_bound = bind_targets(
            pair_frame,
            equal_frame,
            args.pair_target_map,
            args.key_target_map,
        )
        aggregates = target_aggregates(pair_bound, equal_bound)
        target_rows = replay_target_plan(
            args.target_multiplicities,
            aggregates,
            mean_rows,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "seed_model_metrics.csv",
        ["model_seed", "model_id", "estimand", "metric", "metric_value", "support_count"],
        metric_rows,
    )
    write_csv(
        args.output_dir / "seed_contrasts.csv",
        [
            "comparison_id",
            "estimand",
            "metric",
            "model_seed",
            "candidate_value",
            "reference_value",
            "contrast_direction",
            "seed_contrast",
            "candidate_support_count",
            "reference_support_count",
            "status",
        ],
        contrast_rows,
    )
    write_csv(
        args.output_dir / "mean_contrasts.csv",
        [
            "comparison_id",
            "estimand",
            "metric",
            "contrast_direction",
            "model_seed_count",
            "mean_seed_contrast",
            "status",
        ],
        mean_rows,
    )
    if target_rows is not None:
        write_csv(
            args.output_dir / "target_plan_intervals.csv",
            [
                "comparison_id",
                "estimand",
                "metric",
                "contrast_direction",
                "target_count",
                "point_estimate",
                "provided_plan_replicates",
                "estimable_replicates",
                "nonestimable_replicates",
                "bonferroni_family_size",
                "two_sided_alpha",
                "lower_probability",
                "upper_probability",
                "quantile_method",
                "lower_bound",
                "upper_bound",
                "significant_better_positive",
                "status",
                "interpretation",
            ],
            target_rows,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EvaluationError as exc:
        print(f"evaluation error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
