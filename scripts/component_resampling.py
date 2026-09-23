#!/usr/bin/env python3
"""Shared, prediction-only component resampling utilities.

The public command-line programs import this module to recompute metrics from
saved out-of-fold predictions and a saved component multiplicity plan.  The
module never fits a model, creates a bootstrap draw, downloads data, or reads
molecular structures.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


METRICS = ("direction_accuracy", "delta_mae", "delta_rmse")
WEIGHTINGS = ("pair", "equal_group")
SPLIT_SEEDS = (20260624, 20260625, 20260626, 20260724, 20260801)
FOLDS = (0, 1, 2, 3, 4)
MODEL_SEEDS = (20260624, 20260724, 20260801)

FULL_STREAMS: Mapping[str, tuple[int, ...]] = {
    "zero_delta": (-1,),
    "train_mean": (-1,),
    "source_prior": (-1,),
    "nearest_neighbor": (-1,),
    "ridge": (-1,),
    "random_forest": MODEL_SEEDS,
    "xgboost": MODEL_SEEDS,
    "permuted_xgboost": (20260803,),
}
E2_STREAMS: Mapping[str, tuple[int, ...]] = {
    "ridge": (-1,),
    "random_forest": MODEL_SEEDS,
    "xgboost": MODEL_SEEDS,
}

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
BASE_KEYS = ("split_seed", "fold", "pair_id")
STREAM_KEYS = ("model", "model_seed")
ROW_KEYS = (*STREAM_KEYS, *BASE_KEYS)


class AnalysisError(RuntimeError):
    """Raised when an input cannot support the requested fixed analysis."""


@dataclass(frozen=True)
class DrawPlan:
    weights: np.ndarray
    components: tuple[str, ...]
    source_name: str
    source_bytes: int
    source_sha256: str
    recorded_seed: int | None


@dataclass(frozen=True)
class MetricResult:
    values: np.ndarray
    model_seeds: tuple[int, ...]
    point_support_by_seed: tuple[int, ...]


@dataclass(frozen=True)
class AnalysisOutputs:
    descriptive: list[dict[str, Any]]
    contrasts: list[dict[str, Any]]
    diagnostics: list[dict[str, Any]]
    draw_values: np.ndarray
    draw_columns: tuple[str, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise AnalysisError(f"{label} is missing columns: {', '.join(missing)}")


def _coerce_integer(frame: pd.DataFrame, column: str, label: str) -> None:
    try:
        numeric = pd.to_numeric(frame[column], errors="raise")
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"{label}.{column} must contain integers") from exc
    values = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise AnalysisError(f"{label}.{column} must contain finite integers")
    frame[column] = numeric.astype(np.int64)


def _coerce_float(frame: pd.DataFrame, column: str, label: str) -> None:
    try:
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"{label}.{column} must contain numeric values") from exc
    if not np.isfinite(values).all():
        raise AnalysisError(f"{label}.{column} contains a non-finite value")
    frame[column] = values


def read_prediction_csv(
    path: Path,
    *,
    label: str,
    expected_streams: Mapping[str, Sequence[int]],
    allow_other_models: bool = False,
    expected_pairs: int = 874,
    expected_groups: int = 838,
    expected_rows_per_stream: int = 4_370,
) -> pd.DataFrame:
    """Load and validate one fixed-study prediction table.

    Extra columns are ignored.  ``allow_other_models`` is used only when a
    full eight-model binary table is supplied to the E2 command, which selects
    the three representation-comparison models before enforcing the contract.
    """

    path = path.resolve()
    if not path.is_file():
        raise AnalysisError(f"{label} does not exist: {path}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)
    _require_columns(frame, REQUIRED_PREDICTION_COLUMNS, label)
    frame = frame[list(REQUIRED_PREDICTION_COLUMNS)].copy()

    expected_models = set(expected_streams)
    observed_models = set(frame["model"].astype(str))
    if allow_other_models:
        missing = expected_models - observed_models
        if missing:
            raise AnalysisError(f"{label} is missing models: {', '.join(sorted(missing))}")
        frame = frame[frame["model"].isin(expected_models)].copy()
    elif observed_models != expected_models:
        raise AnalysisError(
            f"{label} model set differs from the fixed contract "
            f"(missing={sorted(expected_models - observed_models)}, "
            f"extra={sorted(observed_models - expected_models)})"
        )

    for column in ("pair_id", "comparison_pair_key", "component_id", "model"):
        frame[column] = frame[column].astype(str)
        if frame[column].str.strip().eq("").any():
            raise AnalysisError(f"{label}.{column} contains a blank value")
    for column in ("model_seed", "split_seed", "fold"):
        _coerce_integer(frame, column, label)
    for column in ("true_delta", "prediction"):
        _coerce_float(frame, column, label)
    if frame["true_delta"].eq(0).any():
        raise AnalysisError(f"{label}.true_delta must be nonzero")
    if frame.duplicated(list(ROW_KEYS)).any():
        raise AnalysisError(f"{label} contains duplicate saved-stream OOF keys")

    observed_streams = set(
        frame[["model", "model_seed"]].drop_duplicates().itertuples(index=False, name=None)
    )
    required_streams = {
        (model, int(seed)) for model, seeds in expected_streams.items() for seed in seeds
    }
    if observed_streams != required_streams:
        raise AnalysisError(
            f"{label} stream inventory differs from the fixed contract "
            f"(missing={sorted(required_streams - observed_streams)}, "
            f"extra={sorted(observed_streams - required_streams)})"
        )
    if tuple(sorted(frame["split_seed"].unique().tolist())) != SPLIT_SEEDS:
        raise AnalysisError(f"{label} split seeds differ from the fixed five-seed set")
    if tuple(sorted(frame["fold"].unique().tolist())) != FOLDS:
        raise AnalysisError(f"{label} folds differ from 0..4")

    stream_sizes = frame.groupby(list(STREAM_KEYS), sort=True).size()
    if not stream_sizes.eq(expected_rows_per_stream).all():
        bad = stream_sizes[~stream_sizes.eq(expected_rows_per_stream)].to_dict()
        raise AnalysisError(f"{label} has incomplete saved streams: {bad}")
    per_split = frame.groupby([*STREAM_KEYS, "split_seed"], sort=True).size()
    if not per_split.eq(expected_pairs).all():
        raise AnalysisError(f"{label} does not contain {expected_pairs} rows per saved stream/split")

    base = frame[list(BASE_KEYS)].drop_duplicates()
    if len(base) != expected_rows_per_stream:
        raise AnalysisError(
            f"{label} has {len(base)} OOF keys; expected {expected_rows_per_stream}"
        )
    if base["pair_id"].nunique() != expected_pairs:
        raise AnalysisError(f"{label} does not contain exactly {expected_pairs} pair identifiers")
    pair_by_split = base.groupby("split_seed", sort=True)["pair_id"].nunique()
    if not pair_by_split.eq(expected_pairs).all():
        raise AnalysisError(f"{label} pair universe changes across split seeds")
    if not frame.groupby(["split_seed", "pair_id"], sort=False)["fold"].nunique().eq(1).all():
        raise AnalysisError(f"{label} assigns one pair to multiple OOF folds within a split")

    stream_count = len(required_streams)
    base_counts = frame.groupby(list(BASE_KEYS), sort=False).size()
    if not base_counts.eq(stream_count).all():
        raise AnalysisError(f"{label} saved streams do not share one exact OOF-key universe")
    for column in ("comparison_pair_key", "component_id", "true_delta"):
        if not frame.groupby(list(BASE_KEYS), sort=False)[column].nunique(dropna=False).eq(1).all():
            raise AnalysisError(f"{label}.{column} changes across saved streams")
        if not frame.groupby("pair_id", sort=False)[column].nunique(dropna=False).eq(1).all():
            raise AnalysisError(f"{label}.{column} changes across split seeds for a fixed pair_id")
    if not frame.groupby(["split_seed", "component_id"], sort=False)["fold"].nunique().eq(1).all():
        raise AnalysisError(f"{label} assigns one graph component to multiple folds within a split")
    group_components = frame.groupby(
        ["split_seed", "comparison_pair_key"], sort=False
    )["component_id"].nunique(dropna=False)
    if not group_components.eq(1).all():
        raise AnalysisError(f"{label} has an equal-comparison group spanning components")
    groups_by_split = (
        frame[list(BASE_KEYS) + ["comparison_pair_key"]]
        .drop_duplicates(list(BASE_KEYS))
        .groupby("split_seed", sort=True)["comparison_pair_key"]
        .nunique()
    )
    if not groups_by_split.eq(expected_groups).all():
        raise AnalysisError(
            f"{label} does not contain {expected_groups} comparison groups per split"
        )
    return frame.sort_values(list(ROW_KEYS), kind="stable", ignore_index=True)


def _read_npz_plan(path: Path) -> tuple[np.ndarray, tuple[str, ...], int | None]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            required = {"weights", "components"}
            if not required.issubset(archive.files):
                raise AnalysisError("NPZ draw plan requires weights and components arrays")
            weights = np.asarray(archive["weights"])
            components = tuple(str(value) for value in archive["components"].tolist())
            seed = int(np.asarray(archive["seed"]).item()) if "seed" in archive.files else None
    except (OSError, ValueError) as exc:
        raise AnalysisError(f"cannot read NPZ draw plan: {path}") from exc
    return weights, components, seed


def _read_csv_plan(path: Path) -> tuple[np.ndarray, tuple[str, ...], int | None]:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)
    _require_columns(frame, ("replicate_id", "component_id", "draw_count"), "draw plan")
    frame = frame[["replicate_id", "component_id", "draw_count"]].copy()
    if frame["component_id"].str.strip().eq("").any():
        raise AnalysisError("draw plan contains a blank component_id")
    _coerce_integer(frame, "replicate_id", "draw plan")
    _coerce_integer(frame, "draw_count", "draw plan")
    if (frame["replicate_id"] < 0).any() or (frame["draw_count"] < 0).any():
        raise AnalysisError("draw plan contains a negative replicate id or draw count")
    if frame.duplicated(["replicate_id", "component_id"]).any():
        raise AnalysisError("draw plan contains duplicate replicate/component cells")
    replicate_ids = sorted(frame["replicate_id"].unique().tolist())
    if replicate_ids != list(range(len(replicate_ids))):
        raise AnalysisError("draw plan replicate_id values must be contiguous from zero")
    components = tuple(sorted(frame["component_id"].unique().tolist()))
    expected_rows = len(replicate_ids) * len(components)
    if len(frame) != expected_rows:
        raise AnalysisError("draw plan is missing one or more replicate/component cells")
    pivot = frame.pivot(index="replicate_id", columns="component_id", values="draw_count")
    pivot = pivot.reindex(index=replicate_ids, columns=list(components))
    if pivot.isna().any().any():
        raise AnalysisError("draw plan is incomplete after pivoting")
    return pivot.to_numpy(dtype=np.int64), components, None


def read_draw_plan(
    path: Path,
    *,
    expected_replicates: int,
    expected_components: int,
) -> DrawPlan:
    """Load a saved plan without generating or replacing any draws."""

    path = path.resolve()
    if not path.is_file():
        raise AnalysisError(f"draw plan does not exist: {path}")
    if path.suffix.lower() == ".npz":
        weights, components, seed = _read_npz_plan(path)
    elif path.suffix.lower() == ".csv":
        weights, components, seed = _read_csv_plan(path)
    else:
        raise AnalysisError("draw plan must be .npz or long-form .csv")

    if weights.ndim != 2:
        raise AnalysisError(f"draw-plan weights must be two-dimensional, got {weights.shape}")
    if weights.shape != (expected_replicates, expected_components):
        raise AnalysisError(
            f"draw-plan shape is {weights.shape}; expected "
            f"{(expected_replicates, expected_components)}"
        )
    if len(components) != expected_components or len(set(components)) != expected_components:
        raise AnalysisError("draw-plan component identifiers are missing or duplicated")
    if any(not component.strip() for component in components):
        raise AnalysisError("draw-plan component identifiers must be nonempty")
    if not np.issubdtype(weights.dtype, np.integer):
        if not np.isfinite(weights).all() or not np.equal(weights, np.floor(weights)).all():
            raise AnalysisError("draw-plan weights must contain finite integers")
        weights = weights.astype(np.int64)
    else:
        weights = weights.astype(np.int64, copy=False)
    if (weights < 0).any():
        raise AnalysisError("draw-plan weights contain a negative value")
    if not np.equal(weights.sum(axis=1, dtype=np.int64), expected_components).all():
        raise AnalysisError(
            f"each draw-plan row must contain exactly {expected_components} component tickets"
        )
    return DrawPlan(
        weights=weights,
        components=components,
        source_name=path.name,
        source_bytes=path.stat().st_size,
        source_sha256=sha256_file(path),
        recorded_seed=seed,
    )


def validate_component_support(
    frames: Mapping[str, pd.DataFrame],
    plan: DrawPlan,
    *,
    expected_supported_components: int,
) -> dict[str, int]:
    plan_components = set(plan.components)
    supported_sets: dict[str, set[str]] = {}
    for label, frame in frames.items():
        observed = set(frame["component_id"].astype(str))
        extra = observed - plan_components
        if extra:
            raise AnalysisError(
                f"{label} contains components absent from the saved plan: {sorted(extra)[:10]}"
            )
        if len(observed) != expected_supported_components:
            raise AnalysisError(
                f"{label} has {len(observed)} supported components; "
                f"expected {expected_supported_components}"
            )
        supported_sets[label] = observed
    first_label = next(iter(supported_sets))
    first = supported_sets[first_label]
    for label, observed in supported_sets.items():
        if observed != first:
            raise AnalysisError(
                f"{label} and {first_label} do not use the same supported component universe"
            )
    return {
        "plan_components": len(plan.components),
        "supported_components": len(first),
        "empty_components_retained": len(plan.components) - len(first),
    }


def validate_paired_representations(
    count_frame: pd.DataFrame,
    binary_frame: pd.DataFrame,
) -> None:
    """Require one-to-one count/binary pairing before E2 aggregation."""

    keys = list(ROW_KEYS)
    metadata = ["comparison_pair_key", "component_id", "true_delta"]
    joined = count_frame[keys + metadata].merge(
        binary_frame[keys + metadata],
        on=keys,
        how="outer",
        suffixes=("_count", "_binary"),
        indicator=True,
        validate="one_to_one",
        sort=True,
    )
    if len(joined) != len(count_frame) or not joined["_merge"].eq("both").all():
        raise AnalysisError("E2 count and binary tables do not have identical saved-stream OOF keys")
    for column in ("comparison_pair_key", "component_id"):
        if not joined[f"{column}_count"].equals(joined[f"{column}_binary"]):
            raise AnalysisError(f"E2 count and binary tables disagree on {column}")
    if not np.allclose(
        joined["true_delta_count"].to_numpy(dtype=np.float64),
        joined["true_delta_binary"].to_numpy(dtype=np.float64),
        atol=1e-12,
        rtol=0.0,
    ):
        raise AnalysisError("E2 count and binary tables disagree on true_delta beyond atol=1e-12")


def _loss_units(frame: pd.DataFrame, weighting: str) -> pd.DataFrame:
    if weighting not in WEIGHTINGS:
        raise AnalysisError(f"unknown weighting: {weighting}")
    values = frame[["model_seed", "split_seed", "component_id", "comparison_pair_key"]].copy()
    truth = frame["true_delta"].to_numpy(dtype=np.float64)
    prediction = frame["prediction"].to_numpy(dtype=np.float64)
    values["direction_accuracy"] = (
        (prediction != 0) & (np.sign(prediction) == np.sign(truth))
    ).astype(np.float64)
    values["delta_mae"] = np.abs(prediction - truth)
    values["delta_rmse"] = (prediction - truth) ** 2
    if weighting == "equal_group":
        group_keys = ["model_seed", "split_seed", "component_id", "comparison_pair_key"]
        values = values.groupby(group_keys, as_index=False, sort=True)[list(METRICS)].mean()
    return values


def _metric_result(
    frame: pd.DataFrame,
    weighting: str,
    plan: DrawPlan,
    *,
    label: str,
    diagnostics: list[dict[str, Any]],
) -> MetricResult:
    units = _loss_units(frame, weighting)
    component_index = list(plan.components)
    all_weights = np.vstack(
        [np.ones((1, len(component_index)), dtype=np.float64), plan.weights.astype(np.float64)]
    )
    streams: list[np.ndarray] = []
    seeds: list[int] = []
    point_supports: list[int] = []
    for seed, group in units.groupby("model_seed", sort=True):
        sums = (
            group.groupby("component_id", sort=True)[list(METRICS)]
            .sum()
            .reindex(component_index, fill_value=0.0)
            .to_numpy(dtype=np.float64)
        )
        counts = (
            group.groupby("component_id", sort=True)
            .size()
            .reindex(component_index, fill_value=0)
            .to_numpy(dtype=np.int64)
        )
        numerator = all_weights @ sums
        denominator = all_weights @ counts
        result = np.full((len(all_weights), len(METRICS)), np.nan, dtype=np.float64)
        valid = denominator > 0
        np.divide(
            numerator,
            denominator[:, None],
            out=result,
            where=denominator[:, None] > 0,
        )
        negative_mse = valid & (result[:, 2] < -1e-15)
        if negative_mse.any():
            raise AnalysisError(f"{label}/{weighting}/seed={seed} produced negative weighted MSE")
        result[valid, 2] = np.sqrt(np.maximum(result[valid, 2], 0.0))
        invalid_draws = ~np.isfinite(result[1:]).all(axis=1)
        diagnostics.append(
            {
                "label": label,
                "weight": weighting,
                "model_seed": int(seed),
                "zero_denominator_draws": int((denominator[1:] <= 0).sum()),
                "nonfinite_draws": int(invalid_draws.sum()),
            }
        )
        streams.append(result)
        seeds.append(int(seed))
        point_supports.append(int(denominator[0]))
    stacked = np.stack(streams, axis=0)
    # RMSE has already been square-rooted within each model-seed stream.
    mean_across_seeds = np.mean(stacked, axis=0, dtype=np.float64)
    return MetricResult(
        values=mean_across_seeds,
        model_seeds=tuple(seeds),
        point_support_by_seed=tuple(point_supports),
    )


def analyze_component_family(
    frames: Mapping[str, pd.DataFrame],
    plan: DrawPlan,
    *,
    comparisons: Sequence[tuple[str, str, str]],
    family_size: int,
) -> AnalysisOutputs:
    """Calculate descriptive metrics and a fixed favorable-positive family.

    Each comparison tuple is ``(comparison_name, first_label, second_label)``.
    Direction accuracy is ``first - second``; MAE and RMSE are
    ``second - first``.  Thus positive values favor the first label.
    """

    if family_size != len(comparisons) * len(WEIGHTINGS) * len(METRICS):
        raise AnalysisError("family_size does not match comparisons x weightings x metrics")
    if not frames:
        raise AnalysisError("no analysis frames were supplied")

    cache: dict[tuple[str, str], MetricResult] = {}
    descriptive: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for weighting in WEIGHTINGS:
        for label, frame in frames.items():
            result = _metric_result(
                frame,
                weighting,
                plan,
                label=label,
                diagnostics=diagnostics,
            )
            cache[(label, weighting)] = result
            for metric_index, metric in enumerate(METRICS):
                descriptive.append(
                    {
                        "label": label,
                        "weight": weighting,
                        "metric": metric,
                        "value": float(result.values[0, metric_index]),
                        "model_seed_count": len(result.model_seeds),
                        "model_seeds": ";".join(str(seed) for seed in result.model_seeds),
                        "support_per_model_seed": ";".join(
                            str(value) for value in result.point_support_by_seed
                        ),
                        "aggregation": (
                            "metric within each model-seed stream over supplied splits; "
                            "RMSE square root before equal mean over model seeds"
                        ),
                    }
                )

    alpha = 0.05 / (2 * family_size)
    contrasts: list[dict[str, Any]] = []
    draw_columns: list[str] = []
    draw_arrays: list[np.ndarray] = []
    direction = np.asarray([1.0, -1.0, -1.0], dtype=np.float64)
    for comparison, first_label, second_label in comparisons:
        if first_label not in frames or second_label not in frames:
            raise AnalysisError(f"comparison {comparison} refers to an unknown label")
        for weighting in WEIGHTINGS:
            first = cache[(first_label, weighting)]
            second = cache[(second_label, weighting)]
            if first.model_seeds != second.model_seeds:
                raise AnalysisError(
                    f"comparison {comparison}/{weighting} has unmatched model-seed streams"
                )
            values = (first.values - second.values) * direction
            for metric_index, metric in enumerate(METRICS):
                draws = values[1:, metric_index]
                valid = np.isfinite(draws)
                if valid.all():
                    lower, upper = np.quantile(
                        draws, [alpha, 1 - alpha], method="linear"
                    ).tolist()
                    interval_status = "COMPUTED_UNCALIBRATED"
                else:
                    lower, upper = float("nan"), float("nan")
                    interval_status = "NOT_ESTIMABLE_FAILED_DRAWS"
                column = f"{comparison}__{weighting}__{metric}"
                draw_columns.append(column)
                draw_arrays.append(draws)
                contrasts.append(
                    {
                        "comparison": comparison,
                        "weight": weighting,
                        "metric": metric,
                        "estimate": float(values[0, metric_index]),
                        "adjusted_lower": float(lower),
                        "adjusted_upper": float(upper),
                        "B": len(draws),
                        "family_size": family_size,
                        "tail_alpha": alpha,
                        "expected_tail_draws": len(draws) * alpha,
                        "invalid_draws": int((~valid).sum()),
                        "interval_status": interval_status,
                        "contrast_direction": (
                            "first_minus_second"
                            if metric == "direction_accuracy"
                            else "second_minus_first"
                        ),
                        "positive_favors": first_label,
                    }
                )
    family_complete = all(row["invalid_draws"] == 0 for row in contrasts)
    family_status = (
        "COMPLETE_ALL_INTERVALS"
        if family_complete
        else "INCOMPLETE_NONESTIMABLE_CELL_PRESENT"
    )
    for row in contrasts:
        row["family_status"] = family_status
    return AnalysisOutputs(
        descriptive=descriptive,
        contrasts=contrasts,
        diagnostics=diagnostics,
        draw_values=np.column_stack(draw_arrays),
        draw_columns=tuple(draw_columns),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise AnalysisError(f"refusing to write an empty table: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def prepare_output_dir(output_dir: Path) -> Path:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise AnalysisError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_outputs(
    output_dir: Path,
    outputs: AnalysisOutputs,
    *,
    analysis_id: str,
    plan: DrawPlan,
    component_summary: Mapping[str, int],
    input_files: Mapping[str, Path],
    limitations: Sequence[str],
    semantic_checks: Mapping[str, bool],
) -> dict[str, Any]:
    output_dir = prepare_output_dir(output_dir)
    _write_csv(output_dir / "descriptive_metrics.csv", outputs.descriptive)
    _write_csv(output_dir / "contrasts.csv", outputs.contrasts)
    _write_csv(output_dir / "draw_diagnostics.csv", outputs.diagnostics)
    np.savez_compressed(
        output_dir / "draw_contrasts.npz",
        values=outputs.draw_values,
        columns=np.asarray(outputs.draw_columns),
    )

    all_checks_pass = all(bool(value) for value in semantic_checks.values())
    no_failed_draws = all(
        row["zero_denominator_draws"] == 0 and row["nonfinite_draws"] == 0
        for row in outputs.diagnostics
    ) and all(row["invalid_draws"] == 0 for row in outputs.contrasts)
    receipt = {
        "analysis_id": analysis_id,
        "status": (
            "PASS_COMPUTATION_UNCALIBRATED_INTERVALS"
            if all_checks_pass and no_failed_draws
            else "FAIL_OR_NONESTIMABLE_REQUIRES_REVIEW"
        ),
        "operation_boundary": {
            "model_fits": 0,
            "predictions_created": 0,
            "bootstrap_draws_created": 0,
            "saved_prediction_rescoring_only": True,
        },
        "draw_plan": {
            "file": plan.source_name,
            "bytes": plan.source_bytes,
            "sha256": plan.source_sha256,
            "replicates": int(plan.weights.shape[0]),
            "components": int(plan.weights.shape[1]),
            "recorded_seed": plan.recorded_seed,
            "identity_note": (
                "The digest identifies the supplied saved plan. It does not recover "
                "unpreserved generator history or establish prespecification."
            ),
        },
        "component_support": dict(component_summary),
        "inputs": {
            role: {
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for role, path in input_files.items()
        },
        "outputs": {
            "descriptive_metrics.csv": len(outputs.descriptive),
            "contrasts.csv": len(outputs.contrasts),
            "draw_diagnostics.csv": len(outputs.diagnostics),
            "draw_contrasts.npz": {
                "shape": list(outputs.draw_values.shape),
                "columns": len(outputs.draw_columns),
            },
        },
        "semantic_checks": dict(semantic_checks),
        "failed_draw_policy": (
            "Any nonfinite or zero-denominator draw makes that interval non-estimable; "
            "draws are not silently dropped or replaced."
        ),
        "limitations": list(limitations),
    }
    (output_dir / "RUN_METADATA.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return receipt


def print_receipt(receipt: Mapping[str, Any]) -> None:
    print(json.dumps(receipt, indent=2, ensure_ascii=False))
