#!/usr/bin/env python3
"""Run the frozen 200-fit primary protocol from constructed TACK tables.

The command consumes only outputs of ``reproduce_construction.py`` and the
bundled author-defined fit plan.  It writes saved out-of-fold predictions for
the repository's existing component-rescoring command.  External-overlap and
contamination sensitivities are outside this primary fitting path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd

from lib.primary.training_core import (
    PrimaryProtocolError,
    finite_vector,
    load_fingerprints,
    make_model,
    nonfit_predictions,
    pair_features,
)


SPLIT_SEEDS = (20260624, 20260625, 20260626, 20260724, 20260801)
FOLDS = (0, 1, 2, 3, 4)
MODEL_SEEDS = (20260624, 20260724, 20260801)
BASELINES = ("zero_delta", "train_mean", "source_prior", "nearest_neighbor")
MODEL_IDENTITIES = (
    ("zero_delta", -1),
    ("train_mean", -1),
    ("source_prior", -1),
    ("nearest_neighbor", -1),
    ("ridge", -1),
    ("random_forest", 20260624),
    ("random_forest", 20260724),
    ("random_forest", 20260801),
    ("xgboost", 20260624),
    ("xgboost", 20260724),
    ("xgboost", 20260801),
    ("permuted_xgboost", 20260803),
)
EXPECTED_CELL_SIZES = {
    (20260624, 0): 219,
    (20260624, 1): 305,
    (20260624, 2): 161,
    (20260624, 3): 139,
    (20260624, 4): 50,
    (20260625, 0): 221,
    (20260625, 1): 304,
    (20260625, 2): 161,
    (20260625, 3): 150,
    (20260625, 4): 38,
    (20260626, 0): 221,
    (20260626, 1): 289,
    (20260626, 2): 169,
    (20260626, 3): 117,
    (20260626, 4): 78,
    (20260724, 0): 216,
    (20260724, 1): 290,
    (20260724, 2): 167,
    (20260724, 3): 151,
    (20260724, 4): 50,
    (20260801, 0): 218,
    (20260801, 1): 302,
    (20260801, 2): 162,
    (20260801, 3): 120,
    (20260801, 4): 72,
}
REQUIRED_CONSTRUCTION_FILES = (
    "records_aggregated_v0.1.csv",
    "pairs_v0.1.csv",
    "primary_split_membership.csv",
    "pair_metadata.csv",
    "record_morgan_fp_r2_2048.csv",
)
PREDICTION_COLUMNS = (
    "pair_id",
    "comparison_pair_key",
    "component_id",
    "record_id_i",
    "record_id_j",
    "true_delta",
    "prediction",
    "model",
    "model_seed",
    "split_seed",
    "fold",
    "max_train_tanimoto",
    "support_bin",
)
REVERSE_COLUMNS = (
    "pair_id",
    "record_id_i",
    "record_id_j",
    "reverse_record_id_i",
    "reverse_record_id_j",
    "model",
    "model_seed",
    "split_seed",
    "fold",
    "forward_prediction",
    "reverse_raw_prediction",
    "negative_forward_prediction",
    "antisymmetry_residual",
    "reverse_true_delta",
    "derived_not_independent",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--construction-dir",
        type=Path,
        required=True,
        help="Directory created by reproduce_construction.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory for predictions, ledgers, and receipt.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        help="Threads passed to RF/XGBoost; the recorded protocol used 4.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate inputs and prepare all 25 feature cells without fitting.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    # Match the historical primary executor's default high-precision parser.
    # Switching to round_trip changes last-bit targets and can change RF split ties.
    return pd.read_csv(path, keep_default_na=False, float_precision="high", low_memory=False)


def load_plan(resource_dir: Path) -> pd.DataFrame:
    plan = read_csv(resource_dir / "fit_cell_plan.csv")
    permutations = read_csv(resource_dir / "permutation_seed_table.csv")
    required = {
        "fit_ordinal",
        "cell_id",
        "model",
        "split_seed",
        "fold",
        "model_seed",
        "permutation_seed",
        "fit_budget_units",
    }
    if set(plan.columns) != required or len(plan) != 200:
        raise PrimaryProtocolError("bundled fit plan schema or row count is invalid")
    for column in ("fit_ordinal", "split_seed", "fold", "model_seed", "fit_budget_units"):
        plan[column] = pd.to_numeric(plan[column], errors="raise").astype(int)
    # pandas 3 uses an Arrow string dtype for this mixed blank/integer column;
    # make it object-backed before replacing the fixed nonblank cells by ints.
    plan["permutation_seed"] = plan["permutation_seed"].astype(object)
    if plan["fit_ordinal"].tolist() != list(range(1, 201)):
        raise PrimaryProtocolError("fit ordinals are not contiguous 1..200")
    if not plan["fit_budget_units"].eq(1).all():
        raise PrimaryProtocolError("fit plan budget units changed")
    expected_cells = {(seed, fold) for seed in SPLIT_SEEDS for fold in FOLDS}
    if set(zip(plan["split_seed"], plan["fold"])) != expected_cells:
        raise PrimaryProtocolError("fit plan cell universe changed")
    expected_per_cell = {
        ("ridge", -1),
        *(("random_forest", seed) for seed in MODEL_SEEDS),
        *(("xgboost", seed) for seed in MODEL_SEEDS),
        ("permuted_xgboost", 20260803),
    }
    for _, frame in plan.groupby(["split_seed", "fold"], sort=False):
        if set(zip(frame["model"], frame["model_seed"])) != expected_per_cell:
            raise PrimaryProtocolError("fit identities changed in a split/fold cell")
    for column in ("split_seed", "fold", "permutation_seed"):
        permutations[column] = pd.to_numeric(
            permutations[column], errors="raise"
        ).astype(int)
    expected_permutations = {
        (int(row.split_seed), int(row.fold)): int(row.permutation_seed)
        for row in permutations.itertuples(index=False)
    }
    if len(expected_permutations) != 25:
        raise PrimaryProtocolError("permutation seed table must contain 25 cells")
    for index, row in plan.iterrows():
        raw = str(row["permutation_seed"]).strip()
        if row["model"] == "permuted_xgboost":
            expected = expected_permutations[(row["split_seed"], row["fold"])]
            if not raw or int(raw) != expected:
                raise PrimaryProtocolError("permuted-XGBoost seed binding changed")
            plan.at[index, "permutation_seed"] = expected
        elif raw:
            raise PrimaryProtocolError("non-permuted fit unexpectedly has a permutation seed")
        else:
            plan.at[index, "permutation_seed"] = None
    return plan


def prepare_primary(construction: Path) -> pd.DataFrame:
    missing = [name for name in REQUIRED_CONSTRUCTION_FILES if not (construction / name).is_file()]
    if missing:
        raise FileNotFoundError(f"construction directory is missing {missing}")
    records = read_csv(construction / "records_aggregated_v0.1.csv")
    pairs = read_csv(construction / "pairs_v0.1.csv")
    splits = read_csv(construction / "primary_split_membership.csv")
    metadata = read_csv(construction / "pair_metadata.csv")
    if len(records) != 3302 or records["record_id"].nunique() != 3302:
        raise PrimaryProtocolError("constructed record universe changed")
    if len(pairs) != 6015 or pairs["pair_id"].nunique() != 6015:
        raise PrimaryProtocolError("constructed pair universe changed")
    is_cliff = pairs["is_cliff"].astype(str).str.lower().isin(
        ["true", "1", "pass", "yes"]
    )
    primary = pairs[is_cliff].copy()
    if (
        len(primary) != 874
        or primary["pair_id"].nunique() != 874
        or primary["comparison_pair_key"].nunique() != 838
    ):
        raise PrimaryProtocolError("primary pair/group universe changed")
    if len(metadata) != 874 or set(metadata["pair_id"]) != set(primary["pair_id"]):
        raise PrimaryProtocolError("pair metadata universe changed")
    primary = primary.merge(
        metadata[["pair_id", "source_dataset"]], on="pair_id", validate="one_to_one"
    )
    activity = records.set_index("record_id")["pdc50_median"].astype(float)
    target = primary["record_id_j"].map(activity) - primary["record_id_i"].map(activity)
    legacy = primary["true_delta"].astype(float).to_numpy()
    if (
        target.isna().any()
        or not np.isfinite(target.to_numpy()).all()
        or not np.allclose(target.to_numpy() + legacy, 0, atol=1e-12, rtol=0)
        or (target.to_numpy() == 0).any()
    ):
        raise PrimaryProtocolError("pDC50(j)-pDC50(i) target reconciliation failed")
    primary["target_b_minus_a"] = target.to_numpy()
    for column in ("split_seed", "heldout_fold"):
        splits[column] = pd.to_numeric(splits[column], errors="raise").astype(int)
    merge_keys = ["pair_id", "comparison_pair_key", "record_id_i", "record_id_j"]
    merged = primary.merge(splits, on=merge_keys, validate="one_to_many")
    if len(merged) != 4370 or merged[["split_seed", "pair_id"]].duplicated().any():
        raise PrimaryProtocolError("primary OOF membership is not 5 x 874")
    actual_sizes = {
        (int(seed), int(fold)): int(count)
        for (seed, fold), count in merged.groupby(
            ["split_seed", "heldout_fold"]
        ).size().items()
    }
    if actual_sizes != EXPECTED_CELL_SIZES:
        raise PrimaryProtocolError("split/fold cell sizes differ from the frozen plan")
    for column in ("component_id", "comparison_pair_key"):
        if merged.groupby(["split_seed", column])["heldout_fold"].nunique().max() != 1:
            raise PrimaryProtocolError(f"{column} crosses folds within a split")
    endpoints = pd.concat(
        [
            merged[["split_seed", "heldout_fold", "record_id_i"]].rename(
                columns={"record_id_i": "record_id"}
            ),
            merged[["split_seed", "heldout_fold", "record_id_j"]].rename(
                columns={"record_id_j": "record_id"}
            ),
        ]
    )
    if endpoints.groupby(["split_seed", "record_id"])["heldout_fold"].nunique().max() != 1:
        raise PrimaryProtocolError("an endpoint crosses folds within a split")
    return merged


def build_cells(
    merged: pd.DataFrame,
    fingerprint_index: dict[str, int],
    fingerprints: np.ndarray,
) -> dict[tuple[int, int], dict[str, object]]:
    cells: dict[tuple[int, int], dict[str, object]] = {}
    for split_seed in SPLIT_SEEDS:
        split = merged[merged["split_seed"] == split_seed]
        for fold in FOLDS:
            test = split[split["heldout_fold"] == fold].copy()
            train = split[split["heldout_fold"] != fold].copy()
            expected = EXPECTED_CELL_SIZES[(split_seed, fold)]
            if len(test) != expected or len(train) != 874 - expected:
                raise PrimaryProtocolError("train/test cell cardinality changed")
            x_train = pair_features(train, fingerprint_index, fingerprints)
            x_test = pair_features(test, fingerprint_index, fingerprints)
            y_train = finite_vector(
                train["target_b_minus_a"].to_numpy(float), len(train), "training target"
            )
            baselines, maximum, support = nonfit_predictions(
                train, test, x_train, x_test, y_train
            )
            cells[(split_seed, fold)] = {
                "train": train,
                "test": test,
                "x_train": x_train,
                "x_test": x_test,
                "x_reverse": (-x_test).astype(np.float32, copy=False),
                "y_train": y_train,
                "baselines": baselines,
                "maximum": maximum,
                "support": support,
            }
    return cells


def prediction_rows(
    test: pd.DataFrame,
    values: np.ndarray,
    model: str,
    model_seed: int,
    split_seed: int,
    maximum: np.ndarray,
    support: list[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, row in enumerate(test.itertuples(index=False)):
        rows.append(
            {
                "pair_id": row.pair_id,
                "comparison_pair_key": row.comparison_pair_key,
                "component_id": row.component_id,
                "record_id_i": row.record_id_i,
                "record_id_j": row.record_id_j,
                "true_delta": float(row.target_b_minus_a),
                "prediction": float(values[index]),
                "model": model,
                "model_seed": model_seed,
                "split_seed": split_seed,
                "fold": int(row.heldout_fold),
                "max_train_tanimoto": float(maximum[index]),
                "support_bin": support[index],
            }
        )
    return rows


def write_frame(path: Path, rows: list[dict[str, object]], columns: tuple[str, ...]) -> None:
    pd.DataFrame(rows, columns=columns).to_csv(
        path, index=False, lineterminator="\n", float_format="%.17g"
    )


def main() -> None:
    args = parse_args()
    construction = args.construction_dir.resolve()
    output = args.output_dir.resolve()
    if args.n_jobs != 4:
        raise RuntimeError("the frozen fitting protocol requires --n-jobs 4")
    if output.exists():
        raise RuntimeError("output directory already exists; preserve prior attempts")
    output.mkdir(parents=True)
    resources = Path(__file__).resolve().parent / "lib" / "primary"
    receipt: dict[str, object] = {
        "status": "STARTED",
        "analysis_id": "portable_primary_200_fit_protocol",
        "operation_boundary": {
            "new_data": 0,
            "hyperparameter_searches": 0,
            "planned_model_fits": 0 if args.validate_only else 200,
            "bootstrap_draws": 0,
        },
        "orientation": "pDC50(j)-pDC50(i)",
        "a1_a2_external_sensitivities_included": False,
    }
    started = time.perf_counter()
    try:
        plan = load_plan(resources)
        merged = prepare_primary(construction)
        required_records = set(merged["record_id_i"]) | set(merged["record_id_j"])
        fingerprint_index, fingerprints = load_fingerprints(
            construction / "record_morgan_fp_r2_2048.csv", required_records
        )
        cells = build_cells(merged, fingerprint_index, fingerprints)
        shutil.copyfile(resources / "fit_cell_plan.csv", output / "fit_cell_plan.csv")
        shutil.copyfile(
            resources / "permutation_seed_table.csv",
            output / "permutation_seed_table.csv",
        )
        if args.validate_only:
            receipt.update(
                status="PASS_VALIDATION_ONLY_NO_FITS",
                counts={
                    "pairs": 874,
                    "split_copies": len(merged),
                    "cells": len(cells),
                    "fit_plan_rows": len(plan),
                    "model_fits": 0,
                },
            )
            return

        predictions: list[dict[str, object]] = []
        reversals: list[dict[str, object]] = []
        fit_ledger: list[dict[str, object]] = []
        specs_by_cell = {
            key: frame.sort_values("fit_ordinal", kind="mergesort")
            for key, frame in plan.groupby(["split_seed", "fold"], sort=False)
        }
        for split_seed in SPLIT_SEEDS:
            for fold in FOLDS:
                cell = cells[(split_seed, fold)]
                train = cell["train"]
                test = cell["test"]
                x_train = cell["x_train"]
                x_test = cell["x_test"]
                x_reverse = cell["x_reverse"]
                y_train = cell["y_train"]
                maximum = cell["maximum"]
                support = cell["support"]
                for name in BASELINES:
                    predictions.extend(
                        prediction_rows(
                            test,
                            cell["baselines"][name],
                            name,
                            -1,
                            split_seed,
                            maximum,
                            support,
                        )
                    )
                for spec in specs_by_cell[(split_seed, fold)].itertuples(index=False):
                    began_fit = time.perf_counter()
                    model_seed = int(spec.model_seed)
                    y_fit = y_train
                    if spec.model == "permuted_xgboost":
                        y_fit = np.random.default_rng(
                            int(spec.permutation_seed)
                        ).permutation(y_train)
                    model = make_model(
                        str(spec.model), model_seed, n_jobs=args.n_jobs
                    )
                    model.fit(x_train, y_fit)
                    values = finite_vector(
                        model.predict(x_test), len(test), str(spec.cell_id)
                    )
                    reverse = finite_vector(
                        model.predict(x_reverse),
                        len(test),
                        str(spec.cell_id) + ":reverse",
                    )
                    predictions.extend(
                        prediction_rows(
                            test,
                            values,
                            str(spec.model),
                            model_seed,
                            split_seed,
                            maximum,
                            support,
                        )
                    )
                    for row_index, source in enumerate(test.itertuples(index=False)):
                        reversals.append(
                            {
                                "pair_id": source.pair_id,
                                "record_id_i": source.record_id_i,
                                "record_id_j": source.record_id_j,
                                "reverse_record_id_i": source.record_id_j,
                                "reverse_record_id_j": source.record_id_i,
                                "model": spec.model,
                                "model_seed": model_seed,
                                "split_seed": split_seed,
                                "fold": fold,
                                "forward_prediction": float(values[row_index]),
                                "reverse_raw_prediction": float(reverse[row_index]),
                                "negative_forward_prediction": float(-values[row_index]),
                                "antisymmetry_residual": float(
                                    reverse[row_index] + values[row_index]
                                ),
                                "reverse_true_delta": float(
                                    -source.target_b_minus_a
                                ),
                                "derived_not_independent": True,
                            }
                        )
                    fit_ledger.append(
                        {
                            "fit_ordinal": int(spec.fit_ordinal),
                            "cell_id": spec.cell_id,
                            "model": spec.model,
                            "split_seed": split_seed,
                            "fold": fold,
                            "model_seed": model_seed,
                            "permutation_seed": spec.permutation_seed,
                            "train_rows": len(train),
                            "test_rows": len(test),
                            "status": "SUCCESS",
                            "seconds": time.perf_counter() - began_fit,
                        }
                    )

        identity_rank = {identity: rank for rank, identity in enumerate(MODEL_IDENTITIES)}
        predictions.sort(
            key=lambda row: (
                SPLIT_SEEDS.index(int(row["split_seed"])),
                FOLDS.index(int(row["fold"])),
                identity_rank[(str(row["model"]), int(row["model_seed"]))],
                str(row["pair_id"]),
            )
        )
        reversals.sort(
            key=lambda row: (
                SPLIT_SEEDS.index(int(row["split_seed"])),
                FOLDS.index(int(row["fold"])),
                identity_rank[(str(row["model"]), int(row["model_seed"]))],
                str(row["pair_id"]),
            )
        )
        prediction_frame = pd.DataFrame(predictions, columns=PREDICTION_COLUMNS)
        if (
            len(prediction_frame) != 52440
            or prediction_frame.duplicated(
                ["model", "model_seed", "split_seed", "fold", "pair_id"]
            ).any()
            or not np.isfinite(
                prediction_frame[["true_delta", "prediction"]].to_numpy(float)
            ).all()
        ):
            raise PrimaryProtocolError("final prediction universe is incomplete")
        write_frame(output / "predictions.csv", predictions, PREDICTION_COLUMNS)
        write_frame(output / "reverse_predictions.csv", reversals, REVERSE_COLUMNS)
        pd.DataFrame(fit_ledger).sort_values("fit_ordinal").to_csv(
            output / "fit_ledger.csv",
            index=False,
            lineterminator="\n",
            float_format="%.17g",
        )
        receipt.update(
            status="PASS_PRIMARY_200_FITS",
            counts={
                "pairs": 874,
                "split_copies": 4370,
                "cells": 25,
                "model_fits": len(fit_ledger),
                "forward_prediction_rows": len(predictions),
                "reverse_prediction_rows": len(reversals),
                "prediction_streams": 12,
            },
            outputs={
                "predictions.csv": len(predictions),
                "reverse_predictions.csv": len(reversals),
                "fit_ledger.csv": len(fit_ledger),
            },
            limitations=[
                "Primary component intervals require the separately retained 2,000 x 77 draw plan and the repository rescoring command.",
                "The two external-overlap membership columns in the historical prediction table are intentionally outside this TACK-only fitting path.",
            ],
        )
    except Exception as error:
        receipt.update(
            status="FAILED_PRESERVED",
            error_type=type(error).__name__,
            error=str(error),
        )
        raise
    finally:
        receipt["elapsed_seconds"] = time.perf_counter() - started
        with (output / "PRIMARY_RUN.json").open(
            "w", encoding="utf-8", newline="\n"
        ) as stream:
            json.dump(receipt, stream, indent=2, ensure_ascii=False)
            stream.write("\n")


if __name__ == "__main__":
    main()
