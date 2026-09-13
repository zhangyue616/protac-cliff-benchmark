#!/usr/bin/env python3
"""Evaluate identity and recorded-assay subsets on fixed OOF predictions.

The analysis is post hoc and descriptive. It never fits a model or changes a
split: for each split/fold, the training population is the other folds of the
same split's 874 primary-cliff pairs. Subset flags affect evaluation rows only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


PRIMARY_PAIR_COUNT = 874
EXPECTED_SPLIT_SEEDS = (20260624, 20260625, 20260626, 20260724, 20260801)
EXPECTED_FOLDS = (0, 1, 2, 3, 4)

MODEL_STREAMS = {
    "zero_delta": (-1,),
    "train_mean": (-1,),
    "source_prior": (-1,),
    "nearest_neighbor": (-1,),
    "ridge": (-1,),
    "random_forest": (20260624, 20260724, 20260801),
    "xgboost": (20260624, 20260724, 20260801),
    "permuted_xgboost": (20260803,),
}
MODEL_LABELS = {
    "zero_delta": "Zero delta",
    "train_mean": "Training mean",
    "source_prior": "Source prior",
    "nearest_neighbor": "Nearest neighbor",
    "ridge": "Ridge",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
    "permuted_xgboost": "Permuted XGBoost",
}
TOTAL_SAVED_STREAMS = sum(len(seeds) for seeds in MODEL_STREAMS.values())

ASSAY_MISSING_SENTINEL = "MISSING_ASSAY_ID"
ASSAY_KNOWN_PATTERN = re.compile(r"^RAWASSAY:[0-9A-F]{64}$")
IDENTITY_PATTERN = re.compile(r"^[0-9A-F]{64}$")

SUBSETS = (
    ("all_oof", "in_all_oof"),
    (
        "no_unordered_identity_pair_in_training",
        "in_no_unordered_identity_pair_in_training",
    ),
    (
        "both_endpoint_identities_absent_from_training",
        "in_both_endpoint_identities_absent_from_training",
    ),
    ("both_assay_ids_known", "in_both_assay_ids_known"),
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
)
MEMBERSHIP_COLUMNS = (
    "split_seed",
    "heldout_fold",
    "pair_id",
    "comparison_pair_key",
    "component_id",
    "record_id_i",
    "record_id_j",
)
PAIR_MAP_COLUMNS = (
    "pair_id",
    "comparison_pair_key",
    "record_id_i",
    "record_id_j",
    "normalized_poi_id",
    "true_delta",
    "is_cliff",
)
RECORD_COLUMNS = ("record_id", "identity_hash", "assay_id_or_text")


class AnalysisInputError(RuntimeError):
    """Raised when inputs do not satisfy the fixed-study contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisInputError(message)


def require_columns(
    available: Iterable[str], required: Iterable[str], role: str
) -> None:
    missing = sorted(set(required) - set(available))
    require(not missing, f"{role} is missing required columns: {missing}")


def read_selected_csv(path: Path, role: str, columns: Sequence[str]) -> pd.DataFrame:
    require(path.is_file(), f"{role} input is not a readable file: {path}")
    try:
        header = pd.read_csv(path, nrows=0, keep_default_na=False)
        require_columns(header.columns, columns, role)
        return pd.read_csv(
            path,
            usecols=list(columns),
            dtype=str,
            keep_default_na=False,
        )
    except AnalysisInputError:
        raise
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise AnalysisInputError(f"could not read {role} CSV {path}: {exc}") from exc


def require_nonempty(frame: pd.DataFrame, columns: Iterable[str], role: str) -> None:
    for column in columns:
        blank_count = int(frame[column].astype(str).str.strip().eq("").sum())
        require(
            blank_count == 0,
            f"{role}.{column} contains {blank_count} blank value(s)",
        )


def coerce_integer_columns(
    frame: pd.DataFrame, columns: Iterable[str], role: str
) -> None:
    for column in columns:
        try:
            values = pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise AnalysisInputError(
                f"{role}.{column} must contain integer values"
            ) from exc
        as_float = values.to_numpy(dtype=np.float64)
        require(
            np.isfinite(as_float).all() and np.equal(as_float, np.floor(as_float)).all(),
            f"{role}.{column} must contain finite integer values",
        )
        frame[column] = values.astype("int64")


def coerce_float_columns(
    frame: pd.DataFrame, columns: Iterable[str], role: str
) -> None:
    for column in columns:
        try:
            values = pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise AnalysisInputError(
                f"{role}.{column} must contain numeric values"
            ) from exc
        require(
            np.isfinite(values.to_numpy(dtype=np.float64)).all(),
            f"{role}.{column} contains a nonfinite value",
        )
        frame[column] = values.astype("float64")


def require_unique(frame: pd.DataFrame, keys: Sequence[str], role: str) -> None:
    duplicate_rows = int(frame.duplicated(list(keys), keep=False).sum())
    require(
        duplicate_rows == 0,
        f"{role} contains {duplicate_rows} rows duplicated on {list(keys)}",
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest().upper()


def load_inputs(
    predictions_path: Path,
    membership_path: Path,
    pair_map_path: Path,
    records_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions = read_selected_csv(
        predictions_path, "predictions", PREDICTION_COLUMNS
    )
    membership = read_selected_csv(
        membership_path, "membership", MEMBERSHIP_COLUMNS
    )
    pair_map = read_selected_csv(pair_map_path, "pair-map", PAIR_MAP_COLUMNS)
    records = read_selected_csv(records_path, "records", RECORD_COLUMNS)

    coerce_integer_columns(
        predictions, ("model_seed", "split_seed", "fold"), "predictions"
    )
    coerce_integer_columns(
        membership, ("split_seed", "heldout_fold"), "membership"
    )
    coerce_float_columns(predictions, ("true_delta", "prediction"), "predictions")
    coerce_float_columns(pair_map, ("true_delta",), "pair-map")

    require_nonempty(
        predictions,
        (
            "pair_id",
            "comparison_pair_key",
            "component_id",
            "record_id_i",
            "record_id_j",
            "model",
        ),
        "predictions",
    )
    require_nonempty(
        membership,
        (
            "pair_id",
            "comparison_pair_key",
            "component_id",
            "record_id_i",
            "record_id_j",
        ),
        "membership",
    )
    require_nonempty(
        pair_map,
        (
            "pair_id",
            "comparison_pair_key",
            "record_id_i",
            "record_id_j",
            "normalized_poi_id",
            "is_cliff",
        ),
        "pair-map",
    )
    require_nonempty(records, RECORD_COLUMNS, "records")
    return predictions, membership, pair_map, records


def annotate_membership(
    membership: pd.DataFrame,
    pair_map: pd.DataFrame,
    records: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    expected_membership_rows = PRIMARY_PAIR_COUNT * len(EXPECTED_SPLIT_SEEDS)
    require(
        len(pair_map) == PRIMARY_PAIR_COUNT,
        f"pair-map must contain exactly {PRIMARY_PAIR_COUNT} primary cliffs; "
        f"observed {len(pair_map)} rows",
    )
    require(
        len(membership) == expected_membership_rows,
        f"membership must contain {expected_membership_rows} pair-by-split rows; "
        f"observed {len(membership)}",
    )
    require_unique(membership, ("split_seed", "pair_id"), "membership")
    require_unique(pair_map, ("pair_id",), "pair-map")
    require_unique(records, ("record_id",), "records")

    observed_split_seeds = tuple(sorted(int(v) for v in membership["split_seed"].unique()))
    observed_folds = tuple(sorted(int(v) for v in membership["heldout_fold"].unique()))
    require(
        observed_split_seeds == EXPECTED_SPLIT_SEEDS,
        "membership split_seed values do not match the fixed-study seeds; "
        f"expected {list(EXPECTED_SPLIT_SEEDS)}, observed {list(observed_split_seeds)}",
    )
    require(
        observed_folds == EXPECTED_FOLDS,
        "membership heldout_fold values do not match the fixed five-fold design; "
        f"expected {list(EXPECTED_FOLDS)}, observed {list(observed_folds)}",
    )

    pair_universe = set(pair_map["pair_id"])
    require(
        set(membership["pair_id"]) == pair_universe,
        "membership and pair-map pair_id universes differ",
    )
    for split_seed in EXPECTED_SPLIT_SEEDS:
        split_rows = membership[membership["split_seed"] == split_seed]
        require(
            len(split_rows) == PRIMARY_PAIR_COUNT
            and split_rows["pair_id"].nunique() == PRIMARY_PAIR_COUNT,
            f"split {split_seed} must contain each of the {PRIMARY_PAIR_COUNT} "
            "primary pairs exactly once",
        )
        missing_folds = sorted(set(EXPECTED_FOLDS) - set(split_rows["heldout_fold"]))
        require(
            not missing_folds,
            f"split {split_seed} has empty held-out folds: {missing_folds}",
        )

    require(
        pair_map["is_cliff"].str.casefold().eq("true").all(),
        "pair-map must be restricted to primary cliffs with is_cliff=true",
    )

    joined = membership.merge(
        pair_map[list(PAIR_MAP_COLUMNS)],
        on="pair_id",
        how="left",
        suffixes=("", "_map"),
        validate="many_to_one",
    )
    for column in ("comparison_pair_key", "record_id_i", "record_id_j"):
        require(
            joined[column].eq(joined[f"{column}_map"]).all(),
            f"membership.{column} disagrees with pair-map.{column} for at least one pair_id",
        )
        joined = joined.drop(columns=[f"{column}_map"])

    rec_i = records.rename(
        columns={
            "record_id": "record_id_i",
            "identity_hash": "identity_hash_i",
            "assay_id_or_text": "assay_id_or_text_i",
        }
    )
    rec_j = records.rename(
        columns={
            "record_id": "record_id_j",
            "identity_hash": "identity_hash_j",
            "assay_id_or_text": "assay_id_or_text_j",
        }
    )
    joined = joined.merge(rec_i, on="record_id_i", how="left", validate="many_to_one")
    joined = joined.merge(rec_j, on="record_id_j", how="left", validate="many_to_one")
    endpoint_columns = (
        "identity_hash_i",
        "identity_hash_j",
        "assay_id_or_text_i",
        "assay_id_or_text_j",
    )
    require(
        joined[list(endpoint_columns)].notna().all().all()
        and joined[list(endpoint_columns)]
        .astype(str)
        .apply(lambda column: column.str.len().gt(0))
        .all()
        .all(),
        "one or more primary-pair endpoints could not be joined to records.record_id",
    )
    for column in ("identity_hash_i", "identity_hash_j"):
        invalid_count = int(
            (~joined[column].map(lambda value: bool(IDENTITY_PATTERN.fullmatch(str(value))))).sum()
        )
        require(
            invalid_count == 0,
            f"{column} contains {invalid_count} value(s) outside the 64-character "
            "uppercase hexadecimal identity-hash grammar",
        )
    require(
        joined["identity_hash_i"].ne(joined["identity_hash_j"]).all(),
        "a primary pair has identical endpoint identity hashes",
    )

    endpoint_assay_tokens = pd.concat(
        (joined["assay_id_or_text_i"], joined["assay_id_or_text_j"]),
        ignore_index=True,
    ).drop_duplicates()
    valid_assay_token = endpoint_assay_tokens.astype(str).map(
        lambda token: token == ASSAY_MISSING_SENTINEL
        or ASSAY_KNOWN_PATTERN.fullmatch(token) is not None
    )
    invalid_assay_count = int((~valid_assay_token).sum())
    require(
        invalid_assay_count == 0,
        "records.assay_id_or_text contains "
        f"{invalid_assay_count} distinct token(s) outside the expected grammar "
        f"({ASSAY_MISSING_SENTINEL} or RAWASSAY:<64 uppercase hex characters>)",
    )
    require(
        ASSAY_MISSING_SENTINEL in set(endpoint_assay_tokens),
        f"the expected missing-assay sentinel {ASSAY_MISSING_SENTINEL} was not observed",
    )

    unordered_a: list[str] = []
    unordered_b: list[str] = []
    directed_keys: list[str] = []
    unordered_keys: list[str] = []
    for identity_i, identity_j in zip(
        joined["identity_hash_i"], joined["identity_hash_j"]
    ):
        first, second = sorted((str(identity_i), str(identity_j)))
        unordered_a.append(first)
        unordered_b.append(second)
        directed_keys.append(
            sha256_text(f"canonical_isomeric_directed_pair_v1|{identity_i}|{identity_j}")
        )
        unordered_keys.append(
            sha256_text(f"canonical_isomeric_unordered_pair_v1|{first}|{second}")
        )
    joined["unordered_identity_hash_a"] = unordered_a
    joined["unordered_identity_hash_b"] = unordered_b
    joined["directed_identity_pair_key"] = directed_keys
    joined["unordered_identity_pair_key"] = unordered_keys

    joined["assay_id_i_known"] = joined["assay_id_or_text_i"].map(
        lambda token: ASSAY_KNOWN_PATTERN.fullmatch(str(token)) is not None
    )
    joined["assay_id_j_known"] = joined["assay_id_or_text_j"].map(
        lambda token: ASSAY_KNOWN_PATTERN.fullmatch(str(token)) is not None
    )

    joined = joined.reset_index(drop=True)
    train_pair_rows = np.zeros(len(joined), dtype=np.int64)
    directed_occurrences = np.zeros(len(joined), dtype=np.int64)
    unordered_occurrences = np.zeros(len(joined), dtype=np.int64)
    identity_i_occurrences = np.zeros(len(joined), dtype=np.int64)
    identity_j_occurrences = np.zeros(len(joined), dtype=np.int64)

    for split_seed in EXPECTED_SPLIT_SEEDS:
        split_rows = joined[joined["split_seed"] == split_seed]
        for fold in EXPECTED_FOLDS:
            test = split_rows[split_rows["heldout_fold"] == fold]
            train = split_rows[split_rows["heldout_fold"] != fold]
            require(
                len(test) > 0,
                f"split {split_seed}, fold {fold} has no held-out primary pairs",
            )
            require(
                len(train) == PRIMARY_PAIR_COUNT - len(test),
                f"other-fold training count is inconsistent for split {split_seed}, fold {fold}",
            )

            directed_counter = Counter(
                zip(train["identity_hash_i"], train["identity_hash_j"])
            )
            unordered_counter = Counter(
                zip(
                    train["unordered_identity_hash_a"],
                    train["unordered_identity_hash_b"],
                )
            )
            endpoint_counter = Counter(train["identity_hash_i"])
            endpoint_counter.update(train["identity_hash_j"])

            for row in test.itertuples():
                index = row.Index
                train_pair_rows[index] = len(train)
                directed_occurrences[index] = directed_counter[
                    (row.identity_hash_i, row.identity_hash_j)
                ]
                unordered_occurrences[index] = unordered_counter[
                    (row.unordered_identity_hash_a, row.unordered_identity_hash_b)
                ]
                identity_i_occurrences[index] = endpoint_counter[row.identity_hash_i]
                identity_j_occurrences[index] = endpoint_counter[row.identity_hash_j]

    joined["train_primary_pair_rows"] = train_pair_rows
    joined["train_directed_identity_pair_occurrences"] = directed_occurrences
    joined["train_unordered_identity_pair_occurrences"] = unordered_occurrences
    joined["train_identity_i_endpoint_occurrences"] = identity_i_occurrences
    joined["train_identity_j_endpoint_occurrences"] = identity_j_occurrences
    joined["in_all_oof"] = True
    joined["in_no_unordered_identity_pair_in_training"] = unordered_occurrences == 0
    joined["in_both_endpoint_identities_absent_from_training"] = (
        (identity_i_occurrences == 0) & (identity_j_occurrences == 0)
    )
    joined["in_both_assay_ids_known"] = (
        joined["assay_id_i_known"] & joined["assay_id_j_known"]
    )

    require(
        np.all((directed_occurrences > 0) <= (unordered_occurrences > 0)),
        "directed identity recurrence was not captured by the unordered recurrence audit",
    )

    joined = joined.sort_values(
        ["split_seed", "heldout_fold", "pair_id"], kind="mergesort"
    ).reset_index(drop=True)
    pair_level = joined.drop_duplicates("pair_id")
    endpoint_record_ids = set(pair_level["record_id_i"]) | set(pair_level["record_id_j"])
    missing_record_ids = set(
        pd.concat(
            (
                pair_level.loc[
                    pair_level["assay_id_or_text_i"] == ASSAY_MISSING_SENTINEL,
                    "record_id_i",
                ],
                pair_level.loc[
                    pair_level["assay_id_or_text_j"] == ASSAY_MISSING_SENTINEL,
                    "record_id_j",
                ],
            ),
            ignore_index=True,
        )
    )
    assay_summary = {
        "missing_sentinel": ASSAY_MISSING_SENTINEL,
        "known_token_pattern": ASSAY_KNOWN_PATTERN.pattern,
        "primary_endpoint_records": len(endpoint_record_ids),
        "primary_endpoint_records_with_missing_token": len(missing_record_ids),
        "primary_pairs_both_endpoints_known": int(
            pair_level["in_both_assay_ids_known"].sum()
        ),
        "primary_pairs_with_either_endpoint_missing": int(
            (~pair_level["in_both_assay_ids_known"]).sum()
        ),
    }
    return joined, assay_summary


def validate_and_join_predictions(
    predictions: pd.DataFrame, membership: pd.DataFrame
) -> pd.DataFrame:
    require_unique(
        predictions,
        ("pair_id", "split_seed", "fold", "model", "model_seed"),
        "predictions",
    )
    require(
        (predictions["true_delta"] != 0.0).all(),
        "predictions.true_delta contains zero; primary-cliff truth must be nonzero",
    )
    observed_models = set(predictions["model"])
    require(
        observed_models == set(MODEL_STREAMS),
        "predictions.model inventory does not match the fixed saved-stream design; "
        f"expected {sorted(MODEL_STREAMS)}, observed {sorted(observed_models)}",
    )
    observed_streams = {
        model: tuple(
            sorted(
                int(value)
                for value in predictions.loc[
                    predictions["model"] == model, "model_seed"
                ].unique()
            )
        )
        for model in MODEL_STREAMS
    }
    require(
        observed_streams == MODEL_STREAMS,
        "predictions model_seed inventory does not match the fixed saved streams; "
        f"expected {MODEL_STREAMS}, observed {observed_streams}",
    )

    expected_prediction_rows = len(membership) * TOTAL_SAVED_STREAMS
    require(
        len(predictions) == expected_prediction_rows,
        f"predictions must contain {TOTAL_SAVED_STREAMS} saved streams for each "
        f"of {len(membership)} OOF rows ({expected_prediction_rows} rows total); "
        f"observed {len(predictions)}",
    )
    per_oof_counts = predictions.groupby(
        ["pair_id", "split_seed", "fold"], sort=False
    ).size()
    require(
        per_oof_counts.eq(TOTAL_SAVED_STREAMS).all(),
        f"each OOF row must contain exactly {TOTAL_SAVED_STREAMS} saved streams",
    )
    require(
        predictions.groupby(
            ["pair_id", "split_seed", "fold"], sort=False
        )["true_delta"]
        .nunique()
        .eq(1)
        .all(),
        "predictions.true_delta differs across saved streams for an OOF row",
    )

    membership_keys = membership[
        ["pair_id", "split_seed", "heldout_fold"]
    ].rename(columns={"heldout_fold": "fold"})
    prediction_keys = predictions[
        ["pair_id", "split_seed", "fold"]
    ].drop_duplicates()
    key_check = membership_keys.merge(
        prediction_keys,
        on=["pair_id", "split_seed", "fold"],
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    require(
        key_check["_merge"].eq("both").all(),
        "predictions and membership do not have identical (pair_id, split_seed, fold) OOF keys",
    )

    join_columns = [
        "pair_id",
        "split_seed",
        "heldout_fold",
        "comparison_pair_key",
        "component_id",
        "record_id_i",
        "record_id_j",
        "true_delta",
        "normalized_poi_id",
        "unordered_identity_pair_key",
        "in_all_oof",
        "in_no_unordered_identity_pair_in_training",
        "in_both_endpoint_identities_absent_from_training",
        "in_both_assay_ids_known",
    ]
    joined = predictions.merge(
        membership[join_columns].rename(
            columns={"heldout_fold": "fold", "true_delta": "pair_map_true_delta"}
        ),
        on=["pair_id", "split_seed", "fold"],
        how="left",
        suffixes=("", "_membership"),
        validate="many_to_one",
    )
    require(
        joined["in_all_oof"].notna().all(),
        "at least one prediction row could not be joined to membership",
    )
    for column in ("comparison_pair_key", "component_id", "record_id_i", "record_id_j"):
        require(
            joined[column].eq(joined[f"{column}_membership"]).all(),
            f"predictions.{column} disagrees with membership.{column} for at least one OOF key",
        )
        joined = joined.drop(columns=[f"{column}_membership"])

    require(
        np.allclose(
            joined["true_delta"].to_numpy(dtype=np.float64),
            -joined["pair_map_true_delta"].to_numpy(dtype=np.float64),
            rtol=0.0,
            atol=1e-12,
        ),
        "truth sign mismatch: predictions.true_delta must be j-i and equal "
        "-pair-map.true_delta (which is i-j)",
    )
    return joined


def calculate_metrics(
    predictions: pd.DataFrame, membership: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for subset, flag in SUBSETS:
        base_count = int(membership[flag].sum())
        for model, model_seeds in MODEL_STREAMS.items():
            selected = predictions[
                predictions[flag] & predictions["model"].eq(model)
            ]
            expected_rows = base_count * len(model_seeds)
            require(
                len(selected) == expected_rows,
                f"prediction support mismatch for subset={subset}, model={model}: "
                f"expected {expected_rows}, observed {len(selected)}",
            )
            if selected.empty:
                direction_accuracy = math.nan
                mae = math.nan
                rmse = math.nan
                status = "EMPTY_SUBSET"
            else:
                truth = selected["true_delta"].to_numpy(dtype=np.float64)
                prediction = selected["prediction"].to_numpy(dtype=np.float64)
                error = prediction - truth
                direction_accuracy = float(
                    np.mean(
                        (prediction != 0.0)
                        & (np.sign(prediction) == np.sign(truth))
                    )
                )
                mae = float(np.mean(np.abs(error)))
                rmse = float(np.sqrt(np.mean(np.square(error))))
                status = "ESTIMABLE"
            rows.append(
                {
                    "subset": subset,
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "saved_streams": len(model_seeds),
                    "model_seeds": ";".join(str(seed) for seed in model_seeds),
                    "base_pair_by_split_rows": base_count,
                    "prediction_rows": int(len(selected)),
                    "direction_accuracy": direction_accuracy,
                    "mae_pdc50": mae,
                    "rmse_pdc50": rmse,
                    "status": status,
                }
            )
    return pd.DataFrame(rows)


def support_row(
    subset: str,
    scope: str,
    frame: pd.DataFrame,
    *,
    split_seed: int | str = "ALL",
    heldout_fold: int | str = "ALL",
    model: str = "ALL",
    saved_streams: int = TOTAL_SAVED_STREAMS,
    model_seeds: str = "",
) -> dict[str, Any]:
    endpoint_identities = set(frame["identity_hash_i"]) | set(frame["identity_hash_j"])
    return {
        "subset": subset,
        "scope": scope,
        "split_seed": split_seed,
        "heldout_fold": heldout_fold,
        "model": model,
        "saved_streams": saved_streams,
        "model_seeds": model_seeds,
        "base_pair_by_split_rows": int(len(frame)),
        "prediction_rows": int(len(frame) * saved_streams),
        "unique_pair_ids": int(frame["pair_id"].nunique()),
        "unique_unordered_identity_pairs": int(
            frame["unordered_identity_pair_key"].nunique()
        ),
        "unique_endpoint_identities": int(len(endpoint_identities)),
        "unique_comparison_pair_keys": int(frame["comparison_pair_key"].nunique()),
        "unique_pois": int(frame["normalized_poi_id"].nunique()),
        "min_train_primary_pair_rows": (
            int(frame["train_primary_pair_rows"].min()) if len(frame) else math.nan
        ),
        "max_train_primary_pair_rows": (
            int(frame["train_primary_pair_rows"].max()) if len(frame) else math.nan
        ),
        "test_rows_with_directed_pair_in_training": int(
            (frame["train_directed_identity_pair_occurrences"] > 0).sum()
        ),
        "test_rows_with_unordered_pair_in_training": int(
            (frame["train_unordered_identity_pair_occurrences"] > 0).sum()
        ),
        "test_rows_with_any_endpoint_identity_in_training": int(
            (
                (frame["train_identity_i_endpoint_occurrences"] > 0)
                | (frame["train_identity_j_endpoint_occurrences"] > 0)
            ).sum()
        ),
        "test_rows_with_both_assay_ids_known": int(
            frame["in_both_assay_ids_known"].sum()
        ),
    }


def calculate_support(membership: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for subset, flag in SUBSETS:
        selected = membership[membership[flag]].copy()
        rows.append(support_row(subset, "overall", selected))
        for split_seed in EXPECTED_SPLIT_SEEDS:
            for fold in EXPECTED_FOLDS:
                cell = selected[
                    selected["split_seed"].eq(split_seed)
                    & selected["heldout_fold"].eq(fold)
                ]
                rows.append(
                    support_row(
                        subset,
                        "split_fold",
                        cell,
                        split_seed=split_seed,
                        heldout_fold=fold,
                    )
                )
        for model, model_seeds in MODEL_STREAMS.items():
            rows.append(
                support_row(
                    subset,
                    "model",
                    selected,
                    model=model,
                    saved_streams=len(model_seeds),
                    model_seeds=";".join(str(seed) for seed in model_seeds),
                )
            )
    return pd.DataFrame(rows)


MEMBERSHIP_OUTPUT_COLUMNS = (
    "split_seed",
    "heldout_fold",
    "pair_id",
    "comparison_pair_key",
    "component_id",
    "record_id_i",
    "record_id_j",
    "normalized_poi_id",
    "identity_hash_i",
    "identity_hash_j",
    "unordered_identity_hash_a",
    "unordered_identity_hash_b",
    "directed_identity_pair_key",
    "unordered_identity_pair_key",
    "assay_id_or_text_i",
    "assay_id_or_text_j",
    "assay_id_i_known",
    "assay_id_j_known",
    "train_primary_pair_rows",
    "train_directed_identity_pair_occurrences",
    "train_unordered_identity_pair_occurrences",
    "train_identity_i_endpoint_occurrences",
    "train_identity_j_endpoint_occurrences",
    "in_all_oof",
    "in_no_unordered_identity_pair_in_training",
    "in_both_endpoint_identities_absent_from_training",
    "in_both_assay_ids_known",
)


def write_outputs(
    output_dir: Path,
    membership: pd.DataFrame,
    metrics: pd.DataFrame,
    support: pd.DataFrame,
    assay_summary: dict[str, Any],
    input_rows: dict[str, int],
    *,
    write_membership: bool,
) -> list[str]:
    require(
        not output_dir.exists() or output_dir.is_dir(),
        f"output path exists and is not a directory: {output_dir}",
    )
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(
            output_dir / "metrics.csv",
            index=False,
            lineterminator="\n",
            float_format="%.17g",
        )
        support.to_csv(
            output_dir / "support.csv",
            index=False,
            lineterminator="\n",
            float_format="%.17g",
        )

        outputs = ["metrics.csv", "support.csv", "RUN_METADATA.json"]
        if write_membership:
            membership[list(MEMBERSHIP_OUTPUT_COLUMNS)].to_csv(
                output_dir / "membership.csv",
                index=False,
                lineterminator="\n",
            )
            outputs.append("membership.csv")

        metadata = {
            "schema": "protac_cliff_benchmark.fixed_oof_identity_sensitivity.v1",
            "analysis_class": "post_hoc_descriptive_fixed_oof_no_refit",
            "input_rows": input_rows,
            "study_design": {
                "primary_pairs": PRIMARY_PAIR_COUNT,
                "split_seeds": list(EXPECTED_SPLIT_SEEDS),
                "folds": list(EXPECTED_FOLDS),
                "training_rule": (
                    "For each split/fold, the training population is the same split's "
                    "874 primary-cliff pairs outside the held-out fold."
                ),
                "saved_streams": {
                    model: list(seeds) for model, seeds in MODEL_STREAMS.items()
                },
            },
            "subset_base_pair_by_split_rows": {
                subset: int(membership[flag].sum()) for subset, flag in SUBSETS
            },
            "assay_token_summary": assay_summary,
            "outputs_written": outputs,
            "row_level_membership_written": write_membership,
            "limitations": [
                "No model fitting, prediction, resampling, or bootstrap is performed.",
                "Subset comparisons are descriptive and do not identify causal effects.",
                "RAWASSAY tokens are normalized raw-text hashes, not ontology-verified assay IDs.",
            ],
        }
        (output_dir / "RUN_METADATA.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    except OSError as exc:
        raise AnalysisInputError(f"could not write output directory {output_dir}: {exc}") from exc
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        required=True,
        type=Path,
        help="CSV containing the fixed saved OOF prediction streams",
    )
    parser.add_argument(
        "--membership",
        required=True,
        type=Path,
        help="CSV assigning each primary pair to one held-out fold per split seed",
    )
    parser.add_argument(
        "--pair-map",
        required=True,
        type=Path,
        help="CSV describing the 874 oriented primary-cliff pairs",
    )
    parser.add_argument(
        "--records",
        required=True,
        type=Path,
        help="CSV mapping endpoint record IDs to identity hashes and assay tokens",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for aggregate output files",
    )
    parser.add_argument(
        "--write-membership",
        action="store_true",
        help=(
            "also write row-level membership.csv; this can contain restricted IDs "
            "and should remain local"
        ),
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    predictions, raw_membership, pair_map, records = load_inputs(
        args.predictions,
        args.membership,
        args.pair_map,
        args.records,
    )
    input_rows = {
        "predictions": int(len(predictions)),
        "membership": int(len(raw_membership)),
        "pair_map": int(len(pair_map)),
        "records": int(len(records)),
    }
    membership, assay_summary = annotate_membership(
        raw_membership, pair_map, records
    )
    joined_predictions = validate_and_join_predictions(predictions, membership)
    metrics = calculate_metrics(joined_predictions, membership)
    support = calculate_support(membership)
    outputs = write_outputs(
        args.output_dir,
        membership,
        metrics,
        support,
        assay_summary,
        input_rows,
        write_membership=args.write_membership,
    )
    return {
        "status": "PASS",
        "membership_rows": int(len(membership)),
        "prediction_rows": int(len(joined_predictions)),
        "metric_rows": int(len(metrics)),
        "support_rows": int(len(support)),
        "outputs_written": outputs,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = run(args)
    except AnalysisInputError as exc:
        parser.exit(2, f"identity_sensitivity: error: {exc}\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
