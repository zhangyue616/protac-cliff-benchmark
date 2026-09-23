"""Shared model and feature functions for the frozen fitting protocols.

The functions in this module are the portable scientific subset of the
historical primary executor.  They contain no activation gates, host paths, or
file-system policy.  Callers remain responsible for validating their input
universe, split membership, and output location.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


FINGERPRINT_BITS = 2048
FINGERPRINT_HEX_WIDTH = 512


class PrimaryProtocolError(RuntimeError):
    """Raised when a portable primary-protocol invariant is violated."""


def decode_fp_hex(fp_hex: str) -> np.ndarray:
    """Decode the frozen LSB0 integer-bit representation to float32[2048]."""

    if (
        not isinstance(fp_hex, str)
        or len(fp_hex) != FINGERPRINT_HEX_WIDTH
        or fp_hex != fp_hex.upper()
        or any(char not in "0123456789ABCDEF" for char in fp_hex)
    ):
        raise PrimaryProtocolError("invalid 2,048-bit fingerprint hex value")
    little_endian_bytes = bytes.fromhex(fp_hex)[::-1]
    vector = np.unpackbits(
        np.frombuffer(little_endian_bytes, dtype=np.uint8), bitorder="little"
    ).astype(np.float32, copy=False)
    if vector.shape != (FINGERPRINT_BITS,) or vector.dtype != np.dtype(np.float32):
        raise PrimaryProtocolError("decoded fingerprint has the wrong shape or dtype")
    return vector


def load_fingerprints(
    csv_path: str | Path,
    required_record_ids: Iterable[str] | None = None,
) -> tuple[dict[str, int], np.ndarray]:
    """Read the construction-stage fingerprint table in lexical record order."""

    frame = pd.read_csv(csv_path, keep_default_na=False)
    required_columns = {"record_id", "fp_hex"}
    if not required_columns.issubset(frame.columns):
        raise PrimaryProtocolError(
            f"fingerprint table is missing {sorted(required_columns - set(frame.columns))}"
        )
    if frame["record_id"].astype(str).str.strip().eq("").any():
        raise PrimaryProtocolError("fingerprint table contains a blank record_id")
    if frame["record_id"].duplicated().any():
        raise PrimaryProtocolError("fingerprint table contains duplicate record_id values")
    frame = frame.assign(record_id=frame["record_id"].astype(str)).sort_values(
        "record_id", kind="mergesort"
    )
    record_ids = frame["record_id"].tolist()
    required = set(map(str, required_record_ids or ()))
    missing = required - set(record_ids)
    if missing:
        raise PrimaryProtocolError(
            f"fingerprint table is missing {len(missing)} required record identifiers"
        )
    matrix = np.stack([decode_fp_hex(value) for value in frame["fp_hex"]])
    if matrix.shape != (len(frame), FINGERPRINT_BITS) or not np.isfinite(matrix).all():
        raise PrimaryProtocolError("fingerprint matrix failed shape or finite-value checks")
    return {record_id: index for index, record_id in enumerate(record_ids)}, matrix


def pair_features(
    frame: pd.DataFrame,
    index: dict[str, int],
    fingerprints: np.ndarray,
) -> np.ndarray:
    """Return endpoint-j minus endpoint-i fingerprint vectors."""

    missing_columns = {"record_id_i", "record_id_j"} - set(frame.columns)
    if missing_columns:
        raise PrimaryProtocolError(f"pair table is missing {sorted(missing_columns)}")
    try:
        left = [index[str(value)] for value in frame["record_id_i"]]
        right = [index[str(value)] for value in frame["record_id_j"]]
    except KeyError as error:
        raise PrimaryProtocolError(f"pair references an unknown fingerprint: {error}") from error
    matrix = (fingerprints[right] - fingerprints[left]).astype(np.float32)
    if matrix.shape != (len(frame), FINGERPRINT_BITS) or not np.isfinite(matrix).all():
        raise PrimaryProtocolError("pair-feature matrix failed shape or finite-value checks")
    return matrix


def similarity(vector: np.ndarray, train: np.ndarray) -> np.ndarray:
    """Frozen signed-vector Tanimoto calculation used by the NN control."""

    dot = train @ vector
    denominator = (
        np.einsum("ij,ij->i", train, train) + float(vector @ vector) - dot
    )
    return np.divide(
        dot,
        denominator,
        out=np.full_like(dot, -np.inf),
        where=denominator > 0,
    )


def support_label(value: float) -> str:
    if value < 0.5:
        return "lt_0p5"
    if value < 0.6:
        return "0p5_to_lt_0p6"
    if value < 0.7:
        return "0p6_to_lt_0p7"
    if value < 0.8:
        return "0p7_to_lt_0p8"
    return "ge_0p8"


def make_model(model: str, seed: int, *, n_jobs: int = 4):
    """Construct one model with the frozen primary hyperparameters."""

    if model == "ridge":
        from sklearn.linear_model import Ridge

        return Ridge(alpha=1.0)
    if model == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=360,
            max_depth=12,
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=n_jobs,
            random_state=seed,
        )
    if model in {"xgboost", "permuted_xgboost"}:
        from xgboost import XGBRegressor

        return XGBRegressor(
            objective="reg:squarederror",
            n_estimators=360,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.75,
            reg_lambda=1.5,
            n_jobs=n_jobs,
            tree_method="hist",
            random_state=seed,
        )
    raise PrimaryProtocolError(f"unsupported model identity: {model}")


def finite_vector(values, expected_rows: int, label: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (expected_rows,) or not np.isfinite(vector).all():
        raise PrimaryProtocolError(f"{label} is not a finite vector of length {expected_rows}")
    return vector


def nonfit_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray, list[str]]:
    """Compute the four nonfit controls and nearest-neighbor support values."""

    nearest: list[float] = []
    maximum: list[float] = []
    for vector in x_test:
        similarities = similarity(vector, x_train)
        if not np.isfinite(similarities).any():
            raise PrimaryProtocolError("a test pair has no finite NN similarity")
        best_value = float(np.max(similarities))
        best = np.flatnonzero(similarities == best_value)
        selected = min(best, key=lambda idx: str(train.iloc[int(idx)]["pair_id"]))
        maximum.append(best_value)
        nearest.append(float(y_train[int(selected)]))
    maximum_array = finite_vector(maximum, len(test), "maximum training similarity")
    if train["source_dataset"].isna().any() or test["source_dataset"].isna().any():
        raise PrimaryProtocolError("source_dataset is missing in a fitting cell")
    source_mean = train.groupby("source_dataset")["target_b_minus_a"].mean()
    overall_mean = float(np.mean(y_train))
    values = {
        "zero_delta": np.zeros(len(test), dtype=float),
        "train_mean": np.full(len(test), overall_mean, dtype=float),
        "source_prior": test["source_dataset"]
        .map(source_mean)
        .fillna(overall_mean)
        .to_numpy(float),
        "nearest_neighbor": np.asarray(nearest, dtype=float),
    }
    values = {
        name: finite_vector(vector, len(test), name) for name, vector in values.items()
    }
    return values, maximum_array, [support_label(value) for value in maximum_array]
