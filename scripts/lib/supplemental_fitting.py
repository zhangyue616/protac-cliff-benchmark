"""Portable fitting and checkpoint logic for supplemental experiments E1/E2."""

from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from lib.primary import training_core
from lib import supplemental_protocol
from lib.supplemental_protocol import (
    BOOTSTRAP_SPECS,
    E1_FIT_COUNT,
    E1_PREDICTION_ROWS,
    E2_FIT_COUNT,
    E2_PREDICTION_ROWS,
    ENDPOINT_COUNT,
    FOLDS,
    MODEL_SEEDS,
    OUTPUT_COLUMNS,
    PAIR_COUNT,
    PERMUTED_MODEL_SEED,
    SPLIT_SEEDS,
    TOTAL_FIT_COUNT,
    ProtocolInputs,
    SupplementalError,
    atomic_csv,
    atomic_json,
    build_binding,
    ensure_bootstrap_plans,
    prepare_output_dir,
    sha256_file,
)


@dataclass(frozen=True)
class FeatureBundle:
    binary_pair_features: np.ndarray
    count_pair_features: np.ndarray
    diagnostic: pd.DataFrame
    count_cache_path: Path | None


def software_versions() -> dict[str, str]:
    import rdkit
    import sklearn
    import threadpoolctl
    import xgboost

    return {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "rdkit": rdkit.__version__,
        "scikit-learn": sklearn.__version__,
        "threadpoolctl": threadpoolctl.__version__,
        "xgboost": xgboost.__version__,
    }


def _atomic_npz(path: Path, **arrays: Any) -> None:
    temporary = path.with_name(path.name + ".tmp.npz")
    if temporary.exists():
        raise SupplementalError(f"stale temporary file requires inspection: {temporary}")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _generate_count_fingerprints(
    inputs: ProtocolInputs,
    endpoint_ids: tuple[str, ...],
) -> np.ndarray:
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import rdFingerprintGenerator
    except ImportError as exc:
        raise SupplementalError("RDKit is required for the E2 folded-count refit") from exc

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=2048,
        includeChirality=False,
        countSimulation=False,
    )
    counts = np.zeros((len(endpoint_ids), 2048), dtype=np.float32)
    cache: dict[str, np.ndarray] = {}
    for index, record_id in enumerate(endpoint_ids):
        smiles = str(inputs.records.loc[record_id, "canonical_isomeric_smiles_full"])
        vector = cache.get(smiles)
        if vector is None:
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                raise SupplementalError(f"RDKit could not parse evaluated record {record_id}")
            integer_vector = np.zeros(2048, dtype=np.int32)
            DataStructs.ConvertToNumpyArray(
                generator.GetCountFingerprint(molecule), integer_vector
            )
            if (integer_vector < 0).any():
                raise SupplementalError(f"negative folded count for evaluated record {record_id}")
            vector = integer_vector.astype(np.float32)
            cache[smiles] = vector
        counts[index] = vector
    if counts.shape != (ENDPOINT_COUNT, 2048) or not np.isfinite(counts).all():
        raise SupplementalError("folded-count matrix failed shape or finite-value checks")
    return counts


def _load_or_create_count_cache(
    inputs: ProtocolInputs,
    endpoint_ids: tuple[str, ...],
    output: Path | None,
) -> tuple[np.ndarray, Path | None]:
    path = output / "folded_count_420.npz" if output is not None else None
    if path is not None and path.exists():
        try:
            with np.load(path, allow_pickle=False) as archive:
                observed_ids = tuple(str(value) for value in archive["record_ids"].tolist())
                counts = archive["counts"]
        except (OSError, KeyError, ValueError) as exc:
            raise SupplementalError(f"cannot read folded-count cache: {path}") from exc
        if observed_ids != endpoint_ids:
            raise SupplementalError("folded-count cache record ordering differs from current inputs")
        counts = np.asarray(counts, dtype=np.float32)
    else:
        counts = _generate_count_fingerprints(inputs, endpoint_ids)
        if path is not None:
            _atomic_npz(
                path,
                record_ids=np.asarray(endpoint_ids),
                counts=counts,
            )
    if counts.shape != (ENDPOINT_COUNT, 2048) or not np.isfinite(counts).all():
        raise SupplementalError("folded-count cache failed shape or finite-value checks")
    return counts, path


def build_features(inputs: ProtocolInputs, output: Path | None) -> FeatureBundle:
    endpoint_ids = tuple(
        sorted(set(inputs.pairs["record_id_i"]) | set(inputs.pairs["record_id_j"]))
    )
    try:
        binary_index, binary = training_core.load_fingerprints(
            inputs.paths["binary_fingerprints"], required_record_ids=endpoint_ids
        )
        binary_pair_features = training_core.pair_features(
            inputs.pairs, binary_index, binary
        )
    except training_core.PrimaryProtocolError as exc:
        raise SupplementalError(str(exc)) from exc

    counts, count_cache_path = _load_or_create_count_cache(inputs, endpoint_ids, output)
    binary_used = np.stack([binary[binary_index[record_id]] for record_id in endpoint_ids])
    if not np.array_equal((counts > 0).astype(np.float32), binary_used):
        raise SupplementalError("folded-count presence differs from the frozen binary fingerprints")
    count_index = {record_id: index for index, record_id in enumerate(endpoint_ids)}
    try:
        count_pair_features = training_core.pair_features(inputs.pairs, count_index, counts)
    except training_core.PrimaryProtocolError as exc:
        raise SupplementalError(str(exc)) from exc

    equal = np.all(binary_pair_features == 0, axis=1)
    if int(equal.sum()) != 16:
        raise SupplementalError("fixed pair population must contain 16 binary-equal pairs")
    separated = np.any(count_pair_features[equal] != 0, axis=1)
    if int(separated.sum()) != 14:
        raise SupplementalError("fixed folded-count representation must separate 14 of 16 pairs")
    diagnostic = inputs.pairs.loc[
        equal,
        ["pair_id", "comparison_pair_key", "record_id_i", "record_id_j"],
    ].copy()
    diagnostic["folded_count_separated"] = separated
    diagnostic["folded_count_l1"] = np.abs(count_pair_features[equal]).sum(axis=1)

    if output is not None:
        diagnostic_path = output / "folded_count_16_diagnostic.csv"
        if diagnostic_path.exists():
            observed = pd.read_csv(diagnostic_path, keep_default_na=False)
            if len(observed) != 16 or int(observed["folded_count_separated"].astype(bool).sum()) != 14:
                raise SupplementalError("existing folded-count diagnostic differs from the fixed gate")
        else:
            atomic_csv(diagnostic_path, diagnostic)
        if count_cache_path is None:
            raise SupplementalError("internal error: count cache path was not created")
        gate = {
            "status": "PASS",
            "endpoints": ENDPOINT_COUNT,
            "dimensions": 2048,
            "radius": 2,
            "include_chirality": False,
            "count_simulation": False,
            "all_count_presence_equals_frozen_binary": True,
            "binary_equal_pairs": 16,
            "folded_count_separated": 14,
            "target": "pDC50(j)-pDC50(i), reconstructed from records",
            "count_cache": {
                "file": count_cache_path.name,
                "bytes": count_cache_path.stat().st_size,
                "sha256": sha256_file(count_cache_path),
            },
        }
        gate_path = output / "FEATURE_GATE.json"
        if gate_path.exists():
            observed_gate = json.loads(gate_path.read_text(encoding="utf-8"))
            if observed_gate != gate:
                raise SupplementalError("existing FEATURE_GATE.json differs from current features")
        else:
            atomic_json(gate_path, gate)
    return FeatureBundle(
        binary_pair_features=binary_pair_features,
        count_pair_features=count_pair_features,
        diagnostic=diagnostic,
        count_cache_path=count_cache_path,
    )


def _fold_arrays(inputs: ProtocolInputs, experiment: str) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    folds_by_seed: dict[int, np.ndarray] = {}
    components_by_seed: dict[int, np.ndarray] = {}
    source = inputs.e1_assignments if experiment == "E1" else inputs.primary_splits
    fold_column = "candidate_fold" if experiment == "E1" else "heldout_fold"
    component_column = "global_component" if experiment == "E1" else "component_id"
    for seed in SPLIT_SEEDS:
        membership = source[source["split_seed"] == seed].set_index("pair_id")
        folds_by_seed[seed] = (
            inputs.pairs["pair_id"].map(membership[fold_column]).to_numpy(dtype=np.int64)
        )
        components = inputs.pairs["pair_id"].map(membership[component_column])
        if components.isna().any():
            raise SupplementalError(f"{experiment} membership is incomplete at split seed {seed}")
        components_by_seed[seed] = components.to_numpy(dtype=str)
    return folds_by_seed, components_by_seed


def _prediction_frame(
    test: pd.DataFrame,
    values: np.ndarray,
    components: np.ndarray,
    *,
    model: str,
    model_seed: int,
    split_seed: int,
    fold: int,
) -> pd.DataFrame:
    predictions = np.asarray(values, dtype=np.float64)
    if predictions.shape != (len(test),) or not np.isfinite(predictions).all():
        raise SupplementalError(
            f"non-finite or wrong-length prediction for {model}|{split_seed}|{fold}|{model_seed}"
        )
    frame = test[
        ["pair_id", "comparison_pair_key", "record_id_i", "record_id_j", "normalized_poi_id"]
    ].copy()
    frame["component_id"] = components
    frame["true_delta"] = test["target"].to_numpy(dtype=np.float64)
    frame["prediction"] = predictions
    frame["model"] = model
    frame["model_seed"] = model_seed
    frame["split_seed"] = split_seed
    frame["fold"] = fold
    return frame[list(OUTPUT_COLUMNS)]


def _validate_saved_cell(
    path: Path,
    test: pd.DataFrame,
    components: np.ndarray,
    *,
    expected_models: Mapping[str, int],
    split_seed: int,
    fold: int,
) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, keep_default_na=False, float_precision="round_trip")
    except (OSError, ValueError) as exc:
        raise SupplementalError(f"cannot read saved prediction cell: {path}") from exc
    missing = [column for column in OUTPUT_COLUMNS if column not in frame.columns]
    if missing:
        raise SupplementalError(f"saved prediction cell {path.name} lacks columns: {missing}")
    frame = frame[list(OUTPUT_COLUMNS)].copy()
    expected_rows = len(test) * len(expected_models)
    if len(frame) != expected_rows:
        raise SupplementalError(f"saved prediction cell {path.name} has {len(frame)} rows, expected {expected_rows}")
    observed_streams = set(
        frame[["model", "model_seed"]].itertuples(index=False, name=None)
    )
    if observed_streams != set(expected_models.items()):
        raise SupplementalError(f"saved prediction cell {path.name} has wrong streams")
    if set(pd.to_numeric(frame["split_seed"], errors="coerce")) != {split_seed}:
        raise SupplementalError(f"saved prediction cell {path.name} has wrong split seed")
    if set(pd.to_numeric(frame["fold"], errors="coerce")) != {fold}:
        raise SupplementalError(f"saved prediction cell {path.name} has wrong fold")
    expected_pairs = set(test["pair_id"])
    for _, group in frame.groupby(["model", "model_seed"], sort=False):
        if set(group["pair_id"]) != expected_pairs or group["pair_id"].duplicated().any():
            raise SupplementalError(f"saved prediction cell {path.name} has wrong pair inventory")
    expected_truth = test.set_index("pair_id")["target"]
    expected_component = pd.Series(components, index=test["pair_id"])
    truth = pd.to_numeric(frame["true_delta"], errors="coerce").to_numpy(dtype=np.float64)
    prediction = pd.to_numeric(frame["prediction"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise SupplementalError(f"saved prediction cell {path.name} contains non-finite values")
    expected_truth_rows = frame["pair_id"].map(expected_truth).to_numpy(dtype=np.float64)
    if not np.allclose(truth, expected_truth_rows, atol=1e-12, rtol=0):
        raise SupplementalError(f"saved prediction cell {path.name} has wrong target orientation")
    if not frame["component_id"].astype(str).equals(
        frame["pair_id"].map(expected_component).astype(str)
    ):
        raise SupplementalError(f"saved prediction cell {path.name} has wrong components")
    return frame


def _control_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
) -> Mapping[str, np.ndarray]:
    from threadpoolctl import threadpool_limits

    nearest: list[float] = []
    with threadpool_limits(limits=4):
        for vector in x_test:
            similarities = training_core.similarity(vector, x_train)
            if not np.isfinite(similarities).any():
                raise SupplementalError("a test pair has no finite nearest-neighbor similarity")
            choices = np.flatnonzero(similarities == similarities.max())
            selected = min(choices, key=lambda index: str(train.iloc[int(index)]["pair_id"]))
            nearest.append(float(y_train[int(selected)]))
    overall_mean = float(np.mean(y_train))
    source_means = train.groupby("source_dataset")["target"].mean()
    return {
        "zero_delta": np.zeros(len(test), dtype=np.float64),
        "train_mean": np.full(len(test), overall_mean, dtype=np.float64),
        "source_prior": test["source_dataset"]
        .map(source_means)
        .fillna(overall_mean)
        .to_numpy(dtype=np.float64),
        "nearest_neighbor": np.asarray(nearest, dtype=np.float64),
    }


def _load_ledger(output: Path, protocol_sha256: str) -> dict[str, dict[str, Any]]:
    path = output / "FIT_LEDGER.json"
    if not path.exists():
        return {}
    try:
        ledger = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SupplementalError(f"cannot read fit ledger: {path}") from exc
    if not isinstance(ledger, dict):
        raise SupplementalError("FIT_LEDGER.json must contain an object")
    for key, row in ledger.items():
        if not isinstance(row, dict) or row.get("protocol_sha256") != protocol_sha256:
            raise SupplementalError(f"fit ledger row has a different protocol binding: {key}")
    return ledger


def _pause_if_requested(output: Path, ledger: Mapping[str, Mapping[str, Any]], experiment: str) -> bool:
    if not (output / "PAUSE_REQUESTED").exists():
        return False
    completed = sum(row.get("status") == "SUCCESS" for row in ledger.values())
    atomic_json(
        output / "STATUS.json",
        {
            "status": "PAUSED_AT_CELL_BOUNDARY",
            "experiment": experiment,
            "finished_fits": completed,
            "planned_fits": TOTAL_FIT_COUNT,
        },
    )
    return True


def _fit_cell_file(output: Path, experiment: str, row: Any) -> Path:
    return output / "cells" / experiment / (
        f"{row.model}_{int(row.split_seed)}_{int(row.fold)}_{int(row.model_seed)}.csv"
    )


def _run_experiment(
    inputs: ProtocolInputs,
    features: FeatureBundle,
    output: Path,
    ledger: dict[str, dict[str, Any]],
    *,
    protocol_sha256: str,
    experiment: str,
    n_jobs: int,
) -> bool:
    from threadpoolctl import threadpool_limits

    feature_matrix = (
        features.binary_pair_features if experiment == "E1" else features.count_pair_features
    )
    folds_by_seed, components_by_seed = _fold_arrays(inputs, experiment)
    plan = inputs.fit_plan
    if experiment == "E2":
        plan = plan[plan["model"] != "permuted_xgboost"]
    expected_fits = E1_FIT_COUNT if experiment == "E1" else E2_FIT_COUNT
    if len(plan) != expected_fits:
        raise SupplementalError(f"{experiment} plan contains {len(plan)} fits, expected {expected_fits}")
    destination = output / "cells" / experiment
    destination.mkdir(parents=True, exist_ok=True)
    prepared_controls: set[tuple[int, int]] = set()
    permutation_lookup = inputs.permutation_seeds.set_index(["split_seed", "fold"])[
        "permutation_seed"
    ]

    for row in plan.itertuples(index=False):
        split_seed = int(row.split_seed)
        fold = int(row.fold)
        model_seed = int(row.model_seed)
        folds = folds_by_seed[split_seed]
        mask = folds == fold
        train = inputs.pairs.loc[~mask].copy()
        test = inputs.pairs.loc[mask].copy()
        if len(train) == 0 or len(test) == 0:
            raise SupplementalError(f"empty fitting cell for {experiment}|{row.cell_id}")
        x_train = feature_matrix[~mask]
        x_test = feature_matrix[mask]
        y_train = train["target"].to_numpy(dtype=np.float64)
        components = components_by_seed[split_seed][mask]

        if experiment == "E1" and (split_seed, fold) not in prepared_controls:
            if _pause_if_requested(output, ledger, experiment):
                return False
            control_path = destination / f"baselines_{split_seed}_{fold}.csv"
            expected_controls = {
                "zero_delta": -1,
                "train_mean": -1,
                "source_prior": -1,
                "nearest_neighbor": -1,
            }
            if control_path.exists():
                _validate_saved_cell(
                    control_path,
                    test,
                    components,
                    expected_models=expected_controls,
                    split_seed=split_seed,
                    fold=fold,
                )
            else:
                controls = _control_predictions(train, test, x_train, x_test, y_train)
                control_frame = pd.concat(
                    [
                        _prediction_frame(
                            test,
                            values,
                            components,
                            model=model,
                            model_seed=-1,
                            split_seed=split_seed,
                            fold=fold,
                        )
                        for model, values in controls.items()
                    ],
                    ignore_index=True,
                )
                atomic_csv(control_path, control_frame)
            prepared_controls.add((split_seed, fold))

        if _pause_if_requested(output, ledger, experiment):
            return False
        key = f"{experiment}|{row.cell_id}"
        cell_path = _fit_cell_file(output, experiment, row)
        existing = ledger.get(key)
        expected_stream = {str(row.model): model_seed}
        if existing is not None:
            if existing.get("status") != "SUCCESS" or not cell_path.exists():
                raise SupplementalError(
                    f"unresolved earlier fit attempt requires a fresh output directory: {key}"
                )
            _validate_saved_cell(
                cell_path,
                test,
                components,
                expected_models=expected_stream,
                split_seed=split_seed,
                fold=fold,
            )
            continue
        if cell_path.exists():
            raise SupplementalError(f"prediction file exists without a fit-ledger row: {cell_path}")

        ledger[key] = {
            "protocol_sha256": protocol_sha256,
            "experiment": experiment,
            "cell_id": str(row.cell_id),
            "model": str(row.model),
            "split_seed": split_seed,
            "fold": fold,
            "model_seed": model_seed,
            "train_rows": len(train),
            "test_rows": len(test),
            "status": "RUNNING",
            "started_unix": time.time(),
        }
        atomic_json(output / "FIT_LEDGER.json", ledger)
        try:
            fit_target = y_train
            if str(row.model) == "permuted_xgboost":
                permutation_seed = int(permutation_lookup.loc[(split_seed, fold)])
                fit_target = np.random.default_rng(permutation_seed).permutation(y_train)
            model = training_core.make_model(str(row.model), model_seed, n_jobs=n_jobs)
            with threadpool_limits(limits=n_jobs):
                model.fit(x_train, fit_target)
                values = model.predict(x_test)
            frame = _prediction_frame(
                test,
                values,
                components,
                model=str(row.model),
                model_seed=model_seed,
                split_seed=split_seed,
                fold=fold,
            )
            atomic_csv(cell_path, frame)
            ledger[key].update(
                status="SUCCESS",
                seconds=time.time() - float(ledger[key]["started_unix"]),
                output=cell_path.relative_to(output).as_posix(),
            )
        except Exception as exc:
            ledger[key].update(
                status="FAILED",
                error_type=type(exc).__name__,
                error=str(exc),
                traceback=traceback.format_exc(),
            )
            atomic_json(output / "FIT_LEDGER.json", ledger)
            atomic_json(
                output / "STATUS.json",
                {
                    "status": "FIT_FAILED_REQUIRES_DISPOSITION",
                    "failed_cell": key,
                    "finished_fits": sum(
                        item.get("status") == "SUCCESS" for item in ledger.values()
                    ),
                    "planned_fits": TOTAL_FIT_COUNT,
                },
            )
            raise SupplementalError(f"fit failed and was recorded: {key}: {exc}") from exc
        atomic_json(output / "FIT_LEDGER.json", ledger)
        completed = sum(item.get("status") == "SUCCESS" for item in ledger.values())
        atomic_json(
            output / "STATUS.json",
            {
                "status": "FITTING",
                "experiment": experiment,
                "last_completed_cell": key,
                "finished_fits": completed,
                "planned_fits": TOTAL_FIT_COUNT,
            },
        )
        print(f"{key}: SUCCESS ({completed}/{TOTAL_FIT_COUNT})", flush=True)
    return True


def _expected_cell_paths(inputs: ProtocolInputs, output: Path, experiment: str) -> list[Path]:
    plan = inputs.fit_plan
    if experiment == "E2":
        plan = plan[plan["model"] != "permuted_xgboost"]
    paths = [_fit_cell_file(output, experiment, row) for row in plan.itertuples(index=False)]
    if experiment == "E1":
        paths.extend(
            output / "cells" / "E1" / f"baselines_{seed}_{fold}.csv"
            for seed in SPLIT_SEEDS
            for fold in FOLDS
        )
    return sorted(paths, key=lambda path: path.name)


def _validate_assembled(frame: pd.DataFrame, experiment: str) -> None:
    expected_rows = E1_PREDICTION_ROWS if experiment == "E1" else E2_PREDICTION_ROWS
    if len(frame) != expected_rows:
        raise SupplementalError(
            f"{experiment} assembled prediction table has {len(frame)} rows, expected {expected_rows}"
        )
    if frame.duplicated(["model", "model_seed", "split_seed", "fold", "pair_id"]).any():
        raise SupplementalError(f"{experiment} assembled prediction table has duplicate OOF keys")
    expected_streams = {
        "ridge": {-1},
        "random_forest": set(MODEL_SEEDS),
        "xgboost": set(MODEL_SEEDS),
    }
    if experiment == "E1":
        expected_streams.update(
            {
                "zero_delta": {-1},
                "train_mean": {-1},
                "source_prior": {-1},
                "nearest_neighbor": {-1},
                "permuted_xgboost": {PERMUTED_MODEL_SEED},
            }
        )
    observed_streams = {
        model: set(group["model_seed"].astype(int))
        for model, group in frame.groupby("model", sort=False)
    }
    if observed_streams != expected_streams:
        raise SupplementalError(f"{experiment} assembled stream inventory differs")
    if not frame.groupby(["model", "model_seed", "split_seed"]).size().eq(PAIR_COUNT).all():
        raise SupplementalError(f"{experiment} does not have one OOF row per pair and split stream")
    values = frame[["true_delta", "prediction"]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=np.float64)).all():
        raise SupplementalError(f"{experiment} assembled output contains non-finite values")
    supported = 29 if experiment == "E1" else 40
    if frame["component_id"].nunique() != supported:
        raise SupplementalError(f"{experiment} supported-component count differs from {supported}")


def _same_prediction_table(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    keys = ["model", "model_seed", "split_seed", "fold", "pair_id"]
    left = left.sort_values(keys, kind="stable").reset_index(drop=True)
    right = right.sort_values(keys, kind="stable").reset_index(drop=True)
    if len(left) != len(right):
        return False
    exact_columns = [column for column in OUTPUT_COLUMNS if column not in {"true_delta", "prediction"}]
    if not left[exact_columns].astype(str).equals(right[exact_columns].astype(str)):
        return False
    for column in ("true_delta", "prediction"):
        a = pd.to_numeric(left[column], errors="coerce").to_numpy(dtype=np.float64)
        b = pd.to_numeric(right[column], errors="coerce").to_numpy(dtype=np.float64)
        if not np.array_equal(a, b):
            return False
    return True


def _assemble_predictions(inputs: ProtocolInputs, output: Path, experiment: str) -> dict[str, Any]:
    paths = _expected_cell_paths(inputs, output, experiment)
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise SupplementalError(f"{experiment} is missing prediction cells: {missing[:3]}")
    frame = pd.concat(
        [pd.read_csv(path, keep_default_na=False, float_precision="round_trip") for path in paths],
        ignore_index=True,
    )
    frame = frame[list(OUTPUT_COLUMNS)].sort_values(
        ["model", "model_seed", "split_seed", "fold", "pair_id"], kind="stable"
    ).reset_index(drop=True)
    _validate_assembled(frame, experiment)
    destination = output / f"{experiment}_predictions.csv"
    if destination.exists():
        observed = pd.read_csv(destination, keep_default_na=False, float_precision="round_trip")
        _validate_assembled(observed, experiment)
        if not _same_prediction_table(frame, observed):
            raise SupplementalError(f"existing {destination.name} differs from its bound cell files")
    else:
        atomic_csv(destination, frame)
    return {
        "file": destination.name,
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "rows": len(frame),
        "streams": int(frame[["model", "model_seed"]].drop_duplicates().shape[0]),
    }


def validate_only(inputs: ProtocolInputs) -> dict[str, Any]:
    features = build_features(inputs, None)
    return {
        "status": "PASS_SUPPLEMENTAL_REFIT_INPUT_VALIDATION",
        "fits_executed": 0,
        "pairs": len(inputs.pairs),
        "split_seeds": len(SPLIT_SEEDS),
        "planned_fits": {"E1": E1_FIT_COUNT, "E2": E2_FIT_COUNT, "total": TOTAL_FIT_COUNT},
        "components": {"E1": len(inputs.e1_components), "E2": len(inputs.e2_components)},
        "prediction_rows": {"E1": E1_PREDICTION_ROWS, "E2": E2_PREDICTION_ROWS},
        "binary_equal_pairs": len(features.diagnostic),
        "folded_count_separated": int(features.diagnostic["folded_count_separated"].sum()),
        "target": "pDC50(j)-pDC50(i), reconstructed from records",
        "legacy_pair_true_delta": inputs.legacy_pair_target_orientation,
    }


def run_refit(inputs: ProtocolInputs, output_dir: Path, *, n_jobs: int = 4) -> dict[str, Any]:
    versions = software_versions()
    core_path = Path(training_core.__file__).resolve()
    binding = build_binding(
        inputs,
        training_core_path=core_path,
        implementation_paths={
            "supplemental_protocol": Path(supplemental_protocol.__file__).resolve(),
            "supplemental_fitting": Path(__file__).resolve(),
        },
        versions=versions,
        n_jobs=n_jobs,
    )
    output, _ = prepare_output_dir(output_dir, binding)
    ensure_bootstrap_plans(output, inputs)
    features = build_features(inputs, output)
    protocol_sha256 = str(binding["protocol_sha256"])
    ledger = _load_ledger(output, protocol_sha256)
    if not _run_experiment(
        inputs,
        features,
        output,
        ledger,
        protocol_sha256=protocol_sha256,
        experiment="E1",
        n_jobs=n_jobs,
    ):
        return {"status": "PAUSED_AT_CELL_BOUNDARY"}
    if not _run_experiment(
        inputs,
        features,
        output,
        ledger,
        protocol_sha256=protocol_sha256,
        experiment="E2",
        n_jobs=n_jobs,
    ):
        return {"status": "PAUSED_AT_CELL_BOUNDARY"}

    if len(ledger) != TOTAL_FIT_COUNT or not all(
        row.get("status") == "SUCCESS" for row in ledger.values()
    ):
        raise SupplementalError("fit ledger is not exactly 375 successful fits")
    counts = {
        "E1": sum(row["experiment"] == "E1" for row in ledger.values()),
        "E2": sum(row["experiment"] == "E2" for row in ledger.values()),
    }
    if counts != {"E1": E1_FIT_COUNT, "E2": E2_FIT_COUNT}:
        raise SupplementalError(f"fit ledger experiment counts differ: {counts}")
    predictions = {
        experiment: _assemble_predictions(inputs, output, experiment)
        for experiment in ("E1", "E2")
    }
    receipt = {
        "status": "PASS_375_FITS_PREDICTIONS_AND_FIXED_PLANS_READY_FOR_RESCORING",
        "protocol_sha256": protocol_sha256,
        "fit_counts": {"E1": E1_FIT_COUNT, "E2": E2_FIT_COUNT, "total": TOTAL_FIT_COUNT},
        "predictions": predictions,
        "plans": {
            experiment: {
                "file": f"{experiment}_bootstrap_plan.npz",
                **dict(BOOTSTRAP_SPECS[experiment]),
            }
            for experiment in ("E1", "E2")
        },
        "target": "pDC50(j)-pDC50(i), reconstructed from records",
        "scoring_performed": False,
        "next_commands": {
            "E1": "scripts/run_supplemental_e1.py",
            "E2": "scripts/run_supplemental_e2.py (also requires matching binary predictions)",
        },
        "finished_unix": time.time(),
    }
    atomic_json(output / "FIT_RECEIPT.json", receipt)
    atomic_json(
        output / "STATUS.json",
        {
            "status": "375_FITS_COMPLETE_RESCORING_READY",
            "finished_fits": TOTAL_FIT_COUNT,
            "planned_fits": TOTAL_FIT_COUNT,
            "protocol_sha256": protocol_sha256,
        },
    )
    return receipt
