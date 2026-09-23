"""Fixed protocol and input validation for the supplemental refit command.

This module contains no model fitting.  It validates the author-generated E1
assignments, the reconstructed study tables, and the original fit-cell plan;
it also creates the two pre-specified component-multiplicity plans used by the
prediction-only E1/E2 scoring commands.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


PAIR_COUNT = 874
GROUP_COUNT = 838
ENDPOINT_COUNT = 420
SPLIT_SEEDS = (20260624, 20260625, 20260626, 20260724, 20260801)
FOLDS = (0, 1, 2, 3, 4)
MODEL_SEEDS = (20260624, 20260724, 20260801)
PERMUTED_MODEL_SEED = 20260803

E1_COMPONENT_COUNT = 55
E1_SUPPORTED_COMPONENTS = 29
E2_COMPONENT_COUNT = 77
E2_SUPPORTED_COMPONENTS = 40
E1_FIT_COUNT = 200
E2_FIT_COUNT = 175
TOTAL_FIT_COUNT = E1_FIT_COUNT + E2_FIT_COUNT
E1_PREDICTION_ROWS = 52_440
E2_PREDICTION_ROWS = 30_590

BOOTSTRAP_SPECS: Mapping[str, Mapping[str, int]] = {
    "E1": {"replicates": 2_000, "components": 55, "seed": 2026091601, "family_size": 6},
    "E2": {"replicates": 50_000, "components": 77, "seed": 2026091602, "family_size": 18},
}

OUTPUT_COLUMNS = (
    "pair_id",
    "comparison_pair_key",
    "record_id_i",
    "record_id_j",
    "normalized_poi_id",
    "component_id",
    "true_delta",
    "prediction",
    "model",
    "model_seed",
    "split_seed",
    "fold",
)


class SupplementalError(RuntimeError):
    """Raised when inputs or existing output violate the fixed protocol."""


@dataclass(frozen=True)
class ProtocolInputs:
    records: pd.DataFrame
    pairs: pd.DataFrame
    primary_splits: pd.DataFrame
    e1_assignments: pd.DataFrame
    e1_components: tuple[str, ...]
    e2_components: tuple[str, ...]
    fit_plan: pd.DataFrame
    permutation_seeds: pd.DataFrame
    paths: Mapping[str, Path]
    legacy_pair_target_orientation: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _require_file(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise SupplementalError(f"{label} does not exist: {resolved}")
    return resolved


def _read_csv(path: Path, label: str, columns: Iterable[str]) -> pd.DataFrame:
    path = _require_file(path, label)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise SupplementalError(f"{label} is missing columns: {', '.join(missing)}")
    return frame


def _require_nonblank(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    for column in columns:
        values = frame[column].astype(str)
        if values.str.strip().eq("").any():
            raise SupplementalError(f"{label}.{column} contains a blank value")
        frame[column] = values


def _coerce_integer(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    for column in columns:
        try:
            numeric = pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise SupplementalError(f"{label}.{column} must contain integers") from exc
        values = numeric.to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise SupplementalError(f"{label}.{column} must contain finite integers")
        frame[column] = numeric.astype(np.int64)


def _coerce_float(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    for column in columns:
        try:
            values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise SupplementalError(f"{label}.{column} must contain numeric values") from exc
        if not np.isfinite(values).all():
            raise SupplementalError(f"{label}.{column} contains a non-finite value")
        frame[column] = values


def _validate_membership(
    frame: pd.DataFrame,
    *,
    label: str,
    fold_column: str,
    component_column: str,
    pair_ids: set[str],
    pair_groups: Mapping[str, str],
    expected_supported_components: int,
) -> pd.DataFrame:
    _require_nonblank(frame, ("pair_id", component_column), label)
    _coerce_integer(frame, ("split_seed", fold_column), label)
    if len(frame) != PAIR_COUNT * len(SPLIT_SEEDS):
        raise SupplementalError(f"{label} must contain {PAIR_COUNT * len(SPLIT_SEEDS)} rows")
    if frame.duplicated(["split_seed", "pair_id"]).any():
        raise SupplementalError(f"{label} contains duplicate split_seed/pair_id rows")
    if set(frame["split_seed"]) != set(SPLIT_SEEDS):
        raise SupplementalError(f"{label} split seeds differ from the fixed five-seed protocol")
    if set(frame[fold_column]) != set(FOLDS):
        raise SupplementalError(f"{label} folds differ from 0..4")
    for seed, group in frame.groupby("split_seed", sort=False):
        if set(group["pair_id"]) != pair_ids:
            raise SupplementalError(f"{label} pair universe differs at split seed {seed}")
    if frame.groupby("pair_id")[component_column].nunique().max() != 1:
        raise SupplementalError(f"{label} changes a pair's component across split seeds")
    if frame.groupby(["split_seed", component_column])[fold_column].nunique().max() != 1:
        raise SupplementalError(f"{label} splits a component across folds")
    group_keys = frame["pair_id"].map(pair_groups)
    if group_keys.isna().any():
        raise SupplementalError(f"{label} contains a pair without a comparison group")
    check = frame.assign(_comparison_pair_key=group_keys)
    if check.groupby(["split_seed", "_comparison_pair_key"])[fold_column].nunique().max() != 1:
        raise SupplementalError(f"{label} splits a comparison group across folds")
    if frame[component_column].nunique() != expected_supported_components:
        raise SupplementalError(
            f"{label} must contain {expected_supported_components} supported components"
        )
    return frame


def _validate_fit_plan(fit_plan: pd.DataFrame, permutation_seeds: pd.DataFrame) -> None:
    _require_nonblank(fit_plan, ("cell_id", "model"), "fit plan")
    _coerce_integer(
        fit_plan,
        ("fit_ordinal", "split_seed", "fold", "model_seed", "fit_budget_units"),
        "fit plan",
    )
    if len(fit_plan) != E1_FIT_COUNT:
        raise SupplementalError(f"fit plan must contain {E1_FIT_COUNT} rows")
    if set(fit_plan["fit_ordinal"]) != set(range(1, E1_FIT_COUNT + 1)):
        raise SupplementalError("fit plan ordinals must be exactly 1..200")
    if fit_plan["cell_id"].duplicated().any():
        raise SupplementalError("fit plan cell_id values must be unique")
    if set(fit_plan["split_seed"]) != set(SPLIT_SEEDS) or set(fit_plan["fold"]) != set(FOLDS):
        raise SupplementalError("fit plan seed/fold inventory differs from the fixed protocol")
    expected_counts = {
        "ridge": 25,
        "random_forest": 75,
        "xgboost": 75,
        "permuted_xgboost": 25,
    }
    observed_counts = fit_plan.groupby("model").size().to_dict()
    if observed_counts != expected_counts:
        raise SupplementalError(f"fit plan model counts differ: {observed_counts}")
    expected_model_seeds = {
        "ridge": {-1},
        "random_forest": set(MODEL_SEEDS),
        "xgboost": set(MODEL_SEEDS),
        "permuted_xgboost": {PERMUTED_MODEL_SEED},
    }
    for model, seeds in expected_model_seeds.items():
        observed = set(fit_plan.loc[fit_plan["model"] == model, "model_seed"])
        if observed != seeds:
            raise SupplementalError(f"fit plan model seeds differ for {model}: {sorted(observed)}")
    expected_ids = fit_plan.apply(
        lambda row: f"{row['model']}|{row['split_seed']}|{row['fold']}|{row['model_seed']}",
        axis=1,
    )
    if not expected_ids.equals(fit_plan["cell_id"]):
        raise SupplementalError("fit plan cell_id values do not match their model/seed/fold fields")
    if not fit_plan["fit_budget_units"].eq(1).all():
        raise SupplementalError("fit plan fit_budget_units must all equal one")

    _require_nonblank(permutation_seeds, ("permutation_seed",), "permutation seed table")
    _coerce_integer(
        permutation_seeds,
        ("split_seed", "fold", "permutation_seed"),
        "permutation seed table",
    )
    if len(permutation_seeds) != 25 or permutation_seeds.duplicated(["split_seed", "fold"]).any():
        raise SupplementalError("permutation seed table must contain one row per fixed seed/fold")
    expected_cells = {(seed, fold) for seed in SPLIT_SEEDS for fold in FOLDS}
    observed_cells = set(
        permutation_seeds[["split_seed", "fold"]].itertuples(index=False, name=None)
    )
    if observed_cells != expected_cells:
        raise SupplementalError("permutation seed table seed/fold inventory differs")
    permutation_lookup = permutation_seeds.set_index(["split_seed", "fold"])["permutation_seed"]
    permuted = fit_plan[fit_plan["model"] == "permuted_xgboost"]
    for row in permuted.itertuples(index=False):
        raw = str(row.permutation_seed).strip()
        if not raw:
            raise SupplementalError(f"fit plan lacks permutation seed for {row.cell_id}")
        if int(raw) != int(permutation_lookup.loc[(row.split_seed, row.fold)]):
            raise SupplementalError(f"fit plan permutation seed mismatch for {row.cell_id}")


def load_protocol_inputs(
    construction_dir: Path,
    *,
    e1_assignments_path: Path,
    e1_components_path: Path,
    fit_plan_path: Path,
    permutation_seeds_path: Path,
) -> ProtocolInputs:
    """Load and validate every non-model input used by the 375-fit protocol."""

    construction_dir = construction_dir.resolve()
    if not construction_dir.is_dir():
        raise SupplementalError(f"construction directory does not exist: {construction_dir}")
    paths = {
        "records": construction_dir / "records_aggregated_v0.1.csv",
        "pairs": construction_dir / "primary_pair_map.csv",
        "binary_fingerprints": construction_dir / "record_morgan_fp_r2_2048.csv",
        "primary_splits": construction_dir / "primary_split_membership.csv",
        "primary_components": construction_dir / "merged_endpoint_graph_components.csv",
        "e1_assignments": e1_assignments_path,
        "e1_components": e1_components_path,
        "fit_plan": fit_plan_path,
        "permutation_seeds": permutation_seeds_path,
    }
    paths = {label: _require_file(path, label) for label, path in paths.items()}

    records = _read_csv(
        paths["records"],
        "records",
        (
            "record_id",
            "canonical_isomeric_smiles_full",
            "identity_hash",
            "source_dataset",
            "pdc50_median",
        ),
    )
    _require_nonblank(
        records,
        ("record_id", "canonical_isomeric_smiles_full", "identity_hash", "source_dataset"),
        "records",
    )
    if records["record_id"].duplicated().any():
        raise SupplementalError("records.record_id must be unique")
    # The accepted supplemental runner used pandas' round-trip CSV parser for
    # activities.  Converting an already-tokenized decimal string with
    # ``to_numeric`` can choose a neighboring float for a small subset of
    # values; those last-bit differences can change exact impurity ties in a
    # seeded random forest.  Re-read this one numeric column with the original
    # parser contract while keeping identifiers explicitly textual.
    round_trip_activity = pd.read_csv(
        paths["records"],
        usecols=["record_id", "pdc50_median"],
        dtype={"record_id": str},
        keep_default_na=False,
        float_precision="round_trip",
    )
    if (
        len(round_trip_activity) != len(records)
        or round_trip_activity["record_id"].duplicated().any()
        or set(round_trip_activity["record_id"]) != set(records["record_id"])
    ):
        raise SupplementalError("round-trip activity projection differs from the records table")
    activity_values = round_trip_activity.set_index("record_id")["pdc50_median"]
    records["pdc50_median"] = records["record_id"].map(activity_values).to_numpy(dtype=np.float64)
    if not np.isfinite(records["pdc50_median"].to_numpy(dtype=np.float64)).all():
        raise SupplementalError("records.pdc50_median contains a non-finite value")
    records = records.set_index("record_id", drop=False)

    pairs = _read_csv(
        paths["pairs"],
        "pairs",
        (
            "pair_id",
            "comparison_pair_key",
            "record_id_i",
            "record_id_j",
            "normalized_poi_id",
        ),
    )
    _require_nonblank(
        pairs,
        (
            "pair_id",
            "comparison_pair_key",
            "record_id_i",
            "record_id_j",
            "normalized_poi_id",
        ),
        "pairs",
    )
    pairs = pairs.sort_values("pair_id", kind="stable").reset_index(drop=True)
    if len(pairs) != PAIR_COUNT or pairs["pair_id"].nunique() != PAIR_COUNT:
        raise SupplementalError(f"pairs must contain {PAIR_COUNT} unique pair identifiers")
    if pairs["comparison_pair_key"].nunique() != GROUP_COUNT:
        raise SupplementalError(f"pairs must contain {GROUP_COUNT} comparison groups")
    endpoint_ids = set(pairs["record_id_i"]) | set(pairs["record_id_j"])
    if len(endpoint_ids) != ENDPOINT_COUNT:
        raise SupplementalError(f"pair population must use {ENDPOINT_COUNT} endpoint records")
    missing_records = sorted(endpoint_ids - set(records.index))
    if missing_records:
        raise SupplementalError(f"pairs reference records absent from records table: {missing_records[:3]}")
    activity = records["pdc50_median"]
    target = (
        pairs["record_id_j"].map(activity).to_numpy(dtype=np.float64)
        - pairs["record_id_i"].map(activity).to_numpy(dtype=np.float64)
    )
    if not np.isfinite(target).all() or not (np.abs(target) >= 1.0 - 1e-12).all():
        raise SupplementalError("reconstructed j-minus-i targets fail the measured-cliff gate")
    pairs["target"] = target
    source_i = pairs["record_id_i"].map(records["source_dataset"])
    source_j = pairs["record_id_j"].map(records["source_dataset"])
    if source_i.isna().any() or source_j.isna().any() or not source_i.equals(source_j):
        raise SupplementalError("pair endpoints do not share one reconstructed source_dataset")
    pairs["source_dataset"] = source_i.to_numpy()

    legacy_orientation = "absent"
    if "true_delta" in pairs.columns:
        legacy = pd.to_numeric(pairs["true_delta"], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(legacy).all():
            raise SupplementalError("pairs.true_delta contains non-finite values")
        if np.allclose(legacy, target, atol=1e-12, rtol=0):
            legacy_orientation = "j_minus_i"
        elif np.allclose(legacy, -target, atol=1e-12, rtol=0):
            legacy_orientation = "i_minus_j_ignored"
        else:
            raise SupplementalError("pairs.true_delta matches neither endpoint orientation")

    pair_ids = set(pairs["pair_id"])
    pair_groups = pairs.set_index("pair_id")["comparison_pair_key"].to_dict()
    pair_records_i = pairs.set_index("pair_id")["record_id_i"]
    pair_records_j = pairs.set_index("pair_id")["record_id_j"]

    primary_splits = _read_csv(
        paths["primary_splits"],
        "primary splits",
        (
            "split_seed",
            "heldout_fold",
            "pair_id",
            "comparison_pair_key",
            "component_id",
            "record_id_i",
            "record_id_j",
        ),
    )
    primary_splits = _validate_membership(
        primary_splits,
        label="primary splits",
        fold_column="heldout_fold",
        component_column="component_id",
        pair_ids=pair_ids,
        pair_groups=pair_groups,
        expected_supported_components=E2_SUPPORTED_COMPONENTS,
    )
    if not primary_splits["record_id_i"].equals(primary_splits["pair_id"].map(pair_records_i)):
        raise SupplementalError("primary splits record_id_i differs from the pair map")
    if not primary_splits["record_id_j"].equals(primary_splits["pair_id"].map(pair_records_j)):
        raise SupplementalError("primary splits record_id_j differs from the pair map")
    if not primary_splits["comparison_pair_key"].equals(
        primary_splits["pair_id"].map(pair_groups)
    ):
        raise SupplementalError("primary splits comparison groups differ from the pair map")

    primary_components = _read_csv(
        paths["primary_components"],
        "primary components",
        ("record_id", "component_id"),
    )
    _require_nonblank(primary_components, ("record_id", "component_id"), "primary components")
    if primary_components["record_id"].duplicated().any():
        raise SupplementalError("primary components contains duplicate record identifiers")
    e2_components = tuple(sorted(primary_components["component_id"].unique()))
    if len(e2_components) != E2_COMPONENT_COUNT:
        raise SupplementalError(f"primary component container must contain {E2_COMPONENT_COUNT} units")
    primary_component_map = primary_components.set_index("record_id")["component_id"]
    if not endpoint_ids.issubset(set(primary_component_map.index)):
        raise SupplementalError("primary component map does not cover every evaluated endpoint")
    pair_component_i = pairs["record_id_i"].map(primary_component_map)
    pair_component_j = pairs["record_id_j"].map(primary_component_map)
    if not pair_component_i.equals(pair_component_j):
        raise SupplementalError("one or more evaluated pairs crosses primary components")
    split_components = primary_splits.drop_duplicates("pair_id").set_index("pair_id")["component_id"]
    if not pairs["pair_id"].map(split_components).equals(pair_component_i.reset_index(drop=True)):
        raise SupplementalError("primary split components differ from the reconstructed component map")

    e1_assignments = _read_csv(
        paths["e1_assignments"],
        "E1 assignments",
        ("rule", "split_seed", "pair_id", "global_component", "candidate_fold"),
    )
    if set(e1_assignments["rule"]) != {"record_balanced_original_rule"}:
        raise SupplementalError("E1 assignments must contain only record_balanced_original_rule")
    e1_assignments = _validate_membership(
        e1_assignments,
        label="E1 assignments",
        fold_column="candidate_fold",
        component_column="global_component",
        pair_ids=pair_ids,
        pair_groups=pair_groups,
        expected_supported_components=E1_SUPPORTED_COMPONENTS,
    )
    e1_component_frame = _read_csv(
        paths["e1_components"], "E1 components", ("component_id",)
    )
    _require_nonblank(e1_component_frame, ("component_id",), "E1 components")
    e1_components = tuple(e1_component_frame["component_id"])
    if len(e1_components) != E1_COMPONENT_COUNT or len(set(e1_components)) != E1_COMPONENT_COUNT:
        raise SupplementalError(f"E1 component container must contain {E1_COMPONENT_COUNT} unique units")
    if e1_components != tuple(sorted(e1_components)):
        raise SupplementalError("E1 component inventory must be in lexical order")
    if not set(e1_assignments["global_component"]).issubset(set(e1_components)):
        raise SupplementalError("E1 assignment contains a component outside the 55-unit container")

    identity = records["identity_hash"]
    for seed in SPLIT_SEEDS:
        assignment = e1_assignments[e1_assignments["split_seed"] == seed].set_index("pair_id")
        folds = pairs["pair_id"].map(assignment["candidate_fold"]).to_numpy(dtype=np.int64)
        for fold in FOLDS:
            test = pairs.loc[folds == fold]
            train = pairs.loc[folds != fold]
            test_ids = set(test["record_id_i"].map(identity)) | set(test["record_id_j"].map(identity))
            train_ids = set(train["record_id_i"].map(identity)) | set(train["record_id_j"].map(identity))
            if test_ids & train_ids:
                raise SupplementalError(f"E1 identity overlap at split seed {seed}, fold {fold}")

    fit_plan = _read_csv(
        paths["fit_plan"],
        "fit plan",
        (
            "fit_ordinal",
            "cell_id",
            "model",
            "split_seed",
            "fold",
            "model_seed",
            "permutation_seed",
            "fit_budget_units",
        ),
    )
    permutation_seeds = _read_csv(
        paths["permutation_seeds"],
        "permutation seed table",
        ("split_seed", "fold", "permutation_seed"),
    )
    _validate_fit_plan(fit_plan, permutation_seeds)
    fit_plan = fit_plan.sort_values("fit_ordinal", kind="stable").reset_index(drop=True)
    permutation_seeds = permutation_seeds.sort_values(
        ["split_seed", "fold"], kind="stable"
    ).reset_index(drop=True)

    return ProtocolInputs(
        records=records,
        pairs=pairs,
        primary_splits=primary_splits,
        e1_assignments=e1_assignments,
        e1_components=e1_components,
        e2_components=e2_components,
        fit_plan=fit_plan,
        permutation_seeds=permutation_seeds,
        paths=paths,
        legacy_pair_target_orientation=legacy_orientation,
    )


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise SupplementalError(f"stale temporary file requires inspection: {temporary}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise SupplementalError(f"stale temporary file requires inspection: {temporary}")
    frame.to_csv(temporary, index=False, lineterminator="\n", float_format="%.17g")
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: Any) -> None:
    temporary = path.with_name(path.name + ".tmp.npz")
    if temporary.exists():
        raise SupplementalError(f"stale temporary file requires inspection: {temporary}")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def build_binding(
    inputs: ProtocolInputs,
    *,
    training_core_path: Path,
    implementation_paths: Mapping[str, Path],
    versions: Mapping[str, str],
    n_jobs: int,
) -> dict[str, Any]:
    if n_jobs != 4:
        raise SupplementalError("the fixed protocol requires n_jobs=4")
    input_files = {
        label: {
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for label, path in sorted(inputs.paths.items())
    }
    core_path = _require_file(training_core_path, "primary training core")
    implementation = {
        label: {
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for label, value in sorted(implementation_paths.items())
        for path in (_require_file(value, label),)
    }
    specification: dict[str, Any] = {
        "protocol": "supplemental_A_public_refit_v1",
        "target": "pDC50(j)-pDC50(i), reconstructed from records",
        "legacy_pair_true_delta": inputs.legacy_pair_target_orientation,
        "split_seeds": list(SPLIT_SEEDS),
        "folds": list(FOLDS),
        "fit_counts": {"E1": E1_FIT_COUNT, "E2": E2_FIT_COUNT, "total": TOTAL_FIT_COUNT},
        "prediction_rows": {"E1": E1_PREDICTION_ROWS, "E2": E2_PREDICTION_ROWS},
        "bootstrap": {key: dict(value) for key, value in BOOTSTRAP_SPECS.items()},
        "features": {
            "E1": "binary Morgan radius=2 fpSize=2048 endpoint-j minus endpoint-i",
            "E2": (
                "folded Morgan count radius=2 fpSize=2048 includeChirality=false "
                "countSimulation=false endpoint-j minus endpoint-i"
            ),
        },
        "n_jobs": n_jobs,
        "inputs": input_files,
        "training_core": {
            "name": core_path.name,
            "bytes": core_path.stat().st_size,
            "sha256": sha256_file(core_path),
        },
        "supplemental_implementation": implementation,
        "versions": dict(sorted(versions.items())),
        "python": {"version": sys.version.split()[0], "platform": platform.platform()},
    }
    encoded = json.dumps(specification, sort_keys=True, separators=(",", ":")).encode("utf-8")
    specification["protocol_sha256"] = hashlib.sha256(encoded).hexdigest().upper()
    return specification


def prepare_output_dir(output_dir: Path, binding: Mapping[str, Any]) -> tuple[Path, bool]:
    output = output_dir.resolve()
    if output.exists() and not output.is_dir():
        raise SupplementalError(f"output path is not a directory: {output}")
    created = not output.exists()
    output.mkdir(parents=True, exist_ok=True)
    binding_path = output / "RUN_BINDING.json"
    expected = dict(binding)
    if binding_path.exists():
        try:
            observed = json.loads(binding_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SupplementalError(f"cannot read existing run binding: {binding_path}") from exc
        observed.pop("created_unix", None)
        if observed != expected:
            raise SupplementalError("existing output directory is bound to different inputs or software")
        return output, False
    if any(output.iterdir()):
        raise SupplementalError("nonempty output directory lacks RUN_BINDING.json")
    stored = dict(expected)
    stored["created_unix"] = time.time()
    atomic_json(binding_path, stored)
    return output, created


def _ensure_plan(path: Path, components: tuple[str, ...], spec: Mapping[str, int]) -> None:
    n_components = int(spec["components"])
    replicates = int(spec["replicates"])
    seed = int(spec["seed"])
    if len(components) != n_components:
        raise SupplementalError(f"{path.name} component count differs from the fixed protocol")
    generator = np.random.Generator(np.random.PCG64(seed))
    expected = generator.multinomial(
        n_components,
        np.full(n_components, 1.0 / n_components),
        size=replicates,
    ).astype(np.int16)
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as archive:
                weights = archive["weights"]
                observed_components = tuple(str(value) for value in archive["components"].tolist())
                observed_seed = int(np.asarray(archive["seed"]).reshape(-1)[0])
        except (OSError, KeyError, ValueError) as exc:
            raise SupplementalError(f"cannot read existing draw plan: {path}") from exc
        if observed_components != components or observed_seed != seed or not np.array_equal(weights, expected):
            raise SupplementalError(f"existing draw plan differs from the fixed generated plan: {path}")
        return
    _atomic_npz(
        path,
        weights=expected,
        components=np.asarray(components),
        seed=np.asarray(seed, dtype=np.int64),
    )


def ensure_bootstrap_plans(output: Path, inputs: ProtocolInputs) -> dict[str, Any]:
    plans = {
        "E1": (output / "E1_bootstrap_plan.npz", inputs.e1_components),
        "E2": (output / "E2_bootstrap_plan.npz", inputs.e2_components),
    }
    rows: dict[str, Any] = {}
    for experiment, (path, components) in plans.items():
        spec = BOOTSTRAP_SPECS[experiment]
        _ensure_plan(path, components, spec)
        rows[experiment] = {
            "file": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "shape": [int(spec["replicates"]), int(spec["components"])],
            "tickets_per_replicate": int(spec["components"]),
            "seed": int(spec["seed"]),
            "family_size": int(spec["family_size"]),
        }
    receipt = {
        "status": "FIXED_SUPPLEMENTAL_DRAW_PLANS_READY",
        "plans": rows,
        "draws_generated_from_fixed_protocol": True,
        "prediction_metrics_computed": False,
    }
    receipt_path = output / "PLAN_RECEIPT.json"
    if receipt_path.exists():
        observed = json.loads(receipt_path.read_text(encoding="utf-8"))
        if observed != receipt:
            raise SupplementalError("existing PLAN_RECEIPT.json differs from the fixed plans")
    else:
        atomic_json(receipt_path, receipt)
    return receipt
