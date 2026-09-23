#!/usr/bin/env python3
"""Locally reproduce the fixed PROTAC-DB overlap membership and rescore predictions.

The user must supply the fixed ``protac.csv`` produced from a personally
obtained PROTAC-DB workbook.  This script performs no network access and does
not accept source terms.  Its row-level outputs are PROTAC-DB-derived local
audit material and must not be committed or redistributed without specific
permission from the source provider.

Only the PROTAC-DB structure-overlap union is rebuilt here.  In the frozen
study this union had the same 589 retained / 285 excluded pair membership as
the full nonself union because the external and Wurz anchors added no pair.
Their separate labels, DOI matches, and manual curation are not regenerated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


SCHEMA_VERSION = "local-protacdb-source-overlap-v1"
EXPECTED_PROTACDB_SHA256 = (
    "F4601179B471A3B0CCAF10021A2C6C5289AD12CA6B6716233A734A778D26CE28"
)
EXPECTED_RECORDS_SHA256 = (
    "DA017991BDD0A83E91F086925986E48C113DECEB1347E90EB64DFF48CD7DEEED"
)
EXPECTED_PAIRS_SHA256 = (
    "37F6F70903162E58C41E14DABFC463600F891F7D5830FF895AEE2BFB79C691E7"
)
EXPECTED_PRIMARY_MAP_SHA256 = (
    "4680A5D5BA5A665F22AD122F5F72EE17C09F6B3702590C32F90DA82536AD71D0"
)
EXPECTED_PROTACDB_ROWS = 15_502
EXPECTED_PRIMARY_PAIRS = 874
EXPECTED_PRIMARY_KEYS = 838
EXPECTED_PRIMARY_ENDPOINTS = 420
EXPECTED_CONTAMINATED_ENDPOINTS = 191
EXPECTED_EXCLUDED_PAIRS = 285
EXPECTED_RETAINED_PAIRS = 589

SPLIT_SEEDS = [20260624, 20260625, 20260626, 20260724, 20260801]
MODEL_IDENTITIES = [
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
]
METRICS = ["direction_accuracy", "delta_mae", "delta_rmse"]
POLICIES = ["primary", "protacdb_union_known_exclude_keep"]

MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


class ExecutionError(RuntimeError):
    """Raised when an input or fixed scientific invariant does not hold."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ExecutionError(message)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(rows[0]) if rows else []
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, lineterminator="\n", extrasaction="raise"
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


# The following normalization and anchor builder are carried from the original
# A2 implementation.  DOI parsing is intentionally outside this public local
# subset because construction records do not retain the raw DOI strings and all
# frozen PROTAC-DB DOI positives were already captured by the >=0.80 structure
# match.  The fixed 589/285 census is enforced below.
def canon(smiles: object) -> tuple[str, str, object | None]:
    if pd.isna(smiles) or not str(smiles).strip():
        return "", "", None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return "", "", None
    return (
        Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True),
        Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False),
        MORGAN.GetFingerprint(mol),
    )


def build_anchor(
    name: str, frame: pd.DataFrame, smiles_col: str
) -> dict[str, Any]:
    iso: set[str] = set()
    noniso: set[str] = set()
    fps: list[Any] = []
    parsed = failed = 0
    for smiles in frame[smiles_col]:
        ci, cn, fp = canon(smiles)
        if fp is None:
            failed += 1
            continue
        parsed += 1
        iso.add(ci)
        noniso.add(cn)
        fps.append(fp)
    return {
        "name": name,
        "row_count": len(frame),
        "parsed_rows": parsed,
        "parse_failed_rows": failed,
        "iso": iso,
        "noniso": noniso,
        "fps": fps,
    }


# Copied from the original primary evaluator.
def metric_values(np: Any, truth: Any, prediction: Any) -> dict[str, float]:
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if truth.shape != prediction.shape or truth.size == 0:
        raise ExecutionError("E_METRIC_SUPPORT")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all() or (truth == 0).any():
        raise ExecutionError("E_METRIC_FINITE_OR_ZERO_TRUTH")
    error = prediction - truth
    return {
        "direction_accuracy": float(np.mean((prediction != 0) & (np.sign(prediction) == np.sign(truth)))),
        "delta_mae": float(np.mean(np.abs(error))),
        "delta_rmse": float(np.sqrt(np.mean(error * error))),
    }


def require_columns(
    frame: pd.DataFrame, columns: Iterable[str], label: str
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    require(not missing, f"MISSING_COLUMNS:{label}:{','.join(missing)}")


def require_nonblank(
    frame: pd.DataFrame, columns: Iterable[str], label: str
) -> None:
    for column in columns:
        values = frame[column]
        require(
            not values.isna().any() and not values.astype(str).str.strip().eq("").any(),
            f"BLANK_SOURCE:{label}:{column}",
        )


def resolve_predictions(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.is_dir():
        resolved = resolved / "predictions.csv"
    require(resolved.is_file(), f"PREDICTIONS_NOT_FOUND:{resolved}")
    return resolved


def file_receipt(path: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha(path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--construction-dir",
        required=True,
        type=Path,
        help="Reconstructed directory containing records, pairs, and primary map.",
    )
    parser.add_argument(
        "--protacdb-csv",
        required=True,
        type=Path,
        help="Fixed local protac.csv prepared from the user-obtained workbook.",
    )
    parser.add_argument(
        "--predictions",
        required=True,
        type=Path,
        help="Frozen predictions.csv, or a directory containing that file.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Private local directory for membership, metrics, and summary.",
    )
    return parser.parse_args()


def validate_fixed_input(path: Path, expected: str, label: str) -> None:
    require(path.is_file(), f"INPUT_NOT_FOUND:{label}:{path}")
    observed = sha(path)
    require(observed == expected, f"INPUT_HASH_MISMATCH:{label}:{observed}")


def build_membership(
    records: pd.DataFrame,
    primary: pd.DataFrame,
    anchor: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, bool], dict[str, Any]]:
    record_by_id = records.set_index("record_id", verify_integrity=True)
    endpoint_ids = sorted(
        set(primary["record_id_i"]) | set(primary["record_id_j"])
    )
    require(
        len(endpoint_ids) == EXPECTED_PRIMARY_ENDPOINTS,
        f"EXPECTED_420_PRIMARY_ENDPOINTS_GOT:{len(endpoint_ids)}",
    )
    require(
        set(endpoint_ids).issubset(set(record_by_id.index)),
        "PRIMARY_ENDPOINT_MISSING_FROM_RECORDS",
    )

    endpoint_hit: dict[str, bool] = {}
    endpoint_exact_iso: dict[str, bool] = {}
    endpoint_exact_noniso: dict[str, bool] = {}
    endpoint_max_similarity: dict[str, float] = {}
    for record_id in endpoint_ids:
        smiles = record_by_id.at[record_id, "canonical_isomeric_smiles_full"]
        ci, cn, fp = canon(smiles)
        require(fp is not None, f"RECORD_SMILES_PARSE_FAILED:{record_id}")
        require(ci == smiles, f"RECORD_CANONICAL_SMILES_MISMATCH:{record_id}")
        require(
            hashlib.sha256(ci.encode("utf-8")).hexdigest().upper()
            == record_by_id.at[record_id, "identity_hash"],
            f"RECORD_IDENTITY_HASH_MISMATCH:{record_id}",
        )
        similarities = DataStructs.BulkTanimotoSimilarity(fp, anchor["fps"])
        maximum = max(similarities) if similarities else float("nan")
        require(math.isfinite(maximum), f"NO_ANCHOR_SIMILARITY:{record_id}")
        exact_iso = ci in anchor["iso"]
        exact_noniso = cn in anchor["noniso"]
        near = maximum >= 0.80 - 1e-12
        endpoint_exact_iso[record_id] = exact_iso
        endpoint_exact_noniso[record_id] = exact_noniso
        endpoint_max_similarity[record_id] = maximum
        endpoint_hit[record_id] = bool(exact_iso or exact_noniso or near)

    contaminated_endpoints = sum(endpoint_hit.values())
    require(
        contaminated_endpoints == EXPECTED_CONTAMINATED_ENDPOINTS,
        (
            "EXPECTED_191_CONTAMINATED_PRIMARY_ENDPOINTS_GOT:"
            f"{contaminated_endpoints}"
        ),
    )

    membership_rows: list[dict[str, Any]] = []
    keep_by_pair: dict[str, bool] = {}
    for pair in primary.itertuples(index=False):
        hit_i = endpoint_hit[str(pair.record_id_i)]
        hit_j = endpoint_hit[str(pair.record_id_j)]
        contaminated = bool(hit_i or hit_j)
        keep = not contaminated
        keep_by_pair[str(pair.pair_id)] = keep
        membership_rows.append(
            {
                "pair_id": str(pair.pair_id),
                "comparison_pair_key": str(pair.comparison_pair_key),
                "record_id_i": str(pair.record_id_i),
                "record_id_j": str(pair.record_id_j),
                "protacdb_endpoint_i_exact_iso": endpoint_exact_iso[
                    str(pair.record_id_i)
                ],
                "protacdb_endpoint_i_exact_noniso": endpoint_exact_noniso[
                    str(pair.record_id_i)
                ],
                "protacdb_endpoint_i_max_ecfp4": (
                    f"{endpoint_max_similarity[str(pair.record_id_i)]:.12f}"
                ),
                "protacdb_endpoint_j_exact_iso": endpoint_exact_iso[
                    str(pair.record_id_j)
                ],
                "protacdb_endpoint_j_exact_noniso": endpoint_exact_noniso[
                    str(pair.record_id_j)
                ],
                "protacdb_endpoint_j_max_ecfp4": (
                    f"{endpoint_max_similarity[str(pair.record_id_j)]:.12f}"
                ),
                "protacdb_union_exclude": contaminated,
                "protacdb_union_keep": keep,
            }
        )

    retained = sum(keep_by_pair.values())
    excluded = len(keep_by_pair) - retained
    require(
        (retained, excluded) == (EXPECTED_RETAINED_PAIRS, EXPECTED_EXCLUDED_PAIRS),
        f"EXPECTED_589_285_PAIR_CENSUS_GOT:{retained}:{excluded}",
    )
    endpoint_summary = {
        "primary_endpoints": len(endpoint_ids),
        "protacdb_structure_overlap_endpoints": contaminated_endpoints,
        "retained_primary_endpoints": len(endpoint_ids) - contaminated_endpoints,
    }
    return membership_rows, keep_by_pair, endpoint_summary


def validate_predictions(
    predictions: pd.DataFrame,
    primary: pd.DataFrame,
) -> None:
    required = (
        "pair_id",
        "comparison_pair_key",
        "record_id_i",
        "record_id_j",
        "true_delta",
        "prediction",
        "model",
        "model_seed",
        "split_seed",
    )
    require_columns(predictions, required, "predictions")
    require_nonblank(predictions, required, "predictions")
    primary_by_id = primary.set_index("pair_id", verify_integrity=True)
    require(
        set(predictions["pair_id"]) == set(primary_by_id.index),
        "PREDICTION_PAIR_SET_MISMATCH",
    )
    duplicate_columns = ["model", "model_seed", "split_seed", "pair_id"]
    require(
        not predictions.duplicated(duplicate_columns).any(),
        "DUPLICATE_PREDICTION_STREAM_PAIR",
    )
    expected_identities = set(MODEL_IDENTITIES)
    observed_identities = set(
        zip(predictions["model"].astype(str), predictions["model_seed"].astype(int))
    )
    require(observed_identities == expected_identities, "MODEL_IDENTITY_SET_MISMATCH")
    require(
        set(predictions["split_seed"].astype(int)) == set(SPLIT_SEEDS),
        "SPLIT_SEED_SET_MISMATCH",
    )
    require(
        len(predictions) == len(MODEL_IDENTITIES) * len(SPLIT_SEEDS) * len(primary),
        f"PREDICTION_ROW_COUNT_MISMATCH:{len(predictions)}",
    )
    expected_pairs = len(primary)
    stream_sizes = predictions.groupby(
        ["model", "model_seed", "split_seed"], dropna=False
    ).size()
    require(
        len(stream_sizes) == len(MODEL_IDENTITIES) * len(SPLIT_SEEDS)
        and (stream_sizes == expected_pairs).all(),
        "INCOMPLETE_PREDICTION_STREAM",
    )
    for column in ("comparison_pair_key", "record_id_i", "record_id_j"):
        expected = predictions["pair_id"].map(primary_by_id[column])
        require(
            expected.notna().all()
            and (expected.astype(str) == predictions[column].astype(str)).all(),
            f"PREDICTION_{column.upper()}_MISMATCH",
        )
    truth = predictions["true_delta"].to_numpy(float)
    values = predictions["prediction"].to_numpy(float)
    require(
        np.isfinite(truth).all()
        and np.isfinite(values).all()
        and not (truth == 0).any(),
        "NONFINITE_PREDICTION_OR_ZERO_TRUTH",
    )


def rescore(
    predictions: pd.DataFrame,
    primary: pd.DataFrame,
    keep_by_pair: dict[str, bool],
) -> list[dict[str, Any]]:
    primary_ids = set(primary["pair_id"].astype(str))
    retained_ids = {pair_id for pair_id, keep in keep_by_pair.items() if keep}
    support_by_policy = {
        "primary": primary_ids,
        "protacdb_union_known_exclude_keep": retained_ids,
    }
    rows: list[dict[str, Any]] = []
    for policy in POLICIES:
        support_ids = support_by_policy[policy]
        policy_frame = predictions[predictions["pair_id"].isin(support_ids)]
        for model, model_seed in MODEL_IDENTITIES:
            for split_seed in SPLIT_SEEDS:
                group = policy_frame[
                    (policy_frame["model"] == model)
                    & (policy_frame["model_seed"].astype(int) == model_seed)
                    & (policy_frame["split_seed"].astype(int) == split_seed)
                ]
                require(
                    len(group) == len(support_ids),
                    (
                        "INCOMPLETE_RESCORE_STREAM:"
                        f"{policy}:{model}:{model_seed}:{split_seed}:{len(group)}"
                    ),
                )
                values = metric_values(
                    np, group["true_delta"], group["prediction"]
                )
                for metric in METRICS:
                    rows.append(
                        {
                            "policy": policy,
                            "model": model,
                            "model_seed": model_seed,
                            "split_seed": split_seed,
                            "metric": metric,
                            "rows": len(group),
                            "keys": group["comparison_pair_key"].nunique(),
                            "value": values[metric],
                            "status": "COMPUTED",
                        }
                    )
    require(len(rows) == 360, f"EXPECTED_360_METRIC_ROWS_GOT:{len(rows)}")
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    construction_dir = args.construction_dir.resolve()
    protacdb_path = args.protacdb_csv.resolve()
    predictions_path = resolve_predictions(args.predictions)
    output_dir = args.output_dir.resolve()
    records_path = construction_dir / "records_aggregated_v0.1.csv"
    pairs_path = construction_dir / "pairs_v0.1.csv"
    primary_map_path = construction_dir / "primary_pair_map.csv"

    validate_fixed_input(records_path, EXPECTED_RECORDS_SHA256, "records")
    validate_fixed_input(pairs_path, EXPECTED_PAIRS_SHA256, "pairs")
    validate_fixed_input(primary_map_path, EXPECTED_PRIMARY_MAP_SHA256, "primary_map")
    validate_fixed_input(protacdb_path, EXPECTED_PROTACDB_SHA256, "protacdb_csv")

    records = pd.read_csv(records_path, low_memory=False)
    pairs = pd.read_csv(pairs_path, low_memory=False)
    primary_map = pd.read_csv(primary_map_path, low_memory=False)
    protacdb = pd.read_csv(protacdb_path, low_memory=False)
    predictions = pd.read_csv(predictions_path, low_memory=False)
    require_columns(
        records,
        ("record_id", "canonical_isomeric_smiles_full", "identity_hash"),
        "records",
    )
    require_columns(
        pairs,
        (
            "pair_id",
            "comparison_pair_key",
            "record_id_i",
            "record_id_j",
        ),
        "pairs",
    )
    require_columns(primary_map, ("pair_id",), "primary_map")
    require_columns(protacdb, ("Smiles",), "protacdb")
    require(len(protacdb) == EXPECTED_PROTACDB_ROWS, "PROTACDB_ROW_COUNT_MISMATCH")
    require(
        primary_map["pair_id"].nunique() == len(primary_map) == EXPECTED_PRIMARY_PAIRS,
        "PRIMARY_MAP_CENSUS_MISMATCH",
    )
    pair_by_id = pairs.set_index("pair_id", verify_integrity=True)
    require(
        set(primary_map["pair_id"]).issubset(set(pair_by_id.index)),
        "PRIMARY_PAIR_MISSING_FROM_PAIRS",
    )
    primary = pair_by_id.loc[primary_map["pair_id"]].reset_index()
    require(
        primary["comparison_pair_key"].nunique() == EXPECTED_PRIMARY_KEYS,
        "PRIMARY_KEY_CENSUS_MISMATCH",
    )

    anchor = build_anchor("protacdb", protacdb, "Smiles")
    require(anchor["parsed_rows"] == EXPECTED_PROTACDB_ROWS, "PROTACDB_PARSE_CENSUS_MISMATCH")
    membership_rows, keep_by_pair, endpoint_summary = build_membership(
        records, primary, anchor
    )
    validate_predictions(predictions, primary)
    metric_rows = rescore(predictions, primary, keep_by_pair)

    output_dir.mkdir(parents=True, exist_ok=True)
    membership_path = output_dir / "source_overlap_membership.csv"
    metrics_path = output_dir / "source_overlap_metrics.csv"
    summary_path = output_dir / "source_overlap_summary.json"
    membership_fields = [
        "pair_id",
        "comparison_pair_key",
        "record_id_i",
        "record_id_j",
        "protacdb_endpoint_i_exact_iso",
        "protacdb_endpoint_i_exact_noniso",
        "protacdb_endpoint_i_max_ecfp4",
        "protacdb_endpoint_j_exact_iso",
        "protacdb_endpoint_j_exact_noniso",
        "protacdb_endpoint_j_max_ecfp4",
        "protacdb_union_exclude",
        "protacdb_union_keep",
    ]
    metric_fields = [
        "policy",
        "model",
        "model_seed",
        "split_seed",
        "metric",
        "rows",
        "keys",
        "value",
        "status",
    ]
    write_csv(membership_path, membership_rows, membership_fields)
    write_csv(metrics_path, metric_rows, metric_fields)

    retained_keys = len(
        {
            row["comparison_pair_key"]
            for row in membership_rows
            if row["protacdb_union_keep"]
        }
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS_LOCAL_FIXED_PROTACDB_OVERLAP_AND_RESCORE",
        "release_boundary": {
            "row_level_outputs": "LOCAL_ONLY_DO_NOT_REDISTRIBUTE_WITHOUT_SOURCE_PERMISSION",
            "network_used": False,
            "source_terms_accepted_by_script": False,
            "public_acquisition_gap_closed": False,
        },
        "scope": {
            "anchor": "PROTACDB_FIXED_STRUCTURE_OVERLAP_ONLY",
            "doi_overlap_recomputed": False,
            "external_anchor_labels_recomputed": False,
            "wurz_anchor_labels_recomputed": False,
            "manual_adjudication_recomputed": False,
            "equivalence_basis": (
                "Frozen audit evidence established that external and Wurz anchors "
                "added no primary pair outside the PROTAC-DB union; this script "
                "does not recreate their separate labels."
            ),
        },
        "inputs": {
            "records": file_receipt(records_path),
            "pairs": file_receipt(pairs_path),
            "primary_pair_map": file_receipt(primary_map_path),
            "protacdb_csv": file_receipt(protacdb_path),
            "predictions": file_receipt(predictions_path),
        },
        "runtime": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "rdkit": rdkit.__version__,
        },
        "anchor_census": {
            "rows": anchor["row_count"],
            "parsed_rows": anchor["parsed_rows"],
            "parse_failed_rows": anchor["parse_failed_rows"],
            "unique_isomeric_graphs": len(anchor["iso"]),
            "unique_nonisomeric_graphs": len(anchor["noniso"]),
        },
        "membership": {
            **endpoint_summary,
            "primary_pairs": len(membership_rows),
            "excluded_pairs": sum(not keep for keep in keep_by_pair.values()),
            "retained_pairs": sum(keep_by_pair.values()),
            "retained_comparison_keys": retained_keys,
        },
        "rescoring": {
            "policies": POLICIES,
            "models": len(MODEL_IDENTITIES),
            "split_seeds": len(SPLIT_SEEDS),
            "metrics": METRICS,
            "output_rows": len(metric_rows),
        },
        "outputs": {
            "local_membership": file_receipt(membership_path),
            "local_metrics": file_receipt(metrics_path),
        },
    }
    write_json(summary_path, summary)
    return summary


def main() -> int:
    try:
        summary = run(parse_args())
    except ExecutionError as exc:
        print(f"SOURCE_OVERLAP_FAILED:{exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
