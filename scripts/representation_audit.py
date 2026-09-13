"""Reproduce the bounded Morgan-representation audit for the fixed primary pairs.

The script reads authorized local inputs and writes aggregate results only. It does
not emit SMILES, record identifiers, pair identifiers, or sparse feature IDs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable

import rdkit
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, rdFingerprintGenerator


RADIUS = 2
FP_SIZE = 2048

PAIR_COLUMNS = (
    "pair_id",
    "record_id_i",
    "record_id_j",
    "identity_hash_i",
    "identity_hash_j",
    "normalized_poi_id",
    "tanimoto",
    "is_cliff",
)
RECORD_COLUMNS = (
    "record_id",
    "canonical_isomeric_smiles_full",
    "identity_hash",
)
PRIMARY_MAP_COLUMNS = ("pair_id",)

EXPECTED_SELECTED_PAIR_ROWS = 16
EXPECTED_SELECTED_ENDPOINT_RECORDS = 25
EXPECTED_IDENTITY_PAIR_GROUPS = 13
EXPECTED_PAIR_DIAGNOSES = {
    "FOLDING_SUPPORTED": 6,
    "MULTIPLICITY_LOSS_SUPPORTED": 8,
    "CHIRALITY_AWARE_ONLY_SEPARATION": 2,
    "UNRESOLVED_AT_TESTED_REPRESENTATIONS": 0,
}
EXPECTED_GROUP_DIAGNOSES = {
    "FOLDING_SUPPORTED": 6,
    "MULTIPLICITY_LOSS_SUPPORTED": 6,
    "CHIRALITY_AWARE_ONLY_SEPARATION": 1,
    "UNRESOLVED_AT_TESTED_REPRESENTATIONS": 0,
}


class AuditError(RuntimeError):
    """Raised when an input or scientific invariant does not hold."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def read_csv(path: Path, required_columns: Iterable[str]) -> list[dict[str, str]]:
    require(path.is_file(), f"INPUT_NOT_FOUND:{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        require(len(headers) == len(set(headers)), f"DUPLICATE_HEADER:{path.name}")
        missing = [column for column in required_columns if column not in headers]
        require(not missing, f"MISSING_COLUMNS:{path.name}:{','.join(missing)}")
        rows = list(reader)
    require(bool(rows), f"EMPTY_INPUT:{path.name}")
    return rows


def index_unique(rows: list[dict[str, str]], key: str, label: str) -> dict[str, dict[str, str]]:
    values = [row[key] for row in rows]
    require(all(values), f"EMPTY_{label.upper()}_{key.upper()}")
    require(len(values) == len(set(values)), f"DUPLICATE_{label.upper()}_{key.upper()}")
    return {row[key]: row for row in rows}


def parse_bool(value: str, label: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise AuditError(f"INVALID_BOOLEAN:{label}:{value!r}")


def decimal_is_one(value: str, label: str) -> bool:
    try:
        return Decimal(value.strip()) == Decimal(1)
    except InvalidOperation as exc:
        raise AuditError(f"INVALID_DECIMAL:{label}:{value!r}") from exc


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest().upper()


@dataclass
class EndpointRepresentation:
    identity_hash: str
    binary_fp: Any
    binary_bits: frozenset[int]
    nonchiral_counts: dict[int, int]
    chiral_counts: dict[int, int]


@dataclass(frozen=True)
class PairComparison:
    identity_pair: tuple[str, str]
    poi_id: str
    binary_equal: bool
    nonchiral_set_equal: bool
    nonchiral_count_equal: bool
    chiral_count_equal: bool
    diagnosis: str


def sparse_counts(generator: Any, molecule: Chem.Mol) -> dict[int, int]:
    return {
        int(key): int(value)
        for key, value in generator.GetSparseCountFingerprint(molecule)
        .GetNonzeroElements()
        .items()
    }


def represent_endpoint(
    row: dict[str, str], reference_generator: Any, chiral_generator: Any
) -> EndpointRepresentation:
    record_id = row["record_id"]
    source_smiles = row["canonical_isomeric_smiles_full"]
    molecule = Chem.MolFromSmiles(source_smiles)
    require(molecule is not None, f"SMILES_PARSE_FAILED:{record_id}")

    canonical_smiles = Chem.MolToSmiles(
        molecule, canonical=True, isomericSmiles=True
    )
    require(
        canonical_smiles == source_smiles,
        f"CANONICAL_ISOMERIC_SMILES_MISMATCH:{record_id}",
    )
    computed_identity = sha256_text(canonical_smiles)
    require(
        computed_identity == row["identity_hash"],
        f"IDENTITY_HASH_MISMATCH:{record_id}",
    )

    modern_binary = reference_generator.GetFingerprint(molecule)
    legacy_binary = AllChem.GetMorganFingerprintAsBitVect(
        molecule,
        RADIUS,
        nBits=FP_SIZE,
        useChirality=False,
    )
    modern_bits = frozenset(int(bit) for bit in modern_binary.GetOnBits())
    legacy_bits = frozenset(int(bit) for bit in legacy_binary.GetOnBits())
    require(
        modern_bits == legacy_bits,
        f"LEGACY_GENERATOR_BINARY_MISMATCH:{record_id}",
    )

    nonchiral = sparse_counts(reference_generator, molecule)
    chiral = sparse_counts(chiral_generator, molecule)
    require(
        {feature_id % FP_SIZE for feature_id in nonchiral} == modern_bits,
        f"SPARSE_SUPPORT_FOLD_MISMATCH:{record_id}",
    )
    return EndpointRepresentation(
        identity_hash=row["identity_hash"],
        binary_fp=modern_binary,
        binary_bits=modern_bits,
        nonchiral_counts=nonchiral,
        chiral_counts=chiral,
    )


def compare_pair(
    pair: dict[str, str],
    left: EndpointRepresentation,
    right: EndpointRepresentation,
) -> PairComparison:
    require(
        left.identity_hash == pair["identity_hash_i"],
        f"LEFT_IDENTITY_JOIN_MISMATCH:{pair['pair_id']}",
    )
    require(
        right.identity_hash == pair["identity_hash_j"],
        f"RIGHT_IDENTITY_JOIN_MISMATCH:{pair['pair_id']}",
    )

    binary_equal = left.binary_bits == right.binary_bits
    binary_tanimoto = float(
        DataStructs.TanimotoSimilarity(left.binary_fp, right.binary_fp)
    )
    require(
        binary_equal and binary_tanimoto == 1.0,
        f"REFERENCE_BINARY_NOT_EQUAL:{pair['pair_id']}",
    )

    left_support = set(left.nonchiral_counts)
    right_support = set(right.nonchiral_counts)
    set_equal = left_support == right_support
    nonchiral_count_equal = left.nonchiral_counts == right.nonchiral_counts
    chiral_count_equal = left.chiral_counts == right.chiral_counts

    if not set_equal:
        diagnosis = "FOLDING_SUPPORTED"
    elif not nonchiral_count_equal:
        diagnosis = "MULTIPLICITY_LOSS_SUPPORTED"
    elif not chiral_count_equal:
        diagnosis = "CHIRALITY_AWARE_ONLY_SEPARATION"
    else:
        diagnosis = "UNRESOLVED_AT_TESTED_REPRESENTATIONS"

    return PairComparison(
        identity_pair=tuple(sorted((left.identity_hash, right.identity_hash))),
        poi_id=pair["normalized_poi_id"],
        binary_equal=binary_equal,
        nonchiral_set_equal=set_equal,
        nonchiral_count_equal=nonchiral_count_equal,
        chiral_count_equal=chiral_count_equal,
        diagnosis=diagnosis,
    )


def exact_counter(counter: Counter[str], keys: Iterable[str]) -> dict[str, int]:
    return {key: int(counter[key]) for key in keys}


def validate_expected_benchmark(
    comparisons: list[PairComparison], selected_endpoint_count: int
) -> dict[tuple[str, str], list[PairComparison]]:
    require(
        len(comparisons) == EXPECTED_SELECTED_PAIR_ROWS,
        f"EXPECTED_16_SELECTED_PAIRS_GOT:{len(comparisons)}",
    )
    require(
        selected_endpoint_count == EXPECTED_SELECTED_ENDPOINT_RECORDS,
        f"EXPECTED_25_SELECTED_ENDPOINTS_GOT:{selected_endpoint_count}",
    )

    groups: dict[tuple[str, str], list[PairComparison]] = defaultdict(list)
    for comparison in comparisons:
        groups[comparison.identity_pair].append(comparison)
    require(
        len(groups) == EXPECTED_IDENTITY_PAIR_GROUPS,
        f"EXPECTED_13_IDENTITY_PAIRS_GOT:{len(groups)}",
    )

    for identity_pair, rows in groups.items():
        require(
            len({row.diagnosis for row in rows}) == 1,
            f"IDENTITY_PAIR_DIAGNOSIS_MISMATCH:{sha256_text('|'.join(identity_pair))}",
        )
        require(
            len({row.nonchiral_count_equal for row in rows}) == 1,
            f"IDENTITY_PAIR_NONCHIRAL_COUNT_MISMATCH:{sha256_text('|'.join(identity_pair))}",
        )
        require(
            len({row.chiral_count_equal for row in rows}) == 1,
            f"IDENTITY_PAIR_CHIRAL_COUNT_MISMATCH:{sha256_text('|'.join(identity_pair))}",
        )

    pair_diagnoses = exact_counter(Counter(row.diagnosis for row in comparisons), EXPECTED_PAIR_DIAGNOSES)
    require(
        pair_diagnoses == EXPECTED_PAIR_DIAGNOSES,
        f"PAIR_DIAGNOSIS_COUNTS:{pair_diagnoses}",
    )
    representatives = [rows[0] for rows in groups.values()]
    group_diagnoses = exact_counter(Counter(row.diagnosis for row in representatives), EXPECTED_GROUP_DIAGNOSES)
    require(
        group_diagnoses == EXPECTED_GROUP_DIAGNOSES,
        f"IDENTITY_PAIR_DIAGNOSIS_COUNTS:{group_diagnoses}",
    )

    chirality_only = [
        row
        for row in comparisons
        if row.diagnosis == "CHIRALITY_AWARE_ONLY_SEPARATION"
    ]
    require(len(chirality_only) == 2, "EXPECTED_TWO_CHIRALITY_ONLY_PAIR_ROWS")
    require(
        len({row.identity_pair for row in chirality_only}) == 1,
        "CHIRALITY_ONLY_ROWS_NOT_ONE_IDENTITY_PAIR",
    )
    require(
        len({row.poi_id for row in chirality_only}) == 2,
        "CHIRALITY_ONLY_ROWS_NOT_ACROSS_TWO_POIS",
    )
    return groups


def build_summary(
    *,
    pairs: list[dict[str, str]],
    records: list[dict[str, str]],
    primary_rows: list[dict[str, str]],
    selected_record_ids: set[str],
    endpoints: dict[str, EndpointRepresentation],
    comparisons: list[PairComparison],
    groups: dict[tuple[str, str], list[PairComparison]],
) -> dict[str, Any]:
    diagnosis_keys = tuple(EXPECTED_PAIR_DIAGNOSES)
    pair_diagnoses = exact_counter(Counter(row.diagnosis for row in comparisons), diagnosis_keys)
    group_representatives = [rows[0] for rows in groups.values()]
    group_diagnoses = exact_counter(
        Counter(row.diagnosis for row in group_representatives), diagnosis_keys
    )

    pair_results = {
        "reference_binary_equal_pairs": sum(row.binary_equal for row in comparisons),
        "nonchiral_sparse_set_distinguishable_pairs": sum(
            not row.nonchiral_set_equal for row in comparisons
        ),
        "nonchiral_sparse_count_distinguishable_pairs": sum(
            not row.nonchiral_count_equal for row in comparisons
        ),
        "chiral_sparse_count_distinguishable_pairs": sum(
            not row.chiral_count_equal for row in comparisons
        ),
        "folding_supported_pairs": pair_diagnoses["FOLDING_SUPPORTED"],
        "multiplicity_loss_supported_pairs": pair_diagnoses[
            "MULTIPLICITY_LOSS_SUPPORTED"
        ],
        "chirality_aware_only_separation_pairs": pair_diagnoses[
            "CHIRALITY_AWARE_ONLY_SEPARATION"
        ],
        "unresolved_at_tested_representations_pairs": pair_diagnoses[
            "UNRESOLVED_AT_TESTED_REPRESENTATIONS"
        ],
    }
    identity_results = {
        "groups": len(groups),
        "nonchiral_sparse_set_distinguishable_groups": sum(
            not row.nonchiral_set_equal for row in group_representatives
        ),
        "nonchiral_sparse_count_distinguishable_groups": sum(
            not row.nonchiral_count_equal for row in group_representatives
        ),
        "chiral_sparse_count_distinguishable_groups": sum(
            not row.chiral_count_equal for row in group_representatives
        ),
        "folding_supported_groups": group_diagnoses["FOLDING_SUPPORTED"],
        "multiplicity_loss_supported_groups": group_diagnoses[
            "MULTIPLICITY_LOSS_SUPPORTED"
        ],
        "chirality_aware_only_separation_groups": group_diagnoses[
            "CHIRALITY_AWARE_ONLY_SEPARATION"
        ],
        "unresolved_at_tested_representations_groups": group_diagnoses[
            "UNRESOLVED_AT_TESTED_REPRESENTATIONS"
        ],
    }
    chiral_only = [
        row
        for row in comparisons
        if row.diagnosis == "CHIRALITY_AWARE_ONLY_SEPARATION"
    ]

    return {
        "status": "PASS_EXPECTED_BOUNDED_REPRESENTATION_AUDIT",
        "scope": "POST_HOC_DESCRIPTIVE_REPRESENTATION_AUDIT_ONLY",
        "runtime": {
            "python_version": platform.python_version(),
            "rdkit_version": rdkit.__version__,
        },
        "input_census": {
            "pair_rows": len(pairs),
            "record_rows": len(records),
            "primary_pair_rows": len(primary_rows),
        },
        "selection": {
            "rule": "primary-map membership and source Tanimoto exactly equal to 1",
            "selected_pair_rows": len(comparisons),
            "selected_endpoint_records": len(selected_record_ids),
            "selected_molecular_identities": len(
                {endpoint.identity_hash for endpoint in endpoints.values()}
            ),
            "selected_unordered_identity_pairs": len(groups),
        },
        "parameters": {
            "reference_binary_morgan": {
                "radius": RADIUS,
                "fp_size": FP_SIZE,
                "include_chirality": False,
                "count_simulation": False,
                "include_redundant_environments": False,
                "apis_cross_checked": [
                    "AllChem.GetMorganFingerprintAsBitVect",
                    "rdFingerprintGenerator.GetMorganGenerator(...).GetFingerprint",
                ],
            },
            "nonchiral_sparse_count_morgan": {
                "radius": RADIUS,
                "include_chirality": False,
                "feature_ids_folded_to_2048": False,
            },
            "chiral_sparse_count_morgan": {
                "radius": RADIUS,
                "include_chirality": True,
                "feature_ids_folded_to_2048": False,
            },
        },
        "representation_results": pair_results,
        "unique_identity_pair_results": identity_results,
        "chirality_aware_only_cross_poi_check": {
            "pair_rows": len(chiral_only),
            "identity_pair_groups": len({row.identity_pair for row in chiral_only}),
            "distinct_pois": len({row.poi_id for row in chiral_only}),
        },
        "interpretation_boundaries": [
            "Sparse Morgan feature IDs are hashed and are not guaranteed collision-free.",
            "Representational separability does not imply improved prediction.",
            "The audit does not establish mechanism, transferability, causal SAR, or a causal role for chirality.",
            "No model fitting, prediction, metric reanalysis, bootstrap, or atom-level component mapping is performed.",
        ],
    }


def summary_csv_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for section in (
        "input_census",
        "selection",
        "representation_results",
        "unique_identity_pair_results",
        "chirality_aware_only_cross_poi_check",
    ):
        for metric, value in summary[section].items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                rows.append({"section": section, "metric": metric, "value": value})
    return rows


def write_outputs(output_dir: Path, summary: dict[str, Any]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "representation_audit_summary.json"
    csv_path = output_dir / "representation_audit_summary.csv"
    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    rows = summary_csv_rows(summary)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("section", "metric", "value"))
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the fixed primary cliff-pair rows whose source radius-2/2048 "
            "non-chiral binary Morgan Tanimoto is exactly one."
        )
    )
    parser.add_argument("--pairs", required=True, type=Path, help="Pair table CSV.")
    parser.add_argument("--records", required=True, type=Path, help="Record table CSV.")
    parser.add_argument(
        "--primary-pair-map",
        required=True,
        type=Path,
        help="CSV whose pair_id rows define the primary pair set.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for aggregate JSON and CSV outputs.",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    pairs = read_csv(args.pairs, PAIR_COLUMNS)
    records = read_csv(args.records, RECORD_COLUMNS)
    primary_rows = read_csv(args.primary_pair_map, PRIMARY_MAP_COLUMNS)

    pairs_by_id = index_unique(pairs, "pair_id", "pair")
    records_by_id = index_unique(records, "record_id", "record")
    primary_by_id = index_unique(primary_rows, "pair_id", "primary_map")

    primary_ids = set(primary_by_id)
    missing_primary_pairs = primary_ids - set(pairs_by_id)
    require(
        not missing_primary_pairs,
        f"PRIMARY_PAIR_IDS_MISSING_FROM_PAIRS:{len(missing_primary_pairs)}",
    )
    cliff_ids = {
        row["pair_id"]
        for row in pairs
        if parse_bool(row["is_cliff"], f"is_cliff:{row['pair_id']}")
    }
    require(primary_ids == cliff_ids, "PRIMARY_MAP_NOT_EQUAL_TO_IS_CLIFF_PAIR_SET")

    selected_pairs = [
        pairs_by_id[row["pair_id"]]
        for row in primary_rows
        if decimal_is_one(
            pairs_by_id[row["pair_id"]]["tanimoto"],
            f"tanimoto:{row['pair_id']}",
        )
    ]
    require(
        len(selected_pairs) == EXPECTED_SELECTED_PAIR_ROWS,
        f"EXPECTED_16_SELECTED_PAIRS_GOT:{len(selected_pairs)}",
    )
    for pair in selected_pairs:
        require(
            pair["identity_hash_i"] != pair["identity_hash_j"],
            f"SELECTED_ENDPOINT_IDENTITIES_NOT_DISTINCT:{pair['pair_id']}",
        )

    selected_record_ids = {
        record_id
        for pair in selected_pairs
        for record_id in (pair["record_id_i"], pair["record_id_j"])
    }
    missing_records = selected_record_ids - set(records_by_id)
    require(
        not missing_records,
        f"SELECTED_RECORD_IDS_MISSING_FROM_RECORDS:{len(missing_records)}",
    )

    reference_generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=RADIUS,
        fpSize=FP_SIZE,
        includeChirality=False,
        countSimulation=False,
        includeRedundantEnvironments=False,
    )
    chiral_generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=RADIUS,
        fpSize=FP_SIZE,
        includeChirality=True,
        countSimulation=False,
        includeRedundantEnvironments=False,
    )
    endpoints = {
        record_id: represent_endpoint(
            records_by_id[record_id], reference_generator, chiral_generator
        )
        for record_id in sorted(selected_record_ids)
    }
    comparisons = [
        compare_pair(
            pair,
            endpoints[pair["record_id_i"]],
            endpoints[pair["record_id_j"]],
        )
        for pair in selected_pairs
    ]
    groups = validate_expected_benchmark(comparisons, len(selected_record_ids))
    summary = build_summary(
        pairs=pairs,
        records=records,
        primary_rows=primary_rows,
        selected_record_ids=selected_record_ids,
        endpoints=endpoints,
        comparisons=comparisons,
        groups=groups,
    )
    write_outputs(args.output_dir, summary)
    return summary


def main() -> int:
    args = parse_args()
    summary = run(args)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "selected_pair_rows": summary["selection"]["selected_pair_rows"],
                "selected_unordered_identity_pairs": summary["selection"][
                    "selected_unordered_identity_pairs"
                ],
                "output_files": [
                    "representation_audit_summary.json",
                    "representation_audit_summary.csv",
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    RDLogger.DisableLog("rdApp.warning")
    try:
        raise SystemExit(main())
    except AuditError as exc:
        print(f"REPRESENTATION_AUDIT_FAILED:{exc}", file=sys.stderr)
        raise SystemExit(2)
