"""Reproduce the bounded full-graph audit for the 16 binary-equal primary pairs.

Inputs are taken only from a reconstructed construction directory.  Pair and
endpoint identities, structures, and the 16-row scope are derived at runtime;
no historical identity mapping or manually copied structure table is used.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable

import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors

from lib.structure_core import (
    STRUCT_CH2,
    STRUCT_EG,
    STRUCT_STEREO,
    STRUCT_UNRESOLVED,
    AuditError,
    audit_chain_pair,
    audit_stereo_pair,
    classify_formula,
    prepare_mol,
    require,
    sha256_file,
    sha256_text,
    write_csv_lf,
    write_json_lf,
)


SCHEMA_VERSION = "public-structure-audit-v1"
EXPECTED_PAIR_ROWS = 16
EXPECTED_IDENTITY_PAIR_GROUPS = 13
EXPECTED_ENDPOINT_RECORDS = 25
EXPECTED_IDENTITIES = 20
RADIUS = 2
FP_SIZE = 2048

PAIR_COLUMNS = (
    "pair_id",
    "comparison_pair_key",
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
    "stereo_status",
)
PRIMARY_COLUMNS = ("pair_id",)


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


def index_unique(
    rows: list[dict[str, str]], key: str, label: str
) -> dict[str, dict[str, str]]:
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


def file_receipt(path: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def unordered_identity_key(pair: dict[str, str]) -> str:
    identities = sorted((pair["identity_hash_i"], pair["identity_hash_j"]))
    return sha256_text("|".join(identities))


def structural_change_text(result: dict[str, Any]) -> str:
    structure_class = result["structure_class"]
    if structure_class == STRUCT_CH2:
        return f"one acyclic CH2 chain segment; delta={result['modified_unit_count']} CH2"
    if structure_class == STRUCT_EG:
        return (
            "one acyclic polyether segment; "
            f"delta={result['modified_unit_count']} C2H4O"
        )
    if structure_class == STRUCT_STEREO:
        return "same atom/bond connectivity; one mapped R/S descriptor differs"
    return "strict full-graph relation unresolved"


def compact_result(
    index: int, pair: dict[str, str], structural: dict[str, Any]
) -> dict[str, Any]:
    structure_class = structural["structure_class"]
    if structure_class in {STRUCT_CH2, STRUCT_EG}:
        direction = (
            f"{structural['longer_endpoint_role']}>"
            f"{structural['shorter_endpoint_role']}"
        )
        delta = (
            f"{structural['modified_unit_count']} {structural['unit_formula']} "
            f"({direction})"
        )
        ports = structural["chosen_edit"]["ports"]
        ports_short = ";".join(
            f"{structural['longer_endpoint_role']}:"
            f"{port['port_atom_index_in_longer_endpoint']}->"
            f"{structural['shorter_endpoint_role']}:"
            f"{port['port_atom_index_in_shorter_endpoint']}"
            for port in ports
        )
        graph_check = "PASS_EXACT_LABELED_GRAPH_AFTER_SINGLE_PATH_EDIT"
        stereo_check = "PASS_ALL_MAPPED_RECORDED_STEREO"
    elif structure_class == STRUCT_STEREO:
        delta = "1 mapped R/S descriptor"
        ports_short = "NA_STEREO_ONLY"
        graph_check = "PASS_CONNECTIVITY_EXACT_ONE_STEREO_DIFFERENCE"
        stereo_check = "PASS_EXACTLY_ONE_MAPPED_R_S_DIFFERENCE"
    else:
        delta = "UNRESOLVED"
        ports_short = "UNRESOLVED"
        graph_check = "UNRESOLVED"
        stereo_check = "UNRESOLVED"
    return {
        "audit": index,
        "pair_id_short": pair["pair_id"][:12],
        "pair_id": pair["pair_id"],
        "unordered_identity_pair_key_sha256": unordered_identity_key(pair),
        "poi": pair["normalized_poi_id"],
        "record_id_i": pair["record_id_i"],
        "record_id_j": pair["record_id_j"],
        "structure_class": structure_class,
        "structural_change": structural_change_text(structural),
        "delta_units": delta,
        "ports_long_to_short": ports_short,
        "port_full_graph_check": graph_check,
        "stereo_check": stereo_check,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the 16 primary cliff pairs whose reconstructed radius-2/2048 "
            "non-chiral binary Morgan fingerprints are equal."
        )
    )
    parser.add_argument(
        "--construction-dir",
        required=True,
        type=Path,
        help=(
            "Directory containing pairs_v0.1.csv, records_aggregated_v0.1.csv, "
            "and primary_pair_map.csv."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for the 16-row CSV, detailed evidence, and summary.",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    construction_dir = args.construction_dir.resolve()
    output_dir = args.output_dir.resolve()
    pair_path = construction_dir / "pairs_v0.1.csv"
    record_path = construction_dir / "records_aggregated_v0.1.csv"
    primary_path = construction_dir / "primary_pair_map.csv"

    pairs = read_csv(pair_path, PAIR_COLUMNS)
    records = read_csv(record_path, RECORD_COLUMNS)
    primary_rows = read_csv(primary_path, PRIMARY_COLUMNS)
    pairs_by_id = index_unique(pairs, "pair_id", "pair")
    records_by_id = index_unique(records, "record_id", "record")
    primary_by_id = index_unique(primary_rows, "pair_id", "primary_pair")

    primary_ids = set(primary_by_id)
    require(
        not (primary_ids - set(pairs_by_id)),
        f"PRIMARY_PAIR_IDS_MISSING_FROM_PAIRS:{len(primary_ids - set(pairs_by_id))}",
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
        len(selected_pairs) == EXPECTED_PAIR_ROWS,
        f"EXPECTED_16_SELECTED_PAIRS_GOT:{len(selected_pairs)}",
    )
    selected_record_ids = {
        record_id
        for pair in selected_pairs
        for record_id in (pair["record_id_i"], pair["record_id_j"])
    }
    require(
        len(selected_record_ids) == EXPECTED_ENDPOINT_RECORDS,
        f"EXPECTED_25_SELECTED_ENDPOINTS_GOT:{len(selected_record_ids)}",
    )
    missing_records = selected_record_ids - set(records_by_id)
    require(
        not missing_records,
        f"SELECTED_RECORD_IDS_MISSING_FROM_RECORDS:{len(missing_records)}",
    )

    selected_identities = {
        value
        for pair in selected_pairs
        for value in (pair["identity_hash_i"], pair["identity_hash_j"])
    }
    require(
        len(selected_identities) == EXPECTED_IDENTITIES,
        f"EXPECTED_20_SELECTED_IDENTITIES_GOT:{len(selected_identities)}",
    )
    identity_groups = {unordered_identity_key(pair) for pair in selected_pairs}
    require(
        len(identity_groups) == EXPECTED_IDENTITY_PAIR_GROUPS,
        f"EXPECTED_13_IDENTITY_PAIRS_GOT:{len(identity_groups)}",
    )

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=RADIUS,
        fpSize=FP_SIZE,
        includeChirality=False,
        countSimulation=False,
        includeRedundantEnvironments=False,
    )
    molecules: dict[str, Chem.Mol] = {}
    binary_bits: dict[str, frozenset[int]] = {}
    for record_id in sorted(selected_record_ids):
        record = records_by_id[record_id]
        molecule = prepare_mol(record["canonical_isomeric_smiles_full"])
        canonical = Chem.MolToSmiles(
            molecule, canonical=True, isomericSmiles=True
        )
        require(
            canonical == record["canonical_isomeric_smiles_full"],
            f"CANONICAL_ISOMERIC_SMILES_MISMATCH:{record_id}",
        )
        require(
            sha256_text(canonical) == record["identity_hash"],
            f"IDENTITY_HASH_MISMATCH:{record_id}",
        )
        molecules[record_id] = molecule
        binary_bits[record_id] = frozenset(
            int(bit) for bit in generator.GetFingerprint(molecule).GetOnBits()
        )

    detailed_rows: list[dict[str, Any]] = []
    compact_rows: list[dict[str, Any]] = []
    for index, pair in enumerate(selected_pairs, start=1):
        record_i = records_by_id[pair["record_id_i"]]
        record_j = records_by_id[pair["record_id_j"]]
        require(
            record_i["identity_hash"] == pair["identity_hash_i"],
            f"LEFT_IDENTITY_JOIN_MISMATCH:{pair['pair_id']}",
        )
        require(
            record_j["identity_hash"] == pair["identity_hash_j"],
            f"RIGHT_IDENTITY_JOIN_MISMATCH:{pair['pair_id']}",
        )
        require(
            pair["identity_hash_i"] != pair["identity_hash_j"],
            f"SELECTED_ENDPOINT_IDENTITIES_NOT_DISTINCT:{pair['pair_id']}",
        )
        require(
            binary_bits[pair["record_id_i"]] == binary_bits[pair["record_id_j"]],
            f"RECOMPUTED_BINARY_NOT_EQUAL:{pair['pair_id']}",
        )

        mol_i = molecules[pair["record_id_i"]]
        mol_j = molecules[pair["record_id_j"]]
        formula_i = rdMolDescriptors.CalcMolFormula(mol_i)
        formula_j = rdMolDescriptors.CalcMolFormula(mol_j)
        endpoint_i = {
            "record_id": pair["record_id_i"],
            "source_stereo_status": record_i["stereo_status"],
        }
        endpoint_j = {
            "record_id": pair["record_id_j"],
            "source_stereo_status": record_j["stereo_status"],
        }
        formula_class = classify_formula(mol_i, mol_j, formula_i, formula_j)
        if formula_class["candidate_class"] in {STRUCT_CH2, STRUCT_EG}:
            structural = audit_chain_pair(
                mol_i, mol_j, formula_class, endpoint_i, endpoint_j
            )
        elif formula_class["candidate_class"] == STRUCT_STEREO:
            structural = audit_stereo_pair(mol_i, mol_j, endpoint_i, endpoint_j)
        else:
            structural = {
                "structure_class": STRUCT_UNRESOLVED,
                "reason": (
                    "Formula difference is not CH2-repeat, C2H4O-repeat, "
                    "or equal-size stereo-only"
                ),
                "formula_classification": {
                    **formula_class,
                    "formula_delta": dict(formula_class["formula_delta"]),
                },
            }

        group_key = unordered_identity_key(pair)
        detailed_rows.append(
            {
                "audit_pair_index": index,
                "pair_id": pair["pair_id"],
                "pair_id_short": pair["pair_id"][:12],
                "comparison_pair_key": pair["comparison_pair_key"],
                "unordered_identity_pair_key_sha256": group_key,
                "normalized_poi_id": pair["normalized_poi_id"],
                "record_id_i": pair["record_id_i"],
                "record_id_j": pair["record_id_j"],
                "identity_hash_i": pair["identity_hash_i"],
                "identity_hash_j": pair["identity_hash_j"],
                "source_tanimoto": pair["tanimoto"],
                "recomputed_binary_equal": True,
                "input_graph_summary": {
                    "i": {
                        "canonical_isomeric_smiles_sha256": sha256_text(
                            record_i["canonical_isomeric_smiles_full"]
                        ),
                        "molecular_formula": formula_i,
                        "atom_count": mol_i.GetNumAtoms(),
                        "bond_count": mol_i.GetNumBonds(),
                        "ring_count": mol_i.GetRingInfo().NumRings(),
                        "formal_charge": Chem.GetFormalCharge(mol_i),
                        "source_stereo_status": record_i["stereo_status"],
                    },
                    "j": {
                        "canonical_isomeric_smiles_sha256": sha256_text(
                            record_j["canonical_isomeric_smiles_full"]
                        ),
                        "molecular_formula": formula_j,
                        "atom_count": mol_j.GetNumAtoms(),
                        "bond_count": mol_j.GetNumBonds(),
                        "ring_count": mol_j.GetRingInfo().NumRings(),
                        "formal_charge": Chem.GetFormalCharge(mol_j),
                        "source_stereo_status": record_j["stereo_status"],
                    },
                },
                "structural_evidence": structural,
            }
        )
        compact_rows.append(compact_result(index, pair, structural))

    group_first: dict[str, dict[str, Any]] = {}
    for row in detailed_rows:
        key = row["unordered_identity_pair_key_sha256"]
        if key in group_first:
            require(
                group_first[key]["structural_evidence"]["structure_class"]
                == row["structural_evidence"]["structure_class"],
                f"IDENTITY_PAIR_STRUCTURE_CLASS_MISMATCH:{key}",
            )
        else:
            group_first[key] = row
    require(
        len(group_first) == EXPECTED_IDENTITY_PAIR_GROUPS,
        f"EXPECTED_13_IDENTITY_PAIRS_GOT:{len(group_first)}",
    )

    unresolved = [
        row
        for row in detailed_rows
        if row["structural_evidence"]["structure_class"] == STRUCT_UNRESOLVED
    ]
    row_counts = Counter(
        row["structural_evidence"]["structure_class"] for row in detailed_rows
    )
    group_counts = Counter(
        row["structural_evidence"]["structure_class"]
        for row in group_first.values()
    )
    status = (
        "PASS_ALL_16_STRICT_FULL_GRAPH_RELATIONS"
        if not unresolved
        else "PARTIAL_STRICT_GRAPH_RELATIONS"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    compact_path = output_dir / "structure_audit_pairs.csv"
    evidence_path = output_dir / "structure_audit_evidence.json"
    summary_path = output_dir / "structure_audit_summary.json"
    compact_fields = (
        "audit",
        "pair_id_short",
        "pair_id",
        "unordered_identity_pair_key_sha256",
        "poi",
        "record_id_i",
        "record_id_j",
        "structure_class",
        "structural_change",
        "delta_units",
        "ports_long_to_short",
        "port_full_graph_check",
        "stereo_check",
    )
    write_csv_lf(compact_path, compact_rows, list(compact_fields))
    evidence = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "scope": {
            "pair_records": len(detailed_rows),
            "unordered_identity_pairs": len(group_first),
            "endpoint_records": len(selected_record_ids),
            "canonical_isomeric_identities": len(selected_identities),
            "population_limit": (
                "primary-map pairs with source Tanimoto exactly one; no extension "
                "to the other primary pairs"
            ),
        },
        "runtime": {
            "python": sys.version,
            "rdkit": rdkit.__version__,
        },
        "algorithm_contract": {
            "selection": (
                "primary-map membership, source Tanimoto exactly one, and "
                "recomputed equality of radius-2/2048 non-chiral binary Morgan bits"
            ),
            "uses_historical_identity_mapping": False,
            "uses_fixed_structure_constants": False,
            "uses_mcs": False,
            "uses_ring_fragment_multiset_as_proof": False,
            "chain_acceptance": (
                "single connected acyclic degree-two formula-supported path deletion "
                "plus exact chirality-aware labeled full-graph isomorphism after "
                "two-port reconnection"
            ),
            "stereo_acceptance": (
                "exact atom/bond connectivity and exactly one mapped R/S descriptor "
                "difference"
            ),
        },
        "inputs": {
            "pairs": file_receipt(pair_path),
            "records": file_receipt(record_path),
            "primary_pair_map": file_receipt(primary_path),
        },
        "counts": {
            "structure_by_pair_record": dict(row_counts),
            "structure_by_unordered_identity_pair": dict(group_counts),
        },
        "unresolved_pair_records": [row["pair_id"] for row in unresolved],
        "pair_results": detailed_rows,
    }
    write_json_lf(evidence_path, evidence)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "selection": evidence["scope"],
        "counts": evidence["counts"],
        "unresolved_pair_records": evidence["unresolved_pair_records"],
        "outputs": {
            "compact_16_row_csv": file_receipt(compact_path),
            "detailed_machine_evidence": file_receipt(evidence_path),
        },
    }
    write_json_lf(summary_path, summary)
    return summary


def main() -> int:
    summary = run(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not summary["unresolved_pair_records"] else 2


if __name__ == "__main__":
    RDLogger.DisableLog("rdApp.warning")
    try:
        raise SystemExit(main())
    except AuditError as exc:
        print(f"STRUCTURE_AUDIT_FAILED:{exc}", file=sys.stderr)
        raise SystemExit(1)
