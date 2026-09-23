#!/usr/bin/env python3
"""Bounded full-graph audit for the 16 fixed fingerprint-equal pair records.

The structural classification is computed without using the inherited
representation diagnosis.  Chain cases are accepted only when deletion of a
single formula-supported, acyclic, degree-two path from the larger graph,
followed by reconnection of its two retained ports, gives an exact labeled
full-graph isomorphism to the smaller graph.  Stereo-only cases require exact
atom/bond connectivity and exactly one mapped R/S descriptor difference.

No MCS, ring-fragment collapse, model fitting, resampling, or network access is
used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors


SCHEMA_VERSION = "r31-structure-16-v1"
STRUCT_CH2 = "ACYCLIC_METHYLENE_CHAIN_LENGTH"
STRUCT_EG = "ACYCLIC_ETHYLENE_GLYCOL_REPEAT_LENGTH"
STRUCT_STEREO = "STEREOCHEMISTRY_ONLY"
STRUCT_UNRESOLVED = "UNRESOLVED_STRICT_GRAPH"


class AuditError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest().upper()


def json_digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_text(payload)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_text_lf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    tmp.replace(path)


def write_json_lf(path: Path, value: Any) -> None:
    write_text_lf(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_csv_lf(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def parse_formula(formula: str) -> Counter[str]:
    parts = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    rebuilt = "".join(element + count for element, count in parts)
    require(rebuilt == formula, f"Unsupported molecular formula syntax: {formula}")
    return Counter({element: int(count or "1") for element, count in parts})


def formula_delta(long_formula: str, short_formula: str) -> Counter[str]:
    long_counts = parse_formula(long_formula)
    short_counts = parse_formula(short_formula)
    elements = set(long_counts) | set(short_counts)
    delta = Counter({element: long_counts[element] - short_counts[element] for element in elements})
    return Counter({element: count for element, count in delta.items() if count})


def format_formula_counts(counts: Counter[str]) -> str:
    if not counts:
        return "0"
    ordered: list[str] = []
    for element in ("C", "H"):
        if element in counts:
            ordered.append(element)
    ordered.extend(sorted(element for element in counts if element not in {"C", "H"}))
    return "".join(element + (str(counts[element]) if counts[element] != 1 else "") for element in ordered)


def prepare_mol(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    require(mol is not None, f"RDKit failed to parse SMILES: {smiles}")
    Chem.SanitizeMol(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return mol


def atom_label(mol: Chem.Mol, atom_idx: int) -> dict[str, Any]:
    atom = mol.GetAtomWithIdx(atom_idx)
    return {
        "atomic_number": atom.GetAtomicNum(),
        "symbol": atom.GetSymbol(),
        "isotope": atom.GetIsotope(),
        "formal_charge": atom.GetFormalCharge(),
        "radical_electrons": atom.GetNumRadicalElectrons(),
        "aromatic": atom.GetIsAromatic(),
        "hybridization": str(atom.GetHybridization()),
        "degree": atom.GetDegree(),
        "total_degree": atom.GetTotalDegree(),
        "explicit_hydrogens": atom.GetNumExplicitHs(),
        "total_hydrogens": atom.GetTotalNumHs(includeNeighbors=True),
        "no_implicit": atom.GetNoImplicit(),
        "total_valence": atom.GetTotalValence(),
        "ring_membership_count": mol.GetRingInfo().NumAtomRings(atom_idx),
        "chiral_tag_raw": str(atom.GetChiralTag()),
        "atom_map_number": atom.GetAtomMapNum(),
    }


def atom_base_label(mol: Chem.Mol, atom_idx: int) -> dict[str, Any]:
    label = atom_label(mol, atom_idx)
    label.pop("chiral_tag_raw")
    return label


def bond_label(mol: Chem.Mol, bond: Chem.Bond) -> dict[str, Any]:
    return {
        "bond_type": str(bond.GetBondType()),
        "bond_order": float(bond.GetBondTypeAsDouble()),
        "aromatic": bond.GetIsAromatic(),
        "conjugated": bond.GetIsConjugated(),
        "ring": bond.IsInRing(),
        "stereo": str(bond.GetStereo()),
    }


def bond_base_label(mol: Chem.Mol, bond: Chem.Bond) -> dict[str, Any]:
    label = bond_label(mol, bond)
    label.pop("stereo")
    return label


def stereo_centers(mol: Chem.Mol) -> dict[int, str]:
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return {
        int(atom_idx): str(descriptor)
        for atom_idx, descriptor in Chem.FindMolChiralCenters(
            mol,
            includeUnassigned=True,
            includeCIP=True,
            useLegacyImplementation=False,
        )
    }


def stereo_center_ledger(mol: Chem.Mol) -> list[dict[str, Any]]:
    centers = stereo_centers(mol)
    return [
        {
            "atom_index": atom_idx,
            "descriptor": descriptor,
            "chiral_tag_raw": str(mol.GetAtomWithIdx(atom_idx).GetChiralTag()),
            "atom_label": atom_base_label(mol, atom_idx),
        }
        for atom_idx, descriptor in sorted(centers.items())
    ]


def bond_stereo_ledger(mol: Chem.Mol) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for bond in mol.GetBonds():
        if bond.GetStereo() == Chem.BondStereo.STEREONONE:
            continue
        records.append(
            {
                "bond_index": bond.GetIdx(),
                "begin_atom_index": bond.GetBeginAtomIdx(),
                "end_atom_index": bond.GetEndAtomIdx(),
                "stereo": str(bond.GetStereo()),
                "stereo_atom_indices": [int(idx) for idx in bond.GetStereoAtoms()],
            }
        )
    return records


def canonical_smiles(mol: Chem.Mol, isomeric: bool) -> str:
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)


def mapping_base_mismatches(
    query: Chem.Mol, target: Chem.Mol, match: tuple[int, ...]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    atom_mismatches: list[dict[str, Any]] = []
    bond_mismatches: list[dict[str, Any]] = []
    for query_idx, target_idx in enumerate(match):
        q_label = atom_base_label(query, query_idx)
        t_label = atom_base_label(target, target_idx)
        if q_label != t_label:
            atom_mismatches.append(
                {
                    "query_atom_index": query_idx,
                    "target_atom_index": target_idx,
                    "query_label": q_label,
                    "target_label": t_label,
                }
            )
    for query_bond in query.GetBonds():
        target_bond = target.GetBondBetweenAtoms(
            match[query_bond.GetBeginAtomIdx()], match[query_bond.GetEndAtomIdx()]
        )
        if target_bond is None:
            bond_mismatches.append(
                {
                    "query_bond_index": query_bond.GetIdx(),
                    "reason": "missing_mapped_target_bond",
                }
            )
            continue
        q_label = bond_base_label(query, query_bond)
        t_label = bond_base_label(target, target_bond)
        if q_label != t_label:
            bond_mismatches.append(
                {
                    "query_bond_index": query_bond.GetIdx(),
                    "target_bond_index": target_bond.GetIdx(),
                    "query_label": q_label,
                    "target_label": t_label,
                }
            )
    return atom_mismatches, bond_mismatches


def find_exact_matches(
    query: Chem.Mol, target: Chem.Mol, use_chirality: bool
) -> list[tuple[int, ...]]:
    if query.GetNumAtoms() != target.GetNumAtoms() or query.GetNumBonds() != target.GetNumBonds():
        return []
    params = Chem.SubstructMatchParameters()
    params.useChirality = use_chirality
    params.uniquify = False
    params.maxMatches = 10000
    candidates = target.GetSubstructMatches(query, params)
    valid: list[tuple[int, ...]] = []
    for match in candidates:
        if len(match) != query.GetNumAtoms() or len(set(match)) != len(match):
            continue
        atom_mm, bond_mm = mapping_base_mismatches(query, target, match)
        if not atom_mm and not bond_mm:
            valid.append(tuple(int(value) for value in match))
    return sorted(set(valid))


def mapped_stereo_comparison(
    source: Chem.Mol,
    target: Chem.Mol,
    source_to_target: dict[int, int],
) -> list[dict[str, Any]]:
    source_centers = stereo_centers(source)
    target_centers = stereo_centers(target)
    comparisons: list[dict[str, Any]] = []
    for source_idx, target_idx in sorted(source_to_target.items()):
        source_descriptor = source_centers.get(source_idx)
        target_descriptor = target_centers.get(target_idx)
        if source_descriptor is None and target_descriptor is None:
            continue
        comparisons.append(
            {
                "source_atom_index": source_idx,
                "target_atom_index": target_idx,
                "source_descriptor": source_descriptor,
                "target_descriptor": target_descriptor,
                "equal": source_descriptor == target_descriptor,
                "source_chiral_tag_raw": str(source.GetAtomWithIdx(source_idx).GetChiralTag()),
                "target_chiral_tag_raw": str(target.GetAtomWithIdx(target_idx).GetChiralTag()),
            }
        )
    return comparisons


def mapped_bond_stereo_mismatches(
    source: Chem.Mol,
    target: Chem.Mol,
    source_to_target: dict[int, int],
    excluded_source_atoms: set[int] | None = None,
) -> list[dict[str, Any]]:
    excluded = excluded_source_atoms or set()
    mismatches: list[dict[str, Any]] = []
    for source_bond in source.GetBonds():
        a = source_bond.GetBeginAtomIdx()
        b = source_bond.GetEndAtomIdx()
        if a in excluded or b in excluded:
            continue
        target_bond = target.GetBondBetweenAtoms(source_to_target[a], source_to_target[b])
        if target_bond is None:
            continue
        source_stereo = str(source_bond.GetStereo())
        target_stereo = str(target_bond.GetStereo())
        if source_stereo != target_stereo:
            mismatches.append(
                {
                    "source_bond_index": source_bond.GetIdx(),
                    "target_bond_index": target_bond.GetIdx(),
                    "source_stereo": source_stereo,
                    "target_stereo": target_stereo,
                }
            )
    return mismatches


def unit_atom_ok(mol: Chem.Mol, atom_idx: int, structure_class: str) -> bool:
    atom = mol.GetAtomWithIdx(atom_idx)
    if atom.IsInRing() or atom.GetIsAromatic() or atom.GetDegree() != 2:
        return False
    if atom.GetFormalCharge() != 0 or atom.GetIsotope() != 0 or atom.GetNumRadicalElectrons() != 0:
        return False
    if structure_class == STRUCT_CH2:
        return (
            atom.GetAtomicNum() == 6
            and atom.GetHybridization() == Chem.HybridizationType.SP3
            and atom.GetTotalNumHs(includeNeighbors=True) == 2
        )
    if structure_class == STRUCT_EG:
        if atom.GetAtomicNum() == 6:
            return (
                atom.GetHybridization() == Chem.HybridizationType.SP3
                and atom.GetTotalNumHs(includeNeighbors=True) == 2
            )
        if atom.GetAtomicNum() == 8:
            return atom.GetTotalNumHs(includeNeighbors=True) == 0
    return False


def unit_sequence_ok(symbols: list[str], structure_class: str) -> bool:
    if structure_class == STRUCT_CH2:
        return all(symbol == "C" for symbol in symbols)
    if structure_class == STRUCT_EG:
        base = ("O", "C", "C")
        return len(symbols) % 3 == 0 and any(
            all(symbol == base[(idx + offset) % 3] for idx, symbol in enumerate(symbols))
            for offset in range(3)
        )
    return False


def path_boundary(
    mol: Chem.Mol, path: tuple[int, ...]
) -> tuple[tuple[int, int, Chem.Bond], tuple[int, int, Chem.Bond]] | None:
    path_set = set(path)
    internal_bond_count = 0
    external: list[tuple[int, int, Chem.Bond]] = []
    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        if a in path_set and b in path_set:
            internal_bond_count += 1
        elif a in path_set and b not in path_set:
            external.append((a, b, bond))
        elif b in path_set and a not in path_set:
            external.append((b, a, bond))
    if internal_bond_count != len(path) - 1 or len(external) != 2:
        return None
    if external[0][1] == external[1][1]:
        return None
    if len(path) == 1:
        return tuple(sorted(external, key=lambda item: item[1]))  # type: ignore[return-value]
    by_path_atom = {item[0]: item for item in external}
    if set(by_path_atom) != {path[0], path[-1]}:
        return None
    return by_path_atom[path[0]], by_path_atom[path[-1]]


def enumerate_candidate_paths(
    mol: Chem.Mol, path_length: int, structure_class: str
) -> list[dict[str, Any]]:
    eligible = {
        idx for idx in range(mol.GetNumAtoms()) if unit_atom_ok(mol, idx, structure_class)
    }
    found: dict[tuple[int, ...], dict[str, Any]] = {}

    def visit(path: list[int]) -> None:
        if len(path) == path_length:
            canonical_path = min(tuple(path), tuple(reversed(path)))
            if canonical_path in found:
                return
            symbols = [mol.GetAtomWithIdx(idx).GetSymbol() for idx in canonical_path]
            if not unit_sequence_ok(symbols, structure_class):
                return
            boundary = path_boundary(mol, canonical_path)
            if boundary is None:
                return
            left, right = boundary
            for _, _, bond in (left, right):
                if bond.GetBondType() != Chem.BondType.SINGLE or bond.GetIsAromatic():
                    return
            for a, b in zip(canonical_path, canonical_path[1:]):
                bond = mol.GetBondBetweenAtoms(a, b)
                if bond is None or bond.GetBondType() != Chem.BondType.SINGLE or bond.GetIsAromatic():
                    return
            if mol.GetBondBetweenAtoms(left[1], right[1]) is not None:
                return
            found[canonical_path] = {
                "path": canonical_path,
                "symbols": symbols,
                "boundary": boundary,
            }
            return
        last = path[-1]
        for neighbor in mol.GetAtomWithIdx(last).GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if neighbor_idx in eligible and neighbor_idx not in path:
                bond = mol.GetBondBetweenAtoms(last, neighbor_idx)
                if bond.GetBondType() == Chem.BondType.SINGLE and not bond.GetIsAromatic():
                    visit(path + [neighbor_idx])

    for start in sorted(eligible):
        visit([start])
    return [found[key] for key in sorted(found)]


def delete_path_and_bridge(
    mol: Chem.Mol, path: tuple[int, ...], port_original_indices: tuple[int, int]
) -> tuple[Chem.Mol, dict[int, int]]:
    removed = set(path)
    remaining = [idx for idx in range(mol.GetNumAtoms()) if idx not in removed]
    original_to_edited = {original_idx: edited_idx for edited_idx, original_idx in enumerate(remaining)}
    rw = Chem.RWMol(Chem.Mol(mol))
    for atom_idx in sorted(removed, reverse=True):
        rw.RemoveAtom(atom_idx)
    rw.AddBond(
        original_to_edited[port_original_indices[0]],
        original_to_edited[port_original_indices[1]],
        Chem.BondType.SINGLE,
    )
    edited = rw.GetMol()
    Chem.SanitizeMol(edited)
    Chem.AssignStereochemistry(edited, cleanIt=True, force=True)
    return edited, original_to_edited


def detailed_original_mapping(
    source: Chem.Mol,
    target: Chem.Mol,
    source_to_target: dict[int, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    atoms: list[dict[str, Any]] = []
    atom_mismatches: list[dict[str, Any]] = []
    bonds: list[dict[str, Any]] = []
    for source_idx, target_idx in sorted(source_to_target.items()):
        source_label = atom_label(source, source_idx)
        target_label = atom_label(target, target_idx)
        source_base = atom_base_label(source, source_idx)
        target_base = atom_base_label(target, target_idx)
        row = {
            "source_atom_index": source_idx,
            "target_atom_index": target_idx,
            "source_label": source_label,
            "target_label": target_label,
            "base_label_equal": source_base == target_base,
        }
        atoms.append(row)
        if source_base != target_base:
            atom_mismatches.append(row)
    for source_bond in source.GetBonds():
        a = source_bond.GetBeginAtomIdx()
        b = source_bond.GetEndAtomIdx()
        if a not in source_to_target or b not in source_to_target:
            continue
        target_bond = target.GetBondBetweenAtoms(source_to_target[a], source_to_target[b])
        row = {
            "source_bond_index": source_bond.GetIdx(),
            "source_begin_atom_index": a,
            "source_end_atom_index": b,
            "target_begin_atom_index": source_to_target[a],
            "target_end_atom_index": source_to_target[b],
            "source_label": bond_label(source, source_bond),
            "target_bond_index": None if target_bond is None else target_bond.GetIdx(),
            "target_label": None if target_bond is None else bond_label(target, target_bond),
            "base_label_equal": (
                target_bond is not None
                and bond_base_label(source, source_bond) == bond_base_label(target, target_bond)
            ),
        }
        bonds.append(row)
    return atoms, atom_mismatches, bonds


def classify_formula(
    mol_i: Chem.Mol, mol_j: Chem.Mol, formula_i: str, formula_j: str
) -> dict[str, Any]:
    atom_i = mol_i.GetNumAtoms()
    atom_j = mol_j.GetNumAtoms()
    if atom_i == atom_j:
        return {
            "candidate_class": STRUCT_STEREO,
            "long_role": None,
            "short_role": None,
            "atom_delta": 0,
            "formula_delta": Counter(),
            "unit_count": 0,
            "unit_formula": None,
        }
    long_role, short_role = ("i", "j") if atom_i > atom_j else ("j", "i")
    long_formula, short_formula = (formula_i, formula_j) if long_role == "i" else (formula_j, formula_i)
    delta = formula_delta(long_formula, short_formula)
    atom_delta = abs(atom_i - atom_j)
    if delta == Counter({"C": atom_delta, "H": 2 * atom_delta}):
        return {
            "candidate_class": STRUCT_CH2,
            "long_role": long_role,
            "short_role": short_role,
            "atom_delta": atom_delta,
            "formula_delta": delta,
            "unit_count": atom_delta,
            "unit_formula": "CH2",
        }
    if atom_delta % 3 == 0:
        units = atom_delta // 3
        if delta == Counter({"C": 2 * units, "H": 4 * units, "O": units}):
            return {
                "candidate_class": STRUCT_EG,
                "long_role": long_role,
                "short_role": short_role,
                "atom_delta": atom_delta,
                "formula_delta": delta,
                "unit_count": units,
                "unit_formula": "C2H4O",
            }
    return {
        "candidate_class": STRUCT_UNRESOLVED,
        "long_role": long_role,
        "short_role": short_role,
        "atom_delta": atom_delta,
        "formula_delta": delta,
        "unit_count": None,
        "unit_formula": None,
    }


def audit_chain_pair(
    mol_i: Chem.Mol,
    mol_j: Chem.Mol,
    formula_class: dict[str, Any],
    endpoint_i: dict[str, str],
    endpoint_j: dict[str, str],
) -> dict[str, Any]:
    structure_class = formula_class["candidate_class"]
    long_role = formula_class["long_role"]
    short_role = formula_class["short_role"]
    long_mol, short_mol = (mol_i, mol_j) if long_role == "i" else (mol_j, mol_i)
    long_endpoint, short_endpoint = (
        (endpoint_i, endpoint_j) if long_role == "i" else (endpoint_j, endpoint_i)
    )
    paths = enumerate_candidate_paths(long_mol, formula_class["atom_delta"], structure_class)
    valid: list[dict[str, Any]] = []
    target_iso = canonical_smiles(short_mol, True)
    for candidate in paths:
        path = candidate["path"]
        boundary = candidate["boundary"]
        ports = (boundary[0][1], boundary[1][1])
        try:
            edited, original_to_edited = delete_path_and_bridge(long_mol, path, ports)
        except Exception:
            continue
        if canonical_smiles(edited, True) != target_iso:
            continue
        matches = find_exact_matches(edited, short_mol, use_chirality=True)
        if not matches:
            continue
        chosen_match = matches[0]
        source_to_target = {
            original_idx: chosen_match[edited_idx]
            for original_idx, edited_idx in original_to_edited.items()
        }
        stereo_cmp = mapped_stereo_comparison(long_mol, short_mol, source_to_target)
        stereo_mm = [row for row in stereo_cmp if not row["equal"]]
        bond_stereo_mm = mapped_bond_stereo_mismatches(
            long_mol, short_mol, source_to_target, excluded_source_atoms=set(path)
        )
        if stereo_mm or bond_stereo_mm:
            continue
        valid.append(
            {
                "path": path,
                "symbols": candidate["symbols"],
                "boundary": boundary,
                "ports": ports,
                "edited": edited,
                "original_to_edited": original_to_edited,
                "match": chosen_match,
                "match_count": len(matches),
                "source_to_target": source_to_target,
                "stereo_comparison": stereo_cmp,
            }
        )
    if not valid:
        return {
            "structure_class": STRUCT_UNRESOLVED,
            "reason": "No allowed single-path edit yielded an exact chirality-aware labeled full-graph isomorphism",
            "candidate_path_count": len(paths),
            "formula_classification": {
                **formula_class,
                "formula_delta": dict(formula_class["formula_delta"]),
            },
        }
    valid.sort(key=lambda item: (item["path"], item["match"]))
    chosen = valid[0]
    path = chosen["path"]
    path_set = set(path)
    source_to_target = chosen["source_to_target"]
    mapped_atoms, atom_mm, mapped_bonds = detailed_original_mapping(
        long_mol, short_mol, source_to_target
    )
    mapped_bond_mm = [row for row in mapped_bonds if not row["base_label_equal"]]
    inverse = {target_idx: source_idx for source_idx, target_idx in source_to_target.items()}
    short_extra_edges: list[dict[str, Any]] = []
    for short_bond in short_mol.GetBonds():
        a = short_bond.GetBeginAtomIdx()
        b = short_bond.GetEndAtomIdx()
        long_bond = long_mol.GetBondBetweenAtoms(inverse[a], inverse[b])
        if long_bond is None:
            short_extra_edges.append(
                {
                    "short_bond_index": short_bond.GetIdx(),
                    "short_begin_atom_index": a,
                    "short_end_atom_index": b,
                    "mapped_long_begin_atom_index": inverse[a],
                    "mapped_long_end_atom_index": inverse[b],
                    "short_label": bond_label(short_mol, short_bond),
                }
            )
    removed_bonds: list[dict[str, Any]] = []
    for bond in long_mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        if a in path_set or b in path_set:
            removed_bonds.append(
                {
                    "bond_index": bond.GetIdx(),
                    "begin_atom_index": a,
                    "end_atom_index": b,
                    "label": bond_label(long_mol, bond),
                }
            )
    require(not atom_mm, "Unexpected mapped atom-label mismatch after accepted edit")
    require(not mapped_bond_mm, "Unexpected mapped bond-label mismatch after accepted edit")
    require(len(short_extra_edges) == 1, "Accepted chain edit must add exactly one target bridge edge")
    require(len(removed_bonds) == len(path) + 1, "Removed path must have path_length + 1 incident bonds")
    ports = chosen["ports"]
    mapped_ports = [source_to_target[idx] for idx in ports]
    bridge = short_mol.GetBondBetweenAtoms(mapped_ports[0], mapped_ports[1])
    require(bridge is not None, "Mapped ports are not adjacent in the shorter endpoint")
    require(
        bond_base_label(short_mol, bridge)
        == {
            "bond_type": "SINGLE",
            "bond_order": 1.0,
            "aromatic": False,
            "conjugated": False,
            "ring": False,
        },
        "Shorter-endpoint port bridge is not an acyclic nonconjugated single bond",
    )
    boundary = chosen["boundary"]
    valid_summaries = [
        {
            "removed_path_atom_indices": list(item["path"]),
            "removed_path_symbols": item["symbols"],
            "port_atom_indices_in_longer_endpoint": list(item["ports"]),
            "exact_mapping_count": item["match_count"],
        }
        for item in valid
    ]
    stereo_cmp = chosen["stereo_comparison"]
    removed_atoms = [
        {
            "atom_index": idx,
            "label": atom_label(long_mol, idx),
            "neighbor_atom_indices": sorted(neighbor.GetIdx() for neighbor in long_mol.GetAtomWithIdx(idx).GetNeighbors()),
        }
        for idx in path
    ]
    port_records: list[dict[str, Any]] = []
    for path_atom_idx, port_idx, boundary_bond in boundary:
        short_idx = source_to_target[port_idx]
        port_records.append(
            {
                "path_atom_index_in_longer_endpoint": path_atom_idx,
                "port_atom_index_in_longer_endpoint": port_idx,
                "port_atom_index_in_shorter_endpoint": short_idx,
                "longer_endpoint_label": atom_label(long_mol, port_idx),
                "shorter_endpoint_label": atom_label(short_mol, short_idx),
                "boundary_bond_in_longer_endpoint": bond_label(long_mol, boundary_bond),
            }
        )
    return {
        "structure_class": structure_class,
        "reason": None,
        "longer_endpoint_role": long_role,
        "shorter_endpoint_role": short_role,
        "longer_record_id": long_endpoint["record_id"],
        "shorter_record_id": short_endpoint["record_id"],
        "unit_formula": formula_class["unit_formula"],
        "modified_unit_count": formula_class["unit_count"],
        "heavy_atom_delta": formula_class["atom_delta"],
        "formula_delta_long_minus_short": format_formula_counts(formula_class["formula_delta"]),
        "candidate_path_count_before_full_graph_filter": len(paths),
        "valid_exact_edit_count": len(valid),
        "valid_exact_edit_summaries": valid_summaries,
        "chosen_edit": {
            "removed_path_atom_indices_in_longer_endpoint": list(path),
            "removed_path_symbols": chosen["symbols"],
            "removed_path_atom_records": removed_atoms,
            "removed_segment_bonds": removed_bonds,
            "ports": port_records,
            "port_bridge_in_shorter_endpoint": {
                "shorter_endpoint_atom_indices": mapped_ports,
                "label": bond_label(short_mol, bridge),
            },
        },
        "full_graph_mapping": {
            "source_role": long_role,
            "target_role": short_role,
            "mapped_atom_count": len(source_to_target),
            "removed_atom_count": len(path),
            "mapped_atoms": mapped_atoms,
            "mapped_bonds_outside_removed_path": mapped_bonds,
            "mapped_atom_base_label_mismatch_count": len(atom_mm),
            "mapped_bond_base_label_mismatch_count": len(mapped_bond_mm),
            "shorter_edges_without_original_longer_edge": short_extra_edges,
            "shorter_extra_edge_count": len(short_extra_edges),
            "edited_graph_canonical_isomeric_smiles": canonical_smiles(chosen["edited"], True),
            "shorter_graph_canonical_isomeric_smiles": canonical_smiles(short_mol, True),
            "canonical_isomeric_smiles_equal_after_edit": canonical_smiles(chosen["edited"], True)
            == canonical_smiles(short_mol, True),
            "chirality_aware_exact_isomorphism": True,
            "mapping_sha256": json_digest(
                [{"longer_atom_index": k, "shorter_atom_index": v} for k, v in sorted(source_to_target.items())]
            ),
            "mapped_bond_ledger_sha256": json_digest(mapped_bonds),
        },
        "stereo_check": {
            "longer_source_stereo_status": long_endpoint["source_stereo_status"],
            "shorter_source_stereo_status": short_endpoint["source_stereo_status"],
            "longer_centers": stereo_center_ledger(long_mol),
            "shorter_centers": stereo_center_ledger(short_mol),
            "mapped_center_comparison": stereo_cmp,
            "mapped_center_mismatch_count": sum(not row["equal"] for row in stereo_cmp),
            "mapped_bond_stereo_mismatch_count": len(
                mapped_bond_stereo_mismatches(
                    long_mol, short_mol, source_to_target, excluded_source_atoms=path_set
                )
            ),
            "recorded_bond_stereo_longer": bond_stereo_ledger(long_mol),
            "recorded_bond_stereo_shorter": bond_stereo_ledger(short_mol),
            "all_mapped_recorded_stereo_preserved": True,
        },
        "strict_checks": {
            "single_connected_removed_path": True,
            "removed_path_all_acyclic": all(not long_mol.GetAtomWithIdx(idx).IsInRing() for idx in path),
            "removed_path_all_degree_two": all(long_mol.GetAtomWithIdx(idx).GetDegree() == 2 for idx in path),
            "exactly_two_retained_ports": len(port_records) == 2,
            "all_removed_and_boundary_bonds_single_nonaromatic": all(
                row["label"]["bond_type"] == "SINGLE" and not row["label"]["aromatic"]
                for row in removed_bonds
            ),
            "remaining_atom_labels_exact": not atom_mm,
            "remaining_bond_labels_exact": not mapped_bond_mm,
            "port_bridge_exact": len(short_extra_edges) == 1,
            "recorded_stereo_exact": sum(not row["equal"] for row in stereo_cmp) == 0,
            "full_graph_exact_after_edit": True,
        },
    }


def audit_stereo_pair(
    mol_i: Chem.Mol,
    mol_j: Chem.Mol,
    endpoint_i: dict[str, str],
    endpoint_j: dict[str, str],
) -> dict[str, Any]:
    if canonical_smiles(mol_i, False) != canonical_smiles(mol_j, False):
        return {
            "structure_class": STRUCT_UNRESOLVED,
            "reason": "Equal-size pair is not identical under canonical nonisomeric full-graph serialization",
        }
    matches = find_exact_matches(mol_i, mol_j, use_chirality=False)
    if not matches:
        return {
            "structure_class": STRUCT_UNRESOLVED,
            "reason": "No exact atom/bond labeled connectivity isomorphism found when stereo is excluded",
        }
    ranked: list[tuple[int, tuple[int, ...], list[dict[str, Any]]]] = []
    for match in matches:
        mapping = {idx: target_idx for idx, target_idx in enumerate(match)}
        comparisons = mapped_stereo_comparison(mol_i, mol_j, mapping)
        mismatch_count = sum(not row["equal"] for row in comparisons)
        ranked.append((mismatch_count, match, comparisons))
    ranked.sort(key=lambda item: (item[0], item[1]))
    mismatch_count, chosen_match, stereo_cmp = ranked[0]
    mapping = {idx: target_idx for idx, target_idx in enumerate(chosen_match)}
    mismatches = [row for row in stereo_cmp if not row["equal"]]
    bond_stereo_mm = mapped_bond_stereo_mismatches(mol_i, mol_j, mapping)
    inversion = (
        mismatch_count == 1
        and {mismatches[0]["source_descriptor"], mismatches[0]["target_descriptor"]} == {"R", "S"}
        and not bond_stereo_mm
    )
    if not inversion:
        return {
            "structure_class": STRUCT_UNRESOLVED,
            "reason": "Connectivity matched, but the mapped recorded stereo difference was not exactly one R/S inversion",
            "minimum_stereo_mismatch_count": mismatch_count,
            "mapped_stereo_mismatches": mismatches,
            "mapped_bond_stereo_mismatches": bond_stereo_mm,
        }
    mapped_atoms, atom_mm, mapped_bonds = detailed_original_mapping(mol_i, mol_j, mapping)
    mapped_bond_mm = [row for row in mapped_bonds if not row["base_label_equal"]]
    require(not atom_mm and not mapped_bond_mm, "Stereo-only base atom/bond labels must match exactly")
    require(canonical_smiles(mol_i, True) != canonical_smiles(mol_j, True), "Stereo-only isomeric SMILES must differ")
    return {
        "structure_class": STRUCT_STEREO,
        "reason": None,
        "longer_endpoint_role": None,
        "shorter_endpoint_role": None,
        "unit_formula": None,
        "modified_unit_count": 1,
        "heavy_atom_delta": 0,
        "formula_delta_long_minus_short": "0",
        "chosen_edit": {
            "type": "one_mapped_recorded_tetrahedral_descriptor_inversion",
            "stereocenter_source_atom_index_i": mismatches[0]["source_atom_index"],
            "stereocenter_target_atom_index_j": mismatches[0]["target_atom_index"],
            "descriptor_i": mismatches[0]["source_descriptor"],
            "descriptor_j": mismatches[0]["target_descriptor"],
            "ports": [],
        },
        "full_graph_mapping": {
            "source_role": "i",
            "target_role": "j",
            "mapped_atom_count": len(mapping),
            "removed_atom_count": 0,
            "mapped_atoms": mapped_atoms,
            "mapped_bonds": mapped_bonds,
            "mapped_atom_base_label_mismatch_count": len(atom_mm),
            "mapped_bond_base_label_mismatch_count": len(mapped_bond_mm),
            "canonical_nonisomeric_smiles_i": canonical_smiles(mol_i, False),
            "canonical_nonisomeric_smiles_j": canonical_smiles(mol_j, False),
            "canonical_nonisomeric_smiles_equal": True,
            "canonical_isomeric_smiles_i": canonical_smiles(mol_i, True),
            "canonical_isomeric_smiles_j": canonical_smiles(mol_j, True),
            "canonical_isomeric_smiles_differ": True,
            "connectivity_exact_isomorphism_ignoring_only_stereo": True,
            "mapping_sha256": json_digest(
                [{"i_atom_index": k, "j_atom_index": v} for k, v in sorted(mapping.items())]
            ),
            "mapped_bond_ledger_sha256": json_digest(mapped_bonds),
        },
        "stereo_check": {
            "source_stereo_status_i": endpoint_i["source_stereo_status"],
            "source_stereo_status_j": endpoint_j["source_stereo_status"],
            "centers_i": stereo_center_ledger(mol_i),
            "centers_j": stereo_center_ledger(mol_j),
            "mapped_center_comparison": stereo_cmp,
            "mapped_center_mismatch_count": mismatch_count,
            "mapped_bond_stereo_mismatch_count": len(bond_stereo_mm),
            "recorded_bond_stereo_i": bond_stereo_ledger(mol_i),
            "recorded_bond_stereo_j": bond_stereo_ledger(mol_j),
            "exactly_one_recorded_tetrahedral_descriptor_inversion": True,
        },
        "strict_checks": {
            "all_atom_base_labels_exact": not atom_mm,
            "all_bond_base_labels_exact": not mapped_bond_mm,
            "full_connectivity_exact": True,
            "canonical_nonisomeric_equal": True,
            "canonical_isomeric_different": True,
            "exactly_one_mapped_R_S_descriptor_difference": True,
            "no_mapped_bond_stereo_difference": not bond_stereo_mm,
        },
    }
