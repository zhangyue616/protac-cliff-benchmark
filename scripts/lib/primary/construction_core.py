"""Portable scientific functions for the frozen TACK construction protocol."""
from __future__ import annotations

import csv
import hashlib
import io
import math
import unicodedata
from collections import defaultdict
from typing import Any, Iterable

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold


RDLogger.DisableLog("rdApp.*")

CONTRACT_ID = "route-b-b0-contract-v0.1-20260801"
SPLIT_SEEDS = (20260624, 20260625, 20260626, 20260724, 20260801)
FOLD_COUNT = 5
EXPECTED_PAIRS = 874
EXPECTED_KEYS = 838
EXPECTED_RAW_ROWS = 4184
EXPECTED_RECORDS = 3302
EXPECTED_PRIMARY_RECORDS = 731
EXPECTED_EDGE_PAIRS = 6015
EXPECTED_COMPONENT_RECORDS = 629


class ConstructionError(RuntimeError):
    """Raised when a frozen construction invariant is violated."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def normalize_text(value: object, missing: str) -> str:
    if pd.isna(value):
        return missing
    text = unicodedata.normalize("NFKC", str(value))
    text = " ".join(text.split()).strip()
    return text.casefold() if text else missing


def normalize_identifier(value: object, missing: str) -> str:
    text = normalize_text(value, missing)
    return text.upper() if text != missing else missing


def pipe_unique(values: pd.Series) -> str:
    clean = sorted(
        {str(value) for value in values if pd.notna(value) and str(value) != ""}
    )
    return "|".join(clean)


def pdc50_from_nm(value: object) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    if not math.isfinite(number) or number <= 0:
        return None
    return 9.0 - math.log10(number)


def exact_operator(value: object) -> bool:
    if pd.isna(value):
        return True
    return str(value).strip() in {"", "="}


def mol_to_fp(mol: Chem.Mol):
    return AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=2, nBits=2048, useChirality=False
    )


def murcko_smiles(mol: Chem.Mol) -> str:
    try:
        value = MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=False
        )
        return value if value else "NO_SCAFFOLD"
    except Exception:
        return "MURCKO_ERROR"


def require_raw_schema(raw: pd.DataFrame) -> None:
    required = {
        "SMILES",
        "POI_Name",
        "POI_UniProt",
        "Ligase_Name",
        "Ligase_UniProt",
        "Cell_Line",
        "Cell_Line_ID",
        "Value_Type",
        "Value_Unit",
        "Value",
        "Value_Operator",
        "Reference",
        "Database",
        "Assay",
    }
    missing = required - set(raw.columns)
    if missing:
        raise ConstructionError(f"raw TACK table is missing {sorted(missing)}")
    if len(raw) != EXPECTED_RAW_ROWS:
        raise ConstructionError(
            f"raw TACK row count is {len(raw)}; expected {EXPECTED_RAW_ROWS}"
        )


def canonicalize_raw(
    raw: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Chem.Mol], dict[str, object]]:
    require_raw_schema(raw)
    frame = raw.copy().reset_index(drop=True)
    frame["source_row_id"] = [f"tack_row_{idx:06d}" for idx in range(len(frame))]
    frame["poi_name_norm"] = frame["POI_Name"].map(
        lambda value: normalize_text(value, "MISSING_POI_NAME")
    )
    frame["poi_uniprot_norm"] = frame["POI_UniProt"].map(
        lambda value: normalize_identifier(value, "MISSING_POI_UNIPROT")
    )
    name_to_uniprots = frame.groupby("poi_name_norm")["poi_uniprot_norm"].agg(
        lambda values: sorted(set(values))
    )
    uniprot_to_names = frame.groupby("poi_uniprot_norm")["poi_name_norm"].agg(
        lambda values: sorted(set(values))
    )

    mol_by_identity: dict[str, Chem.Mol] = {}
    fp_by_identity: dict[str, object] = {}
    rows: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        raw_smiles = str(row.SMILES) if pd.notna(row.SMILES) else ""
        try:
            mol = Chem.MolFromSmiles(raw_smiles)
        except Exception:
            mol = None
        if mol is None:
            canonical = ""
            nonisomeric = ""
            identity_hash = ""
            fragment_count = 0
            formal_charge = 0
            stereo_status = "not_parsed"
            standardization_status = "invalid_smiles"
            scaffold = "INVALID"
        else:
            canonical = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
            nonisomeric = Chem.MolToSmiles(
                mol, isomericSmiles=False, canonical=True
            )
            identity_hash = sha256_text(canonical)
            fragment_count = len(Chem.GetMolFrags(mol))
            formal_charge = int(
                sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
            )
            centers = Chem.FindMolChiralCenters(
                mol, includeUnassigned=True, useLegacyImplementation=False
            )
            assigned = sum(label != "?" for _, label in centers)
            unassigned = sum(label == "?" for _, label in centers)
            stereo_status = (
                f"tetra_assigned_{assigned}_unassigned_{unassigned}_"
                f"isomeric_diff_{int(canonical != nonisomeric)}"
            )
            standardization_status = "parsed_full_graph_no_fragment_stripping"
            scaffold = murcko_smiles(mol)
            mol_by_identity.setdefault(identity_hash, mol)
            fp_by_identity.setdefault(identity_hash, mol_to_fp(mol))

        name_values = name_to_uniprots.loc[row.poi_name_norm]
        uniprot_values = uniprot_to_names.loc[row.poi_uniprot_norm]
        mapping_unique = (
            row.poi_name_norm != "MISSING_POI_NAME"
            and row.poi_uniprot_norm != "MISSING_POI_UNIPROT"
            and len(name_values) == 1
            and len(uniprot_values) == 1
        )
        mapping_status = (
            "source_unique_bidirectional"
            if mapping_unique
            else "ambiguous_or_missing_alias"
        )
        normalized_poi = (
            f"UNIPROT:{row.poi_uniprot_norm}"
            if mapping_unique
            else "UNRESOLVED:"
            + sha256_text(f"{row.poi_name_norm}|{row.poi_uniprot_norm}")[:20]
        )
        source_dataset = normalize_text(row.Database, "MISSING_SOURCE_DATASET")
        reference_text = normalize_text(row.Reference, "MISSING_ARTICLE_TEXT")
        article_id = (
            "MISSING_ARTICLE_ID"
            if reference_text == "MISSING_ARTICLE_TEXT"
            else "RAWREF:" + sha256_text(reference_text)
        )
        assay_text = normalize_text(row.Assay, "MISSING_ASSAY_TEXT")
        assay_id = (
            "MISSING_ASSAY_ID"
            if assay_text == "MISSING_ASSAY_TEXT"
            else "RAWASSAY:" + sha256_text(assay_text)
        )
        cell_id_source = normalize_identifier(row.Cell_Line_ID, "MISSING_CELL_ID")
        cell_text = normalize_text(row.Cell_Line, "MISSING_CELL_TEXT")
        if cell_id_source != "MISSING_CELL_ID":
            cell_id = "CELL_ID:" + cell_id_source
        elif cell_text != "MISSING_CELL_TEXT":
            cell_id = "RAWCELL:" + sha256_text(cell_text)
        else:
            cell_id = "MISSING_CELL_ID_OR_TEXT"
        e3_uniprot = normalize_identifier(
            row.Ligase_UniProt, "MISSING_E3_UNIPROT"
        )
        e3_name = normalize_text(row.Ligase_Name, "MISSING_E3_NAME")
        if e3_uniprot != "MISSING_E3_UNIPROT":
            normalized_e3 = "UNIPROT:" + e3_uniprot
        elif e3_name != "MISSING_E3_NAME":
            normalized_e3 = "RAWE3:" + sha256_text(e3_name)
        else:
            normalized_e3 = "MISSING_E3_ID_OR_TEXT"
        pair_context_hash = sha256_text(
            "|".join([source_dataset, article_id, assay_id, cell_id, normalized_e3])
        )
        context_hash = sha256_text(
            "|".join(
                [identity_hash or "INVALID_IDENTITY", normalized_poi, pair_context_hash]
            )
        )

        reasons: list[str] = []
        if str(row.Value_Type).strip().upper() != "DC50":
            reasons.append("endpoint_not_dc50")
        if str(row.Value_Unit).strip().lower() != "nm":
            reasons.append("unit_not_nm")
        if not exact_operator(row.Value_Operator):
            reasons.append("operator_not_exact")
        pdc50 = pdc50_from_nm(row.Value)
        if pdc50 is None:
            reasons.append("value_not_positive_finite")
        if mol is None:
            reasons.append("invalid_smiles")
        exact_eligible = not reasons
        rows.append(
            {
                "source_row_id": row.source_row_id,
                "raw_smiles": raw_smiles,
                "raw_smiles_hash": sha256_text(raw_smiles),
                "canonical_isomeric_smiles_full": canonical,
                "canonical_nonisomeric_smiles_full_sensitivity": nonisomeric,
                "identity_hash": identity_hash,
                "fragment_count": fragment_count,
                "has_formal_charge": bool(formal_charge != 0),
                "formal_charge_sum": formal_charge,
                "stereo_status": stereo_status,
                "standardization_status": standardization_status,
                "murcko_scaffold": scaffold,
                "poi_name_raw": row.POI_Name,
                "poi_name_norm": row.poi_name_norm,
                "poi_uniprot_raw": row.POI_UniProt,
                "normalized_poi_id": normalized_poi,
                "poi_mapping_status": mapping_status,
                "source_dataset": source_dataset,
                "article_id": article_id,
                "article_semantic_status": "raw_reference_text_hash_not_semantically_verified",
                "article_text_normalized": reference_text,
                "assay_id_or_text": assay_id,
                "assay_semantic_status": "raw_assay_text_hash_not_ontology_verified",
                "assay_text_normalized": assay_text,
                "cell_line_id_or_text": cell_id,
                "cell_semantic_status": "source_id_or_raw_text_not_ontology_verified",
                "normalized_e3_id": normalized_e3,
                "e3_semantic_status": "source_identifier_not_independently_curated",
                "pair_context_hash": pair_context_hash,
                "context_hash": context_hash,
                "value_type_raw": row.Value_Type,
                "value_unit_raw": row.Value_Unit,
                "value_operator_raw": row.Value_Operator,
                "dc50_nm": row.Value,
                "pdc50": pdc50,
                "exact_endpoint_eligible": exact_eligible,
                "primary_pair_row_eligible": bool(exact_eligible and mapping_unique),
                "exclusion_reason": "|".join(reasons),
            }
        )
    return pd.DataFrame(rows), mol_by_identity, fp_by_identity


def build_v01_records(identity_rows: pd.DataFrame) -> pd.DataFrame:
    eligible = identity_rows[identity_rows["exact_endpoint_eligible"]].copy()
    group_cols = [
        "identity_hash",
        "normalized_poi_id",
        "pair_context_hash",
        "context_hash",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in eligible.groupby(group_cols, sort=True, dropna=False):
        identity_hash, normalized_poi, pair_context_hash, context_hash = keys
        values = group["pdc50"].astype(float).to_numpy()
        pdc50_range = float(values.max() - values.min())
        high = pdc50_range >= 1.0
        low = pdc50_range >= 0.5
        record_id = "rbv01_" + sha256_text(
            f"{CONTRACT_ID}|{identity_hash}|{normalized_poi}|{pair_context_hash}"
        )
        rows.append(
            {
                "record_id": record_id,
                "source_row_ids": ";".join(
                    sorted(group["source_row_id"].astype(str))
                ),
                "raw_smiles_values": pipe_unique(group["raw_smiles"]),
                "canonical_isomeric_smiles_full": group[
                    "canonical_isomeric_smiles_full"
                ].iloc[0],
                "identity_hash": identity_hash,
                "normalized_poi_id": normalized_poi,
                "poi_mapping_status": pipe_unique(group["poi_mapping_status"]),
                "poi_name_values": pipe_unique(group["poi_name_raw"]),
                "source_dataset": pipe_unique(group["source_dataset"]),
                "article_id": pipe_unique(group["article_id"]),
                "assay_id_or_text": pipe_unique(group["assay_id_or_text"]),
                "cell_line_id_or_text": pipe_unique(
                    group["cell_line_id_or_text"]
                ),
                "normalized_e3_id": pipe_unique(group["normalized_e3_id"]),
                "pair_context_hash": pair_context_hash,
                "context_hash": context_hash,
                "n_measurements": len(group),
                "raw_dc50_values_nm": ";".join(
                    format(float(value), ".12g") for value in group["dc50_nm"]
                ),
                "raw_operators": pipe_unique(group["value_operator_raw"]),
                "pdc50_min": float(np.min(values)),
                "pdc50_max": float(np.max(values)),
                "pdc50_median": float(np.median(values)),
                "pdc50_iqr": float(
                    np.percentile(values, 75) - np.percentile(values, 25)
                ),
                "pdc50_range": pdc50_range,
                "discordance_flag_low": low,
                "discordance_flag_high": high,
                "primary_label_eligible": bool(not high),
                "primary_pair_eligible": bool(
                    not high
                    and set(group["poi_mapping_status"])
                    == {"source_unique_bidirectional"}
                ),
                "murcko_scaffold": group["murcko_scaffold"].iloc[0],
                "fragment_count": int(group["fragment_count"].iloc[0]),
                "has_formal_charge": bool(group["has_formal_charge"].iloc[0]),
                "stereo_status": group["stereo_status"].iloc[0],
                "standardization_status": group["standardization_status"].iloc[0],
                "record_exclusion_reason": "discordance_high" if high else "",
            }
        )
    records = pd.DataFrame(rows)
    return records.sort_values(
        ["normalized_poi_id", "identity_hash", "context_hash", "source_row_ids"],
        kind="mergesort",
    ).reset_index(drop=True)


def build_v01_pairs(
    records: pd.DataFrame, fp_by_identity: dict[str, object]
) -> pd.DataFrame:
    use = records[records["primary_pair_eligible"]].copy().reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for _, group in use.groupby(
        ["normalized_poi_id", "pair_context_hash"], sort=True
    ):
        indices = list(group.index)
        for position, idx_i in enumerate(indices[:-1]):
            right = indices[position + 1 :]
            fp_i = fp_by_identity[use.at[idx_i, "identity_hash"]]
            similarities = DataStructs.BulkTanimotoSimilarity(
                fp_i,
                [fp_by_identity[use.at[idx_j, "identity_hash"]] for idx_j in right],
            )
            for idx_j, similarity_value in zip(right, similarities):
                if (
                    use.at[idx_i, "identity_hash"]
                    == use.at[idx_j, "identity_hash"]
                    or similarity_value < 0.5
                ):
                    continue
                left_id = use.at[idx_i, "record_id"]
                right_id = use.at[idx_j, "record_id"]
                delta = float(
                    use.at[idx_i, "pdc50_median"]
                    - use.at[idx_j, "pdc50_median"]
                )
                pair_id = sha256_text(f"{CONTRACT_ID}|{left_id}|{right_id}")
                comparison_key = sha256_text(
                    "|".join(
                        [
                            use.at[idx_i, "normalized_poi_id"],
                            *sorted(
                                [
                                    use.at[idx_i, "identity_hash"],
                                    use.at[idx_j, "identity_hash"],
                                ]
                            ),
                        ]
                    )
                )
                rows.append(
                    {
                        "pair_id": pair_id,
                        "comparison_pair_key": comparison_key,
                        "record_id_i": left_id,
                        "record_id_j": right_id,
                        "identity_hash_i": use.at[idx_i, "identity_hash"],
                        "identity_hash_j": use.at[idx_j, "identity_hash"],
                        "normalized_poi_id": use.at[idx_i, "normalized_poi_id"],
                        "pair_context_hash": use.at[idx_i, "pair_context_hash"],
                        "orientation_rule": "outcome_blind_sorted_record_order_i_before_j",
                        "tanimoto": float(similarity_value),
                        "true_delta": delta,
                        "comparison_oriented_true_delta": (
                            delta
                            if use.at[idx_i, "identity_hash"]
                            <= use.at[idx_j, "identity_hash"]
                            else -delta
                        ),
                        "true_abs_delta": abs(delta),
                        "is_cliff": bool(abs(delta) >= 1.0),
                        "threshold_0p5": True,
                        "threshold_0p6": bool(similarity_value >= 0.6),
                        "threshold_0p7": bool(similarity_value >= 0.7),
                    }
                )
    return pd.DataFrame(rows)


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "pass", "yes"])


def validate_core_tables(
    records: pd.DataFrame, pairs: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cliffs = pairs[bool_series(pairs["is_cliff"])].copy()
    primary = records[bool_series(records["primary_pair_eligible"])].copy()
    if len(pairs) != EXPECTED_EDGE_PAIRS or not bool_series(
        pairs["threshold_0p5"]
    ).all():
        raise ConstructionError("threshold-0.5 training-edge universe drift")
    if len(records) != EXPECTED_RECORDS or len(primary) != EXPECTED_PRIMARY_RECORDS:
        raise ConstructionError("record universe drift")
    if (
        len(cliffs) != EXPECTED_PAIRS
        or cliffs["pair_id"].nunique() != EXPECTED_PAIRS
        or cliffs["comparison_pair_key"].nunique() != EXPECTED_KEYS
        or cliffs["pair_id"].duplicated().any()
        or not np.isfinite(cliffs["true_delta"].astype(float)).all()
        or cliffs["orientation_rule"].isna().any()
    ):
        raise ConstructionError("874-pair/838-key cliff universe drift")
    endpoints = set(pairs["record_id_i"].astype(str)) | set(
        pairs["record_id_j"].astype(str)
    )
    primary_ids = set(primary["record_id"].astype(str))
    cliff_endpoints = set(cliffs["record_id_i"].astype(str)) | set(
        cliffs["record_id_j"].astype(str)
    )
    if not endpoints.issubset(primary_ids) or not cliff_endpoints.issubset(endpoints):
        raise ConstructionError("pair endpoints escape the primary-record universe")
    if len(endpoints) != EXPECTED_COMPONENT_RECORDS:
        raise ConstructionError("threshold-0.5 endpoint count drift")
    return (
        records,
        pairs,
        cliffs.sort_values("pair_id", kind="mergesort").reset_index(drop=True),
    )


def audit_pair_metadata(
    records: pd.DataFrame, cliffs: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    fields = [
        "normalized_poi_id",
        "poi_name_values",
        "pair_context_hash",
        "source_dataset",
        "article_id",
        "assay_id_or_text",
        "cell_line_id_or_text",
        "normalized_e3_id",
    ]
    left = records[["record_id", *fields]].rename(
        columns={"record_id": "record_id_i", **{name: f"{name}_i" for name in fields}}
    )
    right = records[["record_id", *fields]].rename(
        columns={"record_id": "record_id_j", **{name: f"{name}_j" for name in fields}}
    )
    merged = cliffs.merge(left, on="record_id_i", how="left", validate="many_to_one")
    merged = merged.merge(
        right, on="record_id_j", how="left", validate="many_to_one"
    )
    checks: dict[str, Any] = {}
    for name in fields:
        mismatch = int(
            (merged[f"{name}_i"].astype(str) != merged[f"{name}_j"].astype(str)).sum()
        )
        checks[name] = {
            "endpoint_i_j_mismatch": mismatch,
            "pair_vs_i_mismatch": int(
                (merged[name].astype(str) != merged[f"{name}_i"].astype(str)).sum()
            )
            if name in cliffs.columns
            else None,
        }
    if len(merged) != EXPECTED_PAIRS or any(
        item["endpoint_i_j_mismatch"] != 0 for item in checks.values()
    ):
        raise ConstructionError("pair metadata does not map exactly to endpoints")
    for name in ("normalized_poi_id", "pair_context_hash"):
        if checks[name]["pair_vs_i_mismatch"] != 0:
            raise ConstructionError(f"pair-level {name} disagrees with endpoints")
    metadata = merged[
        ["pair_id", "comparison_pair_key", *[f"{name}_i" for name in fields]]
    ].copy()
    metadata = metadata.rename(columns={f"{name}_i": name for name in fields})
    return metadata, {"status": "PASS", "rows": len(metadata), "field_checks": checks}


class UnionFind:
    def __init__(self, values: Iterable[str]):
        self.parent = {value: value for value in values}
        self.rank = {value: 0 for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        root_left, root_right = self.find(left), self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


def build_merged_components(
    pairs: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str]]:
    edges = pairs[bool_series(pairs["threshold_0p5"])].copy()
    nodes = sorted(
        set(edges["record_id_i"].astype(str))
        | set(edges["record_id_j"].astype(str))
    )
    union = UnionFind(nodes)
    for row in edges[["record_id_i", "record_id_j"]].itertuples(index=False):
        union.union(str(row.record_id_i), str(row.record_id_j))
    cliffs = edges[bool_series(edges["is_cliff"])].copy()
    duplicate_keys = set(
        cliffs["comparison_pair_key"].value_counts().loc[lambda count: count > 1].index
    )
    if len(duplicate_keys) != 36:
        raise ConstructionError("duplicated primary-cliff key count drift")
    for _, frame in cliffs[
        cliffs["comparison_pair_key"].isin(duplicate_keys)
    ].groupby("comparison_pair_key", sort=True):
        members = sorted(
            set(frame["record_id_i"].astype(str))
            | set(frame["record_id_j"].astype(str))
        )
        for member in members[1:]:
            union.union(members[0], member)
    groups: dict[str, list[str]] = defaultdict(list)
    for node in nodes:
        groups[union.find(node)].append(node)
    component_map: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    for members in groups.values():
        component_id = min(members)
        for record_id in members:
            component_map[record_id] = component_id
            rows.append(
                {
                    "record_id": record_id,
                    "component_id": component_id,
                    "component_record_count": len(members),
                }
            )
    frame = pd.DataFrame(rows).sort_values(
        ["component_id", "record_id"], kind="mergesort"
    )
    return frame.reset_index(drop=True), component_map


def build_assignments(
    pairs: pd.DataFrame, cliffs: pd.DataFrame, components: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    component_sizes = components[
        ["component_id", "component_record_count"]
    ].drop_duplicates()
    record_rows: list[dict[str, Any]] = []
    pair_frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    edges = pairs[bool_series(pairs["threshold_0p5"])].copy()
    all_pair_ids = set(cliffs["pair_id"])
    all_keys = set(cliffs["comparison_pair_key"])
    for split_seed in SPLIT_SEEDS:
        ordered = component_sizes.copy()
        ordered["assignment_hash"] = ordered["component_id"].map(
            lambda component_id: sha256_text(
                f"route-b-d2-v01|{split_seed}|{component_id}"
            )
        )
        ordered = ordered.sort_values(
            ["component_record_count", "assignment_hash", "component_id"],
            ascending=[False, True, True],
            kind="mergesort",
        )
        fold_sizes = [0] * FOLD_COUNT
        component_fold: dict[str, int] = {}
        for row in ordered.itertuples(index=False):
            fold = min(range(FOLD_COUNT), key=lambda index: (fold_sizes[index], index))
            component_fold[row.component_id] = fold
            fold_sizes[fold] += int(row.component_record_count)
        assignment = components.copy()
        assignment["split_seed"] = split_seed
        assignment["fold"] = assignment["component_id"].map(component_fold)
        assignment["assignment_hash"] = assignment["component_id"].map(
            ordered.set_index("component_id")["assignment_hash"].to_dict()
        )
        record_rows.extend(assignment.to_dict(orient="records"))
        fold_map = assignment.set_index("record_id")["fold"].to_dict()
        component_map = assignment.set_index("record_id")["component_id"].to_dict()
        assigned = cliffs.copy()
        assigned["split_seed"] = split_seed
        assigned["fold_i"] = assigned["record_id_i"].map(fold_map)
        assigned["fold_j"] = assigned["record_id_j"].map(fold_map)
        assigned["component_id_i"] = assigned["record_id_i"].map(component_map)
        assigned["component_id_j"] = assigned["record_id_j"].map(component_map)
        assigned["heldout_fold"] = assigned["fold_i"]
        if not (
            assigned["fold_i"].eq(assigned["fold_j"]).all()
            and assigned["component_id_i"].eq(assigned["component_id_j"]).all()
        ):
            raise ConstructionError(f"endpoint/component crossing for {split_seed}")
        if not assigned.groupby("comparison_pair_key")["heldout_fold"].nunique().eq(1).all():
            raise ConstructionError(f"comparison-key fold crossing for {split_seed}")
        pair_frames.append(
            assigned[
                [
                    "split_seed",
                    "pair_id",
                    "comparison_pair_key",
                    "record_id_i",
                    "record_id_j",
                    "component_id_i",
                    "heldout_fold",
                ]
            ].rename(columns={"component_id_i": "component_id"})
        )
        for fold in range(FOLD_COUNT):
            test_nodes = set(
                assignment.loc[assignment["fold"] == fold, "record_id"].astype(str)
            )
            test_components = set(
                assignment.loc[assignment["fold"] == fold, "component_id"].astype(str)
            )
            train_before_key_gate = edges[
                ~edges["record_id_i"].isin(test_nodes)
                & ~edges["record_id_j"].isin(test_nodes)
            ].copy()
            test = assigned[assigned["heldout_fold"] == fold].copy()
            test_keys = set(test["comparison_pair_key"].astype(str))
            train = train_before_key_gate[
                ~train_before_key_gate["comparison_pair_key"].isin(test_keys)
            ].copy()
            train_endpoints = set(train["record_id_i"].astype(str)) | set(
                train["record_id_j"].astype(str)
            )
            test_endpoints = set(test["record_id_i"].astype(str)) | set(
                test["record_id_j"].astype(str)
            )
            train_components = {component_map[value] for value in train_endpoints}
            fold_rows.append(
                {
                    "split_seed": split_seed,
                    "fold": fold,
                    "test_records": len(test_nodes),
                    "test_components": len(test_components),
                    "training_edge_universe": len(edges),
                    "train_pairs_before_comparison_key_gate": len(train_before_key_gate),
                    "comparison_key_excluded_train_pairs": len(train_before_key_gate)
                    - len(train),
                    "train_pairs": len(train),
                    "test_pair_ids": test["pair_id"].nunique(),
                    "test_comparison_keys": test["comparison_pair_key"].nunique(),
                    "endpoint_crossings": len(train_endpoints & test_endpoints),
                    "component_crossings": len(train_components & test_components),
                    "comparison_key_crossings": len(
                        set(train["comparison_pair_key"].astype(str))
                        & set(test["comparison_pair_key"].astype(str))
                    ),
                    "minimum_train_pairs_pass": len(train) >= 500,
                    "minimum_test_pair_ids_pass": test["pair_id"].nunique() >= 20,
                }
            )
        if set(assigned["pair_id"]) != all_pair_ids or set(
            assigned["comparison_pair_key"]
        ) != all_keys:
            raise ConstructionError(f"coverage drift for split {split_seed}")
    assignments = pd.DataFrame(record_rows).sort_values(
        ["split_seed", "fold", "component_id", "record_id"], kind="mergesort"
    )
    pair_assignments = pd.concat(pair_frames, ignore_index=True).sort_values(
        ["split_seed", "heldout_fold", "pair_id"], kind="mergesort"
    )
    folds = pd.DataFrame(fold_rows)
    gate = {
        "status": "PASS",
        "merged_endpoint_graph_components": int(components["component_id"].nunique()),
        "records": int(components["record_id"].nunique()),
        "split_assignments": len(SPLIT_SEEDS),
        "folds_per_assignment": FOLD_COUNT,
        "pair_id_coverage_each_assignment": {
            str(seed): int(
                pair_assignments.loc[
                    pair_assignments["split_seed"] == seed, "pair_id"
                ].nunique()
            )
            for seed in SPLIT_SEEDS
        },
        "comparison_key_coverage_each_assignment": {
            str(seed): int(
                pair_assignments.loc[
                    pair_assignments["split_seed"] == seed, "comparison_pair_key"
                ].nunique()
            )
            for seed in SPLIT_SEEDS
        },
        "endpoint_crossings": int(folds["endpoint_crossings"].sum()),
        "component_crossings": int(folds["component_crossings"].sum()),
        "comparison_key_crossings": int(folds["comparison_key_crossings"].sum()),
        "nonempty_folds": int((folds["test_pair_ids"] > 0).sum()),
        "minimum_training_pairs_all_folds": bool(
            folds["minimum_train_pairs_pass"].all()
        ),
        "minimum_test_pair_ids_all_folds": bool(
            folds["minimum_test_pair_ids_pass"].all()
        ),
    }
    if not all(
        [
            gate["records"] == EXPECTED_COMPONENT_RECORDS,
            gate["merged_endpoint_graph_components"] == 77,
            gate["split_assignments"] == 5,
            gate["nonempty_folds"] == 25,
            set(gate["pair_id_coverage_each_assignment"].values())
            == {EXPECTED_PAIRS},
            set(gate["comparison_key_coverage_each_assignment"].values())
            == {EXPECTED_KEYS},
            gate["endpoint_crossings"] == 0,
            gate["component_crossings"] == 0,
            gate["comparison_key_crossings"] == 0,
            gate["minimum_training_pairs_all_folds"],
            gate["minimum_test_pair_ids_all_folds"],
        ]
    ):
        gate["status"] = "FAIL_STRUCTURE_GATE"
    return (
        assignments.reset_index(drop=True),
        pair_assignments.reset_index(drop=True),
        folds,
        gate,
    )


def records_projection_bytes(records: pd.DataFrame) -> bytes:
    required = {"record_id", "canonical_isomeric_smiles_full"}
    if not required.issubset(records.columns):
        raise ConstructionError("records table cannot produce fingerprint projection")
    frame = records[["record_id", "canonical_isomeric_smiles_full"]].copy()
    frame = frame.sort_values("record_id", kind="mergesort")
    if (
        len(frame) != EXPECTED_RECORDS
        or frame["record_id"].duplicated().any()
        or frame.astype(str).apply(lambda column: column.str.strip().eq("")).any().any()
    ):
        raise ConstructionError("fingerprint projection universe is invalid")
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    writer.writerow(("record_id", "canonical_isomeric_smiles_full"))
    writer.writerows(frame.itertuples(index=False, name=None))
    return buffer.getvalue().encode("utf-8")


def _bits_to_integer(bit_vector: Any) -> int:
    value = 0
    for index in bit_vector.GetOnBits():
        value |= 1 << index
    return value


def fingerprint_csv_bytes(projection: bytes) -> bytes:
    reader = csv.DictReader(io.StringIO(projection.decode("utf-8"), newline=""))
    if tuple(reader.fieldnames or ()) != (
        "record_id",
        "canonical_isomeric_smiles_full",
    ):
        raise ConstructionError("fingerprint projection header is invalid")
    rows: list[tuple[str, str, str, int, str]] = []
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=False
    )
    for record in reader:
        record_id = record["record_id"]
        smiles = record["canonical_isomeric_smiles_full"]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ConstructionError(f"cannot parse constructed SMILES for {record_id}")
        legacy = AllChem.GetMorganFingerprintAsBitVect(
            mol, 2, nBits=2048, useChirality=False
        )
        modern = generator.GetFingerprint(mol)
        legacy_value = _bits_to_integer(legacy)
        modern_value = _bits_to_integer(modern)
        if legacy_value != modern_value:
            raise ConstructionError(f"RDKit fingerprint APIs disagree for {record_id}")
        fp_hex = format(legacy_value, "0512X")
        smiles_sha = sha256_text(smiles)
        popcount = legacy_value.bit_count()
        row_digest = sha256_text(
            "\t".join((record_id, smiles_sha, fp_hex, str(popcount)))
        )
        rows.append((record_id, smiles_sha, fp_hex, popcount, row_digest))
    if len(rows) != EXPECTED_RECORDS:
        raise ConstructionError("fingerprint row count drift")
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    writer.writerow(("record_id", "smiles_sha256", "fp_hex", "popcount", "row_digest"))
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")
