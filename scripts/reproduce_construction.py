#!/usr/bin/env python3
"""Rebuild the frozen 874-pair construction from the fixed public TACK table.

This command performs source-table normalization, pair construction, graph
components, five graph-constrained split assignments, and binary Morgan
fingerprints.  It does not download the input, fit a model, or run external
overlap and contamination analyses.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import pandas as pd

from lib.primary.construction_core import (
    audit_pair_metadata,
    build_assignments,
    build_merged_components,
    build_v01_pairs,
    build_v01_records,
    canonicalize_raw,
    fingerprint_csv_bytes,
    records_projection_bytes,
    validate_core_tables,
)


FROZEN_TACK_SHA256 = "2893CE64FE1D5AD50F6BF97AA63FDE9597D11E6A1B388CFE30A7CFC79BB06875"
EXPECTED_OUTPUT_SHA256 = {
    "records_aggregated_v0.1.csv": "DA017991BDD0A83E91F086925986E48C113DECEB1347E90EB64DFF48CD7DEEED",
    "pairs_v0.1.csv": "37F6F70903162E58C41E14DABFC463600F891F7D5830FF895AEE2BFB79C691E7",
    "merged_endpoint_graph_components.csv": "7D43054EB1B3B39CB11708652885B6531D8CE694BAB271BE1F64F87FF3327E1F",
    "pair_split_assignments.csv": "32137D943A2B589312161AA8EF7D839F3C557900DE2D5BA0E2F45541685FCF6A",
    "pair_metadata.csv": "8F9C9D5D7787557357589533C59D253F419366600CAA673E4C51C021A6151FFC",
    "record_morgan_fp_r2_2048.csv": "9B416F6923EAFFAAFB26E82AA689EC7FA05C932D935BFB4C2344B0089AF2D9C5",
    "identity_records_v0.1.parquet": "2E03E8B32E41C01966473F25DA0A225BEB0DE27EAC3C1710C7BA1C937444AB1E",
    "primary_pair_map.csv": "4680A5D5BA5A665F22AD122F5F72EE17C09F6B3702590C32F90DA82536AD71D0",
    "primary_split_membership.csv": "03575DC355836B27F126BE1048A96C3924CF45AA75A6DF60CF068AAAC64CBBB5",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(
        path,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        quoting=csv.QUOTE_MINIMAL,
    )


def write_projection(
    source: Path,
    target: Path,
    columns: list[str],
    *,
    cliffs_only: bool = False,
) -> None:
    with source.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if cliffs_only:
        rows = sorted(
            (row for row in rows if row["is_cliff"].lower() == "true"),
            key=lambda row: row["pair_id"],
        )
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=columns,
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-parquet",
        type=Path,
        required=True,
        help="Fixed 4,184-row TACK DC50 parquet file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory for constructed tables and the run receipt.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.input_parquet.resolve()
    output = args.output_dir.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists():
        raise RuntimeError("output directory already exists; preserve prior attempts")
    observed_source_sha = sha256_file(source)
    if observed_source_sha != FROZEN_TACK_SHA256:
        raise RuntimeError(
            "input parquet is not the fixed byte-matching TACK snapshot"
        )

    output.mkdir(parents=True)
    receipt: dict[str, object] = {
        "status": "STARTED",
        "analysis_id": "portable_frozen_tack_construction",
        "input": {
            "file": source.name,
            "bytes": source.stat().st_size,
            "sha256": observed_source_sha,
        },
        "operation_boundary": {
            "downloads": 0,
            "model_fits": 0,
            "predictions": 0,
            "external_overlap_analyses": 0,
        },
    }
    started = time.perf_counter()
    try:
        raw = pd.read_parquet(source)
        identity_rows, _, fingerprints_by_identity = canonicalize_raw(raw)
        identity_rows.to_parquet(
            output / "identity_records_v0.1.parquet", index=False
        )
        records = build_v01_records(identity_rows)
        pairs = build_v01_pairs(records, fingerprints_by_identity)
        write_csv(output / "records_aggregated_v0.1.csv", records)
        write_csv(output / "pairs_v0.1.csv", pairs)

        records = pd.read_csv(output / "records_aggregated_v0.1.csv")
        pairs = pd.read_csv(output / "pairs_v0.1.csv")
        records, pairs, cliffs = validate_core_tables(records, pairs)
        components, _ = build_merged_components(pairs)
        record_assignments, pair_assignments, fold_support, split_gate = (
            build_assignments(pairs, cliffs, components)
        )
        if split_gate["status"] != "PASS":
            raise RuntimeError(f"graph/split gate failed: {split_gate}")
        metadata, metadata_gate = audit_pair_metadata(records, cliffs)
        for name, frame in (
            ("merged_endpoint_graph_components.csv", components),
            ("pair_split_assignments.csv", pair_assignments),
            ("record_split_assignments.csv", record_assignments),
            ("pair_metadata.csv", metadata),
            ("fold_support.csv", fold_support),
        ):
            write_csv(output / name, frame)

        write_projection(
            output / "pairs_v0.1.csv",
            output / "primary_pair_map.csv",
            [
                "pair_id",
                "comparison_pair_key",
                "record_id_i",
                "record_id_j",
                "normalized_poi_id",
                "pair_context_hash",
                "tanimoto",
                "true_delta",
                "true_abs_delta",
                "is_cliff",
            ],
            cliffs_only=True,
        )
        write_projection(
            output / "pair_split_assignments.csv",
            output / "primary_split_membership.csv",
            [
                "split_seed",
                "heldout_fold",
                "pair_id",
                "comparison_pair_key",
                "component_id",
                "record_id_i",
                "record_id_j",
            ],
        )

        projection = records_projection_bytes(records)
        (output / "records_identity_projection.csv").write_bytes(projection)
        (output / "record_morgan_fp_r2_2048.csv").write_bytes(
            fingerprint_csv_bytes(projection)
        )

        observed = {
            name: sha256_file(output / name) for name in EXPECTED_OUTPUT_SHA256
        }
        mismatches = {
            name: {"expected": EXPECTED_OUTPUT_SHA256[name], "observed": value}
            for name, value in observed.items()
            if value != EXPECTED_OUTPUT_SHA256[name]
        }
        if mismatches:
            raise RuntimeError(
                "constructed files differ from the frozen reference: "
                + ", ".join(sorted(mismatches))
            )
        receipt.update(
            status="PASS_CONSTRUCTION_AND_SPLITS_ONLY",
            counts={
                "raw_rows": len(raw),
                "records": len(records),
                "threshold_0p5_pairs": len(pairs),
                "primary_pairs": len(cliffs),
                "primary_comparison_groups": cliffs[
                    "comparison_pair_key"
                ].nunique(),
                "components": components["component_id"].nunique(),
                "split_memberships": len(pair_assignments),
                "fingerprints": len(records),
            },
            output_sha256=observed,
            split_gate=split_gate,
            metadata_gate=metadata_gate,
            limitations=[
                "The fixed source digest is a byte-matching recovery anchor; it does not recover the original download time or upstream row-level provenance.",
                "This command does not perform PROTACDB/Wurz overlap or contamination analyses.",
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
        with (output / "CONSTRUCTION_RUN.json").open(
            "w", encoding="utf-8", newline="\n"
        ) as stream:
            json.dump(receipt, stream, indent=2, ensure_ascii=False)
            stream.write("\n")


if __name__ == "__main__":
    main()
