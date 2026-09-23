#!/usr/bin/env python3
"""Verify and locally convert the fixed PROTAC-DB workbook used by A2.

The user must obtain ``protac.xlsx`` from PROTAC-DB and accept the provider's
terms outside this script.  This program performs no network request and does
not accept any agreement.  It stops unless the workbook matches the fixed
study hash, then applies the original first-sheet pandas/openpyxl conversion.
Neither the workbook nor the resulting CSV is intended for redistribution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "protacdb-local-prepare-v1"
EXPECTED_XLSX_BYTES = 6_268_531
EXPECTED_XLSX_SHA256 = (
    "4E3A7ECC74A24E26877D319B18937E2A81161A43B4BD1A2C6A1C2BAE0FCB263D"
)
EXPECTED_CSV_ROWS = 15_502
EXPECTED_CSV_COLUMNS = 89
EXPECTED_CSV_SHA256 = (
    "F4601179B471A3B0CCAF10021A2C6C5289AD12CA6B6716233A734A778D26CE28"
)
OUTPUT_NAME = "protac.csv"
RECEIPT_NAME = "protacdb_prepare.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def write_receipt(path: Path, receipt: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def base_receipt(input_path: Path) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "NOT_RUN",
        "network_used": False,
        "agreement_accepted_by_script": False,
        "redistribution_authorized_by_script": False,
        "provider_download_page": "https://cadd.zju.edu.cn/protacdb/downloads",
        "input": {
            "path": input_path.as_posix(),
            "expected_bytes": EXPECTED_XLSX_BYTES,
            "expected_sha256": EXPECTED_XLSX_SHA256,
        },
        "conversion_contract": {
            "worksheet": "first worksheet (pandas read_excel default)",
            "reader": "pandas.read_excel(engine='openpyxl')",
            "writer": "pandas.DataFrame.to_csv(index=False); historical CSV hash required",
            "expected_rows": EXPECTED_CSV_ROWS,
            "expected_columns": EXPECTED_CSV_COLUMNS,
            "expected_csv_sha256": EXPECTED_CSV_SHA256,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-xlsx",
        required=True,
        type=Path,
        help="User-obtained PROTAC-DB protac.xlsx file.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Local directory for protac.csv and the preparation receipt.",
    )
    return parser.parse_args()


def run(input_path: Path, output_dir: Path) -> tuple[int, dict[str, Any]]:
    input_path = input_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = output_dir / RECEIPT_NAME
    receipt = base_receipt(input_path)

    if not input_path.is_file():
        receipt["status"] = "INPUT_NOT_FOUND"
        write_receipt(receipt_path, receipt)
        return 2, receipt

    observed_bytes = input_path.stat().st_size
    observed_sha256 = sha256_file(input_path)
    receipt["input"].update(
        {
            "observed_bytes": observed_bytes,
            "observed_sha256": observed_sha256,
            "fixed_hash_match": observed_sha256 == EXPECTED_XLSX_SHA256,
        }
    )
    if (
        observed_bytes != EXPECTED_XLSX_BYTES
        or observed_sha256 != EXPECTED_XLSX_SHA256
    ):
        receipt["status"] = "INPUT_HASH_MISMATCH_NO_CONVERSION"
        write_receipt(receipt_path, receipt)
        return 2, receipt

    output_path = output_dir / OUTPUT_NAME
    if output_path.exists():
        existing_hash = sha256_file(output_path)
        receipt["existing_output"] = {
            "path": output_path.as_posix(),
            "bytes": output_path.stat().st_size,
            "sha256": existing_hash,
        }
        if existing_hash == EXPECTED_CSV_SHA256:
            receipt["status"] = "PASS_EXISTING_FIXED_CSV"
            write_receipt(receipt_path, receipt)
            return 0, receipt
        receipt["status"] = "OUTPUT_EXISTS_HASH_MISMATCH_NOT_OVERWRITTEN"
        write_receipt(receipt_path, receipt)
        return 2, receipt

    try:
        import openpyxl  # noqa: F401
        import pandas as pd
    except ImportError as exc:
        receipt["status"] = "CONVERSION_DEPENDENCY_MISSING"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        write_receipt(receipt_path, receipt)
        return 2, receipt

    receipt["runtime"] = {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "openpyxl": openpyxl.__version__,
    }
    temporary = output_dir / (OUTPUT_NAME + ".tmp")
    if temporary.exists():
        receipt["status"] = "TEMPORARY_OUTPUT_EXISTS_NOT_OVERWRITTEN"
        write_receipt(receipt_path, receipt)
        return 2, receipt

    try:
        frame = pd.read_excel(input_path, engine="openpyxl")
    except Exception as exc:
        receipt["status"] = "XLSX_READ_FAILED"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        write_receipt(receipt_path, receipt)
        return 2, receipt

    rows, columns = frame.shape
    receipt["conversion"] = {
        "observed_rows": int(rows),
        "observed_columns": int(columns),
    }
    if rows != EXPECTED_CSV_ROWS or columns != EXPECTED_CSV_COLUMNS:
        receipt["status"] = "WORKBOOK_SHAPE_MISMATCH_NO_CSV"
        write_receipt(receipt_path, receipt)
        return 2, receipt

    try:
        # Keep the historical converter call unchanged.  The fixed output hash
        # is the authority for platform/newline or pandas-formatting drift.
        frame.to_csv(temporary, index=False)
        observed_csv_hash = sha256_file(temporary)
        receipt["conversion"].update(
            {
                "observed_csv_bytes": temporary.stat().st_size,
                "observed_csv_sha256": observed_csv_hash,
                "fixed_csv_hash_match": observed_csv_hash == EXPECTED_CSV_SHA256,
            }
        )
        if observed_csv_hash != EXPECTED_CSV_SHA256:
            temporary.unlink()
            receipt["status"] = "CSV_HASH_MISMATCH_OUTPUT_REMOVED"
            write_receipt(receipt_path, receipt)
            return 2, receipt
        temporary.replace(output_path)
    except Exception as exc:
        if temporary.exists():
            temporary.unlink()
        receipt["status"] = "CSV_CONVERSION_FAILED"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        write_receipt(receipt_path, receipt)
        return 2, receipt

    receipt["status"] = "PASS_FIXED_PROTACDB_CSV_PREPARED_LOCALLY"
    receipt["output"] = {
        "path": output_path.as_posix(),
        "bytes": output_path.stat().st_size,
        "sha256": sha256_file(output_path),
    }
    write_receipt(receipt_path, receipt)
    return 0, receipt


def main() -> int:
    args = parse_args()
    code, receipt = run(args.input_xlsx, args.output_dir)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
