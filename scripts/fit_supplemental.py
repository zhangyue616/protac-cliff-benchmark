#!/usr/bin/env python3
"""Reproduce the fixed 200-fit E1 and 175-fit E2 prediction generation.

The command reconstructs j-minus-i targets from the supplied construction
directory, validates the fixed author assignments and fit-cell schedule, and
writes only beneath an explicit output directory.  It creates the saved E1/E2
OOF predictions and the two pre-specified component draw plans.  Statistical
scoring remains a separate step in run_supplemental_e1.py and
run_supplemental_e2.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lib.supplemental_fitting import run_refit, validate_only
from lib.supplemental_protocol import SupplementalError, load_protocol_inputs


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--construction-dir",
        required=True,
        type=Path,
        help=(
            "Directory containing records_aggregated_v0.1.csv, primary_pair_map.csv, "
            "record_morgan_fp_r2_2048.csv, primary_split_membership.csv, and "
            "merged_endpoint_graph_components.csv."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Absent, empty, or identically bound resumable output directory.",
    )
    parser.add_argument(
        "--e1-assignments",
        type=Path,
        default=REPOSITORY_ROOT / "config" / "supplemental_e1_assignments.csv",
        help="Fixed author-generated E1 pair/fold/component assignments.",
    )
    parser.add_argument(
        "--e1-components",
        type=Path,
        default=REPOSITORY_ROOT / "config" / "supplemental_e1_components.csv",
        help="Lexically ordered 55-unit E1 component container, including 26 empty units.",
    )
    parser.add_argument(
        "--fit-plan",
        type=Path,
        default=REPOSITORY_ROOT / "scripts" / "lib" / "primary" / "fit_cell_plan.csv",
        help="Original 200-row model/seed/fold fit-cell plan.",
    )
    parser.add_argument(
        "--permutation-seeds",
        type=Path,
        default=(
            REPOSITORY_ROOT / "scripts" / "lib" / "primary" / "permutation_seed_table.csv"
        ),
        help="Original 25-row target-permutation seed table.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        choices=(4,),
        default=4,
        help="Fixed model/thread limit; protocol A requires 4.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Validate inputs, memberships, target orientation, and fingerprint gates "
            "without creating the output directory or fitting a model."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        inputs = load_protocol_inputs(
            args.construction_dir,
            e1_assignments_path=args.e1_assignments,
            e1_components_path=args.e1_components,
            fit_plan_path=args.fit_plan,
            permutation_seeds_path=args.permutation_seeds,
        )
        receipt = validate_only(inputs) if args.validate_only else run_refit(
            inputs, args.output_dir, n_jobs=args.n_jobs
        )
        print(json.dumps(receipt, ensure_ascii=False, indent=2))
        return 0
    except (SupplementalError, OSError, ValueError, ImportError) as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
