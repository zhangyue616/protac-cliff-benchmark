#!/usr/bin/env python3
"""Reproduce the TACK-based analyses with cached predictions or new fixed-recipe fits.

The cached path rebuilds records, pairs, folds and targets, then scores the
released author predictions. The refit path additionally reruns the original
200 primary and 375 supplemental fits. Neither path reconstructs unavailable
PROTAC-DB historical anchors or unpreserved original provenance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cached", "refit"), default="cached")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-file", type=Path, help="Already downloaded fixed TACK parquet; otherwise download it.")
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Use a new or empty output directory; earlier runs are not overwritten.")
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "logs"
    logs.mkdir()
    receipt = {"mode": args.mode, "status": "RUNNING", "stages": [],
               "coverage": "TACK construction, prediction, primary/E1/E2/target summaries, post hoc component extension, identity, Morgan and bounded structure audits",
               "excluded": ["historical PROTAC-DB nonself anchors and curation", "unpreserved upstream provenance"]}
    receipt_path = output / "REPRODUCTION.json"

    def save():
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    def run(name, script, *arguments):
        started = time.perf_counter()
        command = [sys.executable, "-B", str(ROOT / "scripts" / script), *map(str, arguments)]
        print(f"Running {name} ...", flush=True)
        with (logs / (name + ".log")).open("wb") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, check=False)
        receipt["stages"].append({"stage": name, "exit_code": result.returncode,
                                   "elapsed_seconds": round(time.perf_counter() - started, 3)})
        save()
        if result.returncode:
            raise RuntimeError(f"{name} failed; inspect logs/{name}.log. Earlier outputs were preserved.")

    save()
    try:
        source = args.source_file.resolve() if args.source_file else output / "source/TACK_DC50.parquet"
        if not args.source_file:
            run("source", "fetch_source.py", "--output", source)
        construction = output / "construction"
        run("construction", "reproduce_construction.py", "--input-parquet", source,
            "--output-dir", construction)
        frozen = output / "frozen"
        run("frozen_predictions", "prepare_frozen_predictions.py", "--construction-dir", construction,
            "--output-dir", frozen)
        primary = frozen / "primary_predictions.csv"
        e1, e2 = frozen / "E1_predictions.csv", frozen / "E2_predictions.csv"
        if args.mode == "refit":
            run("primary_fits", "reproduce_primary.py", "--construction-dir", construction,
                "--output-dir", output / "primary", "--n-jobs", args.n_jobs)
            primary = output / "primary/predictions.csv"
            run("supplemental_fits", "fit_supplemental.py",
                "--construction-dir", construction,
                "--output-dir", output / "supplemental", "--n-jobs", args.n_jobs)
            e1, e2 = output / "supplemental/E1_predictions.csv", output / "supplemental/E2_predictions.csv"
        plans = ROOT / "reproduction/plans"
        run("primary_intervals", "recompute_primary_component_metrics.py", "--predictions", primary,
            "--draw-plan", plans / "primary.npz", "--output-dir", output / "primary_intervals")
        run("E1_intervals", "run_supplemental_e1.py", "--predictions", e1,
            "--draw-plan", plans / "E1.npz", "--output-dir", output / "E1_intervals")
        run("E2_intervals", "run_supplemental_e2.py", "--count-predictions", e2,
            "--binary-predictions", primary, "--draw-plan", plans / "E2.npz",
            "--output-dir", output / "E2_intervals")
        run("target_intervals", "evaluate_predictions.py", "--predictions", primary,
            "--pair-target-map", frozen / "pair_target_map.csv", "--key-target-map", frozen / "key_target_map.csv",
            "--target-multiplicities", plans / "target_multiplicities.npy",
            "--output-dir", output / "target_intervals")
        run("identity_audit", "identity_sensitivity.py", "--predictions", primary,
            "--membership", construction / "primary_split_membership.csv",
            "--pair-map", construction / "primary_pair_map.csv",
            "--records", construction / "records_aggregated_v0.1.csv", "--output-dir", output / "identity_audit")
        run("representation_audit", "representation_audit.py", "--pairs", construction / "pairs_v0.1.csv",
            "--records", construction / "records_aggregated_v0.1.csv",
            "--primary-pair-map", construction / "primary_pair_map.csv",
            "--output-dir", output / "representation_audit")
        run("component_extension", "component_extension.py", "--predictions", primary,
            "--output-dir", output / "component_extension")
        run("structure_audit", "audit_structure.py", "--construction-dir", construction,
            "--output-dir", output / "structure_audit")
        run("result_comparison", "verify_results.py", "--run-dir", output)
        receipt["status"] = "COMPLETED_DECLARED_ANALYSES"
    except Exception:
        receipt["status"] = "FAILED_INSPECT_STAGE_LOG"
        raise
    finally:
        save()
    print("Completed the declared analyses. See REPRODUCTION.json for coverage and logs.")


if __name__ == "__main__":
    main()
