#!/usr/bin/env python3
"""Recompute E1 identity-disjoint metrics from saved OOF predictions.

The command replays the separate six-contrast family on the saved 2,000 x 55
component plan.  It does not rebuild identity components, folds, fits, or
predictions and does not generate a replacement draw plan.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from component_resampling import (
    FULL_STREAMS,
    AnalysisError,
    analyze_component_family,
    print_receipt,
    read_draw_plan,
    read_prediction_csv,
    validate_component_support,
    write_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        required=True,
        type=Path,
        help="Saved E1 OOF CSV with all eight model/control streams.",
    )
    parser.add_argument(
        "--draw-plan",
        required=True,
        type=Path,
        help="Saved 2,000 x 55 NPZ or long-form CSV component plan.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        predictions_path = args.predictions.resolve()
        plan_path = args.draw_plan.resolve()
        predictions = read_prediction_csv(
            predictions_path,
            label="E1 predictions",
            expected_streams=FULL_STREAMS,
        )
        plan = read_draw_plan(
            plan_path,
            expected_replicates=2_000,
            expected_components=55,
        )
        frames = {
            model: predictions[predictions["model"] == model].copy()
            for model in FULL_STREAMS
        }
        component_summary = validate_component_support(
            frames,
            plan,
            expected_supported_components=29,
        )
        outputs = analyze_component_family(
            frames,
            plan,
            comparisons=(("xgboost_vs_random_forest", "xgboost", "random_forest"),),
            family_size=6,
        )
        receipt = write_outputs(
            args.output_dir,
            outputs,
            analysis_id="E1_identity_disjoint_xgboost_vs_random_forest",
            plan=plan,
            component_summary=component_summary,
            input_files={"E1_predictions": predictions_path},
            limitations=(
                "Intervals are conditional on the supplied saved multiplicity plan.",
                "Coverage was not calibrated.",
                "E1 changes components and folds; cross-protocol changes do not isolate a causal identity effect.",
                "Zero-spanning intervals do not establish equivalence.",
                "The input contract assumes true_delta is oriented as endpoint j minus endpoint i; this command can verify that values are finite and nonzero but cannot reconstruct orientation from records.",
                "This command does not reproduce the 200 E1 fits.",
            ),
            semantic_checks={
                "truth_values_are_finite_and_nonzero": True,
                "rmse_sqrt_within_seed_before_seed_mean": True,
                "all_55_components_including_26_empty_retained": True,
                "pair_and_equal_group_weighting_recomputed": True,
            },
        )
        print_receipt(receipt)
        return 0 if receipt["status"].startswith("PASS") else 2
    except (AnalysisError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
