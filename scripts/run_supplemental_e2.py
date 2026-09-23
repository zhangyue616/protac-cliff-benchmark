#!/usr/bin/env python3
"""Recompute E2 folded-count versus binary metrics from saved predictions.

The command requires one-to-one count/binary OOF keys, truths, comparison
groups, and components before replaying the saved 50,000 x 77 component plan.
It performs no fit, prediction, fingerprint generation, or bootstrap draw.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from component_resampling import (
    E2_STREAMS,
    AnalysisError,
    analyze_component_family,
    print_receipt,
    read_draw_plan,
    read_prediction_csv,
    validate_component_support,
    validate_paired_representations,
    write_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--count-predictions",
        required=True,
        type=Path,
        help="Saved E2 folded-count OOF CSV for ridge, random forest, and XGBoost.",
    )
    parser.add_argument(
        "--binary-predictions",
        required=True,
        type=Path,
        help=(
            "Saved primary binary OOF CSV. It may contain all eight streams; "
            "the same three models are selected and exactly paired."
        ),
    )
    parser.add_argument(
        "--draw-plan",
        required=True,
        type=Path,
        help="Saved 50,000 x 77 NPZ or long-form CSV component plan.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        count_path = args.count_predictions.resolve()
        binary_path = args.binary_predictions.resolve()
        plan_path = args.draw_plan.resolve()
        count = read_prediction_csv(
            count_path,
            label="E2 count predictions",
            expected_streams=E2_STREAMS,
        )
        binary = read_prediction_csv(
            binary_path,
            label="E2 binary predictions",
            expected_streams=E2_STREAMS,
            allow_other_models=True,
        )
        validate_paired_representations(count, binary)
        plan = read_draw_plan(
            plan_path,
            expected_replicates=50_000,
            expected_components=77,
        )
        frames = {}
        comparisons = []
        for model in E2_STREAMS:
            count_label = f"{model}_count"
            binary_label = f"{model}_binary"
            frames[count_label] = count[count["model"] == model].copy()
            frames[binary_label] = binary[binary["model"] == model].copy()
            comparisons.append((f"{model}_count_vs_binary", count_label, binary_label))
        component_summary = validate_component_support(
            frames,
            plan,
            expected_supported_components=40,
        )
        outputs = analyze_component_family(
            frames,
            plan,
            comparisons=tuple(comparisons),
            family_size=18,
        )
        receipt = write_outputs(
            args.output_dir,
            outputs,
            analysis_id="E2_folded_count_vs_binary",
            plan=plan,
            component_summary=component_summary,
            input_files={
                "count_predictions": count_path,
                "binary_predictions": binary_path,
            },
            limitations=(
                "Intervals are conditional on the supplied saved multiplicity plan.",
                "Coverage was not calibrated.",
                "Fixed hyperparameters, including ridge alpha=1, were not retuned for count scale.",
                "Zero-spanning intervals do not establish equivalence.",
                "The input contract assumes true_delta is oriented as endpoint j minus endpoint i; pairing proves equality between the supplied tables but cannot reconstruct orientation from records.",
                "This command does not reproduce fingerprint generation or the 175 E2 fits.",
            ),
            semantic_checks={
                "count_binary_oof_keys_exactly_paired": True,
                "count_binary_truth_equal_at_atol_1e_12_and_groups_components_exact": True,
                "truth_values_are_finite_and_nonzero": True,
                "rmse_sqrt_within_seed_before_seed_mean": True,
                "all_77_components_including_37_empty_retained": True,
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
