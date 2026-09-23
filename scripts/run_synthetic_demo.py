#!/usr/bin/env python3
"""Run a small offline aggregation demo with fabricated values.

The demo exercises the same pair/equal-group, within-seed RMSE, synchronous
component-weight, and count/binary pairing code as the fixed-study commands.
It contains no study data, does not fit a model, and is not a scientific
reproduction or validation of the reported numerical results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from component_resampling import (
    AnalysisError,
    DrawPlan,
    analyze_component_family,
    prepare_output_dir,
    sha256_file,
    validate_paired_representations,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def synthetic_table(representation: str) -> pd.DataFrame:
    rows = []
    truths = {"p1": 1.25, "p2": -1.10, "p3": 1.60, "p4": -1.35}
    groups = {"p1": "g1", "p2": "g1", "p3": "g2", "p4": "g3"}
    components = {"p1": "c1", "p2": "c1", "p3": "c2", "p4": "c2"}
    base_predictions = {
        "random_forest": {"p1": 0.60, "p2": -0.20, "p3": 0.90, "p4": -0.55},
        "xgboost": {"p1": 0.95, "p2": -0.65, "p3": 1.20, "p4": -0.80},
        "ridge": {"p1": 0.80, "p2": -0.45, "p3": 1.05, "p4": -0.70},
    }
    count_shift = {
        "random_forest": 0.03,
        "xgboost": -0.04,
        "ridge": 0.02,
    }
    for split_index, split_seed in enumerate((101, 202)):
        for model, predictions in base_predictions.items():
            seeds = (11, 22) if model != "ridge" else (-1,)
            for model_seed in seeds:
                seed_shift = 0.01 if model_seed == 22 else 0.0
                for pair_index, pair_id in enumerate(("p1", "p2", "p3", "p4")):
                    prediction = predictions[pair_id] + seed_shift + split_index * 0.02
                    if representation == "count":
                        prediction += count_shift[model]
                    rows.append(
                        {
                            "pair_id": pair_id,
                            "comparison_pair_key": groups[pair_id],
                            "component_id": components[pair_id],
                            "true_delta": truths[pair_id],
                            "prediction": prediction,
                            "model": model,
                            "model_seed": model_seed,
                            "split_seed": split_seed,
                            "fold": 0 if components[pair_id] == "c1" else 1,
                        }
                    )
    return pd.DataFrame(rows)


def main() -> int:
    args = build_parser().parse_args()
    try:
        root = prepare_output_dir(args.output_dir)
        inputs = root / "inputs"
        inputs.mkdir()
        binary_path = inputs / "synthetic_binary_predictions.csv"
        count_path = inputs / "synthetic_count_predictions.csv"
        plan_path = inputs / "synthetic_draw_plan.npz"
        binary = synthetic_table("binary")
        count = synthetic_table("count")
        binary.to_csv(binary_path, index=False, lineterminator="\n", float_format="%.17g")
        count.to_csv(count_path, index=False, lineterminator="\n", float_format="%.17g")
        components = ("c1", "c2", "c_empty")
        weights = np.asarray(
            [
                [1, 1, 1],
                [2, 1, 0],
                [1, 2, 0],
                [0, 2, 1],
                [2, 0, 1],
                [0, 1, 2],
            ],
            dtype=np.int16,
        )
        np.savez_compressed(
            plan_path,
            weights=weights,
            components=np.asarray(components),
            seed=np.asarray(-1, dtype=np.int32),
        )
        plan = DrawPlan(
            weights=weights,
            components=components,
            source_name=plan_path.name,
            source_bytes=plan_path.stat().st_size,
            source_sha256=sha256_file(plan_path),
            recorded_seed=None,
        )
        validate_paired_representations(count, binary)

        primary_frames = {
            model: binary[binary["model"] == model].copy()
            for model in ("random_forest", "xgboost", "ridge")
        }
        primary = analyze_component_family(
            primary_frames,
            plan,
            comparisons=(("xgboost_vs_random_forest", "xgboost", "random_forest"),),
            family_size=6,
        )
        e2_frames = {}
        e2_comparisons = []
        for model in ("random_forest", "xgboost", "ridge"):
            a = f"{model}_count"
            b = f"{model}_binary"
            e2_frames[a] = count[count["model"] == model].copy()
            e2_frames[b] = binary[binary["model"] == model].copy()
            e2_comparisons.append((f"{model}_count_vs_binary", a, b))
        e2 = analyze_component_family(
            e2_frames,
            plan,
            comparisons=tuple(e2_comparisons),
            family_size=18,
        )

        for name, outputs in (("primary_like", primary), ("representation_like", e2)):
            destination = root / name
            destination.mkdir()
            pd.DataFrame(outputs.descriptive).to_csv(
                destination / "descriptive_metrics.csv",
                index=False,
                lineterminator="\n",
                float_format="%.17g",
            )
            pd.DataFrame(outputs.contrasts).to_csv(
                destination / "contrasts.csv",
                index=False,
                lineterminator="\n",
                float_format="%.17g",
            )
            pd.DataFrame(outputs.diagnostics).to_csv(
                destination / "draw_diagnostics.csv",
                index=False,
                lineterminator="\n",
            )
        receipt = {
            "status": "PASS_SYNTHETIC_AGGREGATION_DEMO",
            "scientific_reproduction": False,
            "study_data_used": False,
            "model_fits": 0,
            "predictions_created_by_model": 0,
            "bootstrap_draws_generated": 0,
            "fixed_study_contract_checked": False,
            "purpose": (
                "Exercise exact in-memory count/binary pairing, pair and equal-group aggregation, "
                "within-seed RMSE, and saved component-weight replay."
            ),
            "warning": (
                "A PASS here only shows that the local aggregation code runs on fabricated values. "
                "It does not reproduce, validate, or approximate the reported study results."
            ),
        }
        (root / "DEMO_RECEIPT.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        print(json.dumps(receipt, indent=2))
        return 0
    except (AnalysisError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
