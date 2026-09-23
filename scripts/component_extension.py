#!/usr/bin/env python3
"""Replay the reported post hoc 50,000-component extension and MCSE/prefix checks.

The candidate generator reproduces the saved first 2,000 multiplicity vectors.
This is not recovery of the original ordered draws or original probability law.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import numpy as np
import pandas as pd
from component_resampling import (FULL_STREAMS, analyze_component_family, read_prediction_csv,
                                  read_draw_plan, write_outputs, validate_component_support)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError("Use a new or empty output directory.")
    primary = read_draw_plan(ROOT / "reproduction/plans/primary.npz", expected_replicates=2000, expected_components=77)
    rng = random.Random(20260802)
    weights = np.zeros((50000, 77), dtype=np.int16)
    for b in range(50000):
        for _ in range(77):
            weights[b, rng.randrange(77)] += 1
    if not np.array_equal(weights[:2000], primary.weights):
        raise RuntimeError("Candidate generator does not reproduce the saved multiplicities; no substitute was accepted.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = args.output_dir / "posthoc_component_plan.npz"
    np.savez_compressed(plan_path, weights=weights, components=np.array(primary.components), seed=np.int64(20260802))
    plan = read_draw_plan(plan_path, expected_replicates=50000, expected_components=77)
    predictions = read_prediction_csv(args.predictions, label="primary predictions", expected_streams=FULL_STREAMS)
    frames = {name: predictions[predictions.model == name].copy() for name in ("xgboost", "random_forest")}
    support = validate_component_support(frames, plan, expected_supported_components=40)
    computed = analyze_component_family(frames, plan,
        comparisons=(("xgboost_vs_random_forest", "xgboost", "random_forest"),), family_size=6)
    numerical = args.output_dir / "numerical"
    receipt = write_outputs(numerical, computed, analysis_id="posthoc_component_50000_extension",
        plan=plan, component_summary=support, input_files={"predictions": args.predictions},
        limitations=("Post hoc candidate-generator extension; not original RNG-history recovery.",
                     "Coverage uncalibrated; MCSE describes the 50,000-draw extension only."),
        semantic_checks={"first_2000_multiplicity_vectors_match_retained_plan": True})
    if not receipt["status"].startswith("PASS"):
        raise RuntimeError("Nonestimable extension draws; inspect preserved numerical outputs.")
    prefix_rows, mcse_rows = [], []
    q = 0.05 / 12
    for index, name in enumerate(computed.draw_columns):
        _, weight, metric = name.split("__")
        values = computed.draw_values[:, index]
        comparison_id = {"pair":"pair_id", "equal_group":"equal_comparison_key"}[weight]
        for n in (20000, 30000, 40000, 50000):
            lo, hi = np.quantile(values[:n], [q, 1-q], method="linear")
            prefix_rows.append({"prefix_replicates":n, "comparison_id":comparison_id,
                                "metric":metric, "lower":lo, "upper":hi})
        lows = [float(np.quantile(values[j*5000:(j+1)*5000], q, method="linear")) for j in range(10)]
        mcse_rows.append({"comparison_id":comparison_id, "metric":metric,
                          "lower_endpoints":lows, "mcse_sd_over_sqrt10":float(np.std(lows, ddof=1)/np.sqrt(10))})
    pd.DataFrame(prefix_rows).to_csv(args.output_dir / "prefix_intervals.csv", index=False, float_format="%.17g")
    (args.output_dir / "endpoint_mcse.json").write_text(json.dumps(mcse_rows, indent=2)+"\n", encoding="utf-8")
    (args.output_dir / "EXTENSION_RUN.json").write_text(json.dumps({
        "status": "COMPLETED_POSTHOC_CANDIDATE_EXTENSION",
        "candidate_replicates_generated": 50000,
        "components": 77,
        "candidate_seed": 20260802,
        "first_2000_multiplicities_match_retained_plan": True,
        "original_ordered_draws_or_probability_law_recovered": False,
        "numerical_receipt_scope": "The nested numerical receipt describes rescoring the supplied plan, not this wrapper's generation step."
    }, indent=2)+"\n", encoding="utf-8")
    print("Completed the reported post hoc component extension, corrected prefixes and ten-batch lower-endpoint MCSE.")


if __name__ == "__main__":
    main()
