#!/usr/bin/env python3
"""Compare reproduced contrast tables against the reported numerical tables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METRICS = {"DA": "direction_accuracy", "MAE": "delta_mae", "RMSE": "delta_rmse"}
WEIGHTS = {"pair_id": "pair", "equal_comparison_key": "equal_group"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    reports = []

    def compare(name, observed, expected, keys, fields):
        if observed.duplicated(keys).any() or expected.duplicated(keys).any():
            raise RuntimeError(f"Duplicate comparison keys in {name}")
        observed = observed.set_index(keys).sort_index()
        expected = expected.set_index(keys).sort_index()
        if not observed.index.equals(expected.index):
            raise RuntimeError(f"Comparison membership differs: {name}")
        differences = np.abs(observed[fields].to_numpy(float) - expected[fields].to_numpy(float))
        passed = bool(np.isfinite(differences).all() and (differences <= 1e-12).all())
        reports.append({"analysis": name, "rows": len(expected), "max_absolute_difference": float(differences.max()),
                        "within_atol_1e_12_rtol_0": passed})

    primary = pd.read_csv(ROOT / "results/primary_contrasts.csv").rename(columns={
        "estimand":"weight", "point_estimate":"estimate", "lower_bound":"adjusted_lower", "upper_bound":"adjusted_upper"})
    primary["weight"] = primary.weight.map(WEIGHTS)
    observed = pd.read_csv(args.run_dir / "primary_intervals/contrasts.csv")
    fields = ["estimate", "adjusted_lower", "adjusted_upper"]
    compare("primary", observed, primary, ["weight", "metric"], fields)
    supplemental = pd.read_csv(ROOT / "results/supplemental_contrasts.csv")
    supplemental["metric"] = supplemental.metric.map(METRICS)
    for name in ("E1", "E2"):
        compare(name, pd.read_csv(args.run_dir / (name + "_intervals/contrasts.csv")),
                supplemental[supplemental.experiment == name], ["comparison", "weight", "metric"], fields)
    target_path = args.run_dir / "target_intervals/target_plan_intervals.csv"
    if target_path.is_file():
        compare("supported_target", pd.read_csv(target_path),
                pd.read_csv(ROOT / "results/target_sensitivity_contrasts.csv"),
                ["estimand", "metric"], ["point_estimate", "lower_bound", "upper_bound"])
    else:
        raise FileNotFoundError("Missing supported-target contrast table.")
    extension = args.run_dir / "component_extension"
    if extension.is_dir():
        compare("component_extension_prefixes", pd.read_csv(extension / "prefix_intervals.csv"),
                pd.read_csv(ROOT / "results/component_extension_prefixes.csv"),
                ["prefix_replicates", "comparison_id", "metric"], ["lower", "upper"])
        compare("component_extension_mcse", pd.DataFrame(json.loads((extension / "endpoint_mcse.json").read_text())),
                pd.read_csv(ROOT / "results/component_extension_mcse.csv"),
                ["comparison_id", "metric"], ["mcse_sd_over_sqrt10"])
    result = {"status": "MATCH_REPORTED_CONTRASTS" if all(r["within_atol_1e_12_rtol_0"] for r in reports) else "MISMATCH",
              "analyses":reports, "scope":"Estimates and interval endpoints; this comparison does not establish calibrated coverage or missing-source reproducibility."}
    (args.run_dir / "RESULT_COMPARISON.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "MATCH_REPORTED_CONTRASTS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
