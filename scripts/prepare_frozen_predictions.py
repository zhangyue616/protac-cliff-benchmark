#!/usr/bin/env python3
"""Attach reconstructed j-minus-i targets to the released author predictions.

The cache contains predictions, author-generated keys, and split annotations;
it contains no SMILES, source activities, or source text. Targets are calculated
from records reconstructed from the fixed public TACK input. This path rescores
historical predictions; it does not validate model fitting.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KEY = ["model", "model_seed", "split_seed", "fold", "pair_id"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--construction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Output directory must be absent or empty.")
    records = pd.read_csv(args.construction_dir / "records_aggregated_v0.1.csv", float_precision="high")
    supplemental_records = pd.read_csv(args.construction_dir / "records_aggregated_v0.1.csv", float_precision="round_trip")
    pairs = pd.read_csv(args.construction_dir / "primary_pair_map.csv")
    if records.record_id.duplicated().any() or pairs.pair_id.duplicated().any() or len(pairs) != 874:
        raise RuntimeError("Construction keys or pair population do not match the fixed study.")
    activities = records.set_index("record_id").pdc50_median
    pairs["true_delta"] = (pairs.record_id_j.map(activities) - pairs.record_id_i.map(activities))
    if not np.isfinite(pairs.true_delta).all() or not (pairs.true_delta.abs() >= 1.0 - 1e-12).all():
        raise RuntimeError("Reconstructed targets fail the measured-cliff population gate.")
    annotations = pairs[["pair_id", "record_id_i", "record_id_j", "true_delta"]]
    supplemental_annotations = annotations.copy()
    supplemental_activity = supplemental_records.set_index("record_id").pdc50_median
    supplemental_annotations["true_delta"] = (
        pairs.record_id_j.map(supplemental_activity) - pairs.record_id_i.map(supplemental_activity))
    output.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, expected in (("primary", 52440), ("E1", 52440), ("E2", 30590)):
        cached = pd.read_csv(ROOT / "reproduction" / "predictions" / (name + ".csv.gz"),
                             float_precision="round_trip")
        if len(cached) != expected or cached.duplicated(KEY).any():
            raise RuntimeError(f"Invalid frozen prediction key inventory: {name}")
        # Preserve the historical parser used by each analysis, including last-bit targets.
        target_annotations = annotations if name == "primary" else supplemental_annotations
        merged = cached.merge(target_annotations, on="pair_id", how="left", validate="many_to_one")
        if merged.true_delta.isna().any() or not np.isfinite(merged.prediction).all():
            raise RuntimeError(f"Unmatched or nonfinite prediction rows: {name}")
        merged.to_csv(output / (name + "_predictions.csv"), index=False, float_format="%.17g")
        counts[name] = len(merged)
        if name == "primary":
            mapping = cached[["pair_id", "comparison_pair_key", "component_id"]].drop_duplicates()
            mapping = mapping.merge(pairs[["pair_id", "normalized_poi_id"]],
                                    on="pair_id", validate="one_to_one")
            targets = sorted(mapping.normalized_poi_id.unique())
            if len(targets) != 17:
                raise RuntimeError("Reconstructed target universe differs from the 17-target analysis.")
            mapping["target_index"] = mapping.normalized_poi_id.map({target:i for i,target in enumerate(targets)})
            mapping.to_csv(output / "pair_target_map.csv", index=False)
            keys = mapping.drop(columns="pair_id").drop_duplicates()
            if len(keys) != 838 or keys.comparison_pair_key.duplicated().any():
                raise RuntimeError("Comparison groups do not map uniquely to the target universe.")
            keys.to_csv(output / "key_target_map.csv", index=False)
    (output / "PREPARATION.json").write_text(json.dumps({
        "status": "PREPARED_FROZEN_PREDICTIONS_WITH_RECONSTRUCTED_TARGETS",
        "target": "pDC50(j) minus pDC50(i)", "rows": counts,
        "model_fitting_verified": False}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
