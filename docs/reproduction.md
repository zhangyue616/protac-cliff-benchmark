# Reproduction guide

The repository supports two routes through the TACK-based analyses. Both start with the fixed public source and reconstruct the same records, pairs, graph, folds, fingerprints and signed targets.

- **Cached:** attach reconstructed targets to the released historical predictions and recompute the reported summaries.
- **Refit:** rerun the 200 primary and 375 supplemental fits with the recorded recipes, then compute the same summaries.

Neither unified route reconstructs the historical PROTAC-DB nonself-overlap anchors, manual source adjudications, or unpreserved upstream provenance. A separate [conditional local audit](data_access.md#historical-source-diagnostics) accepts a user-obtained fixed workbook; source access remains unresolved. The reported post hoc 50,000-component extension and bounded full-graph chemical mapping are included in both unified routes. The extension checks its first 2,000 multiplicity vectors against the saved plan; it does not recover the original draw order or probability law.

## Environment and command

The recorded fitting environment used CPython 3.11.15. Create an isolated Python 3.11 environment and install the pinned dependencies:

~~~bash
python -m pip install -r environment/primary-requirements.txt
python scripts/reproduce.py --mode cached --output-dir ../protac-cached
~~~

For the fitting route:

~~~bash
python scripts/reproduce.py --mode refit --output-dir ../protac-refit
~~~

Use a new or empty output directory. To reuse a fixed source download, add `--source-file <parquet>`. The source is checked against the recovered fixed snapshot before construction. A failed stage preserves its outputs and log; do not interpret them as a completed run.

The pinned environment is a reproduction recipe, not a guarantee that every operating system has matching wheels or identical native-library behavior. Compare regenerated results with the supplied reference tables. The lightweight `requirements.txt` supports rescoring only; it does not provide RDKit or model-fitting dependencies.

The loaders preserve the original CSV floating-point parsers: `high` for the primary fit and `round_trip` for the supplemental fits. Their last-bit differences can change random-forest split ties, so the two parsing paths are intentionally retained.

## Outputs and coverage

| Directory | Recomputed content |
| --- | --- |
| construction/ | 4,184 source identities; 3,302 endpoint-eligible records; 731 pair-eligible records; 6,015 compatible pairs; 874 cliffs; 77 graph components; original folds and binary fingerprints. |
| frozen/ | Historical predictions joined to reconstructed j-minus-i targets; 17-target pair and comparison-group maps. |
| primary/ | Refit mode only: original 200 fits, 52,440 OOF rows and orientation-reversal predictions. |
| supplemental/ | Refit mode only: E1 identity-disjoint and E2 folded-count fits, predictions and plans. |
| primary_intervals/ | Six original component contrasts using the saved 2,000 × 77 plan. |
| E1_intervals/ | Six identity-disjoint contrasts using the saved 2,000 × 55 plan. |
| E2_intervals/ | Eighteen count-versus-binary contrasts using the saved 50,000 × 77 plan. |
| target_intervals/ | Six separate supported-target contrasts using the saved 50,000 × 17 plan. |
| identity_audit/ | Frozen-prediction identity-recurrence and recorded-assay subsets. |
| representation_audit/ | Binary, count, sparse and chirality-aware Morgan comparisons for the bounded fingerprint-equality subset. |
| component_extension/ | Post hoc 50,000-component candidate plan, corrected 20k/30k/40k/50k prefixes and ten-batch lower-endpoint MCSE. |
| structure_audit/ | Full-graph mapping of the 16 selected pairs: 8 methylene-chain, 6 polyether-repeat and 2 stereochemistry-only differences. |
| logs/ | Output and errors from each stage. |

`REPRODUCTION.json` records the stages actually executed. `RESULT_COMPARISON.json` compares all 36 contrast estimates and interval endpoints, the extension prefixes and its MCSE against the reported tables using absolute tolerance 1e-12 and zero relative tolerance. Matching numbers establish only this declared coverage, not calibrated intervals, causal conclusions, or recovery of missing historical sources.

## Individual stages

~~~bash
python scripts/fetch_source.py --output ../inputs/TACK_DC50.parquet
python scripts/reproduce_construction.py --input-parquet ../inputs/TACK_DC50.parquet --output-dir ../work/construction
python scripts/prepare_frozen_predictions.py --construction-dir ../work/construction --output-dir ../work/frozen
python scripts/reproduce_primary.py --construction-dir ../work/construction --output-dir ../work/primary
~~~

The statistical entry points accept either reconstructed historical prediction tables or newly fitted prediction tables:

~~~bash
python scripts/recompute_primary_component_metrics.py --predictions <csv> --draw-plan reproduction/plans/primary.npz --output-dir <dir>
python scripts/run_supplemental_e1.py --predictions <csv> --draw-plan reproduction/plans/E1.npz --output-dir <dir>
python scripts/run_supplemental_e2.py --count-predictions <csv> --binary-predictions <csv> --draw-plan reproduction/plans/E2.npz --output-dir <dir>
~~~

Exact schemas and aggregation order are in [input contracts](input_contracts.md). The individual statistical commands consume supplied plans; they do not silently create replacement draws, discard failed replicates, or reselect the analysis population.

## Checked execution

On 23 September 2026, a fresh download of the fixed TACK file reproduced the retained construction outputs. The 200 primary and 375 supplemental fitting recipes were checked in the recorded Python 3.11 environment; the resulting predictions matched the retained predictions within absolute tolerance 1e-12. All 36 contrast estimates and interval endpoints, the component-extension prefixes and its MCSE also matched within that tolerance. The 16-pair structure audit recovered the stated 8/6/2 chemical categories.

This was a check in the existing recorded environment, not a fresh dependency installation or a cross-platform test. The cached driver ran construction and numerical stages; the subsequently added extension and structure stages and the corrected refit outputs were checked separately. The final integrated refit command was not rerun as a second complete job. Historical PROTAC-DB access is not validated by these checks.

## Optional synthetic check

~~~bash
python -m pip install -r requirements.txt
python scripts/run_synthetic_demo.py --output-dir ../synthetic-demo
~~~

This small, offline example exercises shared aggregation code on fabricated data. It is useful for checking dependencies, but it does not reproduce or validate the scientific results.
