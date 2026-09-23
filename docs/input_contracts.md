# Input and output contracts

The three fixed-study commands rescore saved out-of-fold predictions with saved component-multiplicity plans. They do not generate either input. All CSV files are read by header name; extra prediction or draw-plan columns are ignored.

## Commands

~~~bash
python scripts/recompute_primary_component_metrics.py --predictions <csv> --draw-plan <npz-or-csv> --output-dir <dir>
python scripts/run_supplemental_e1.py --predictions <csv> --draw-plan <npz-or-csv> --output-dir <dir>
python scripts/run_supplemental_e2.py --count-predictions <csv> --binary-predictions <csv> --draw-plan <npz-or-csv> --output-dir <dir>
~~~

Each output directory must be absent or empty. The command refuses to mix new output with existing files.

## Prediction CSV

Every prediction input requires these columns:

| Column | Required content |
| --- | --- |
| pair_id | Nonempty directed-pair identifier. There must be exactly 874 pair identifiers, each appearing once per saved stream and split. |
| comparison_pair_key | Nonempty comparison-group identifier. There must be exactly 838 groups per split, and a group cannot span components within a split. |
| component_id | Nonempty component identifier. It must occur in the supplied draw plan. |
| true_delta | Finite, nonzero signed pDC50 difference. It must be identical across streams for the same OOF key. The study meaning is pDC50(j) minus pDC50(i); the CSV alone cannot prove that the caller used that orientation. |
| prediction | Finite saved prediction on the same orientation as true_delta. |
| model | Exact model or control name from the applicable stream inventory below. |
| model_seed | Finite integer model-seed identifier. |
| split_seed | One of 20260624, 20260625, 20260626, 20260724, or 20260801. |
| fold | Integer 0, 1, 2, 3, or 4. |

The unique row key is:

~~~text
(model, model_seed, split_seed, fold, pair_id)
~~~

All streams in a file must share the same 4,370 OOF keys, defined by (split_seed, fold, pair_id). Each stream has 4,370 rows, with the same 874-pair universe in each of the five split seeds. comparison_pair_key, component_id, and true_delta must agree across streams for every OOF key. For a fixed pair_id, those three fields must also remain unchanged across all split seeds. Within one split seed, a pair may belong to only one fold, and every pair assigned to the same component must share one fold.

### Primary and E1 stream inventory

The primary and E1 files require all 12 saved streams:

| model | model_seed values |
| --- | --- |
| zero_delta | -1 |
| train_mean | -1 |
| source_prior | -1 |
| nearest_neighbor | -1 |
| ridge | -1 |
| random_forest | 20260624, 20260724, 20260801 |
| xgboost | 20260624, 20260724, 20260801 |
| permuted_xgboost | 20260803 |

The primary prediction table must use exactly 40 supported components from the 77-component plan. E1 must use exactly 29 supported components from the 55-component plan. Every model must use the same supported-component set.

### E2 stream inventory and pairing

The folded-count file requires seven streams:

| model | model_seed values |
| --- | --- |
| ridge | -1 |
| random_forest | 20260624, 20260724, 20260801 |
| xgboost | 20260624, 20260724, 20260801 |

The binary file may contain either these three models or the full 12-stream primary table. When other primary models are present, the command selects ridge, random_forest, and xgboost before enforcing the E2 contract.

After that selection, count and binary rows must pair one to one on the full saved-stream OOF key. comparison_pair_key and component_id must match exactly. true_delta is compared with absolute tolerance 1 × 10^-12 and zero relative tolerance; this admits floating-point serialization differences without changing orientation or membership. In the frozen inputs, 1,960 of 30,590 paired rows differ at the bit level, the maximum absolute difference is 1.9984014443252818 × 10^-15, and no row exceeds the gate. Both representations must use the same 40 supported components from the 77-component draw plan.

## Saved draw plan

The draw plan can be NPZ or long-form CSV. The commands read the supplied multiplicities and never create replacement draws.

### NPZ form

The archive requires:

| Array | Contract |
| --- | --- |
| weights | Two-dimensional finite, nonnegative integer multiplicities. |
| components | One-dimensional component identifiers, one per weights column; identifiers must be nonempty and unique. |
| seed | Optional scalar integer recorded as provenance. It is not used to generate draws. |

### Long CSV form

The required columns are replicate_id, component_id, and draw_count. replicate_id and draw_count must be nonnegative integers. replicate identifiers must be contiguous from zero. Every replicate/component cell must appear exactly once, and no component identifier may be blank.

### Fixed shapes

| Command | Plan shape | Tickets per replicate | Supported / retained-empty components |
| --- | ---: | ---: | ---: |
| Primary | 2,000 × 77 | 77 | 40 / 37 |
| E1 | 2,000 × 55 | 55 | 29 / 26 |
| E2 | 50,000 × 77 | 77 | 40 / 37 |

Every row of weights must sum to the stated ticket count. A digest in RUN_METADATA.json identifies the supplied plan bytes; it does not recover unpreserved generator history or establish when the plan was specified.

## Aggregation

For pair weighting, every pair row is a unit. For equal-group weighting, row-level direction indicators, absolute errors, and squared errors are averaged within comparison_pair_key before groups receive equal weight.

Metrics are calculated within each model-seed stream after pooling its five split copies. RMSE is square-rooted within each seed before seed-level metrics or contrasts are averaged equally. One saved component-weight vector is applied synchronously across all compared models or representations, seeds, weightings, metrics, and split copies.

See [analysis definitions](analysis_definitions.md) for contrast directions and family identities.

## Fixed-study outputs

Each successful fixed-study command writes:

| File | Contents |
| --- | --- |
| descriptive_metrics.csv | One row per label, weighting, and metric, with seed inventory, support, and aggregation note. Primary and E1 write 48 rows; E2 writes 36. |
| contrasts.csv | Point estimates, multiplicity-adjusted bounds, cell-level interval status, and family_status. Primary and E1 write six rows; E2 writes 18. |
| draw_diagnostics.csv | Zero-denominator and nonfinite-draw counts for every label, weighting, and model seed. |
| draw_contrasts.npz | Draw-level contrast matrix named by comparison, weighting, and metric. Its shape is B × 6 for primary and E1 or B × 18 for E2. |
| RUN_METADATA.json | Input basenames, byte sizes and SHA-256 digests; draw-plan identity and shape; component support; output row counts; semantic checks; operation boundary; limitations; and final status. |

contrasts.csv uses these direction fields:

- direction_accuracy: first_minus_second;
- delta_mae and delta_rmse: second_minus_first;
- positive_favors names the first label.

Any zero-denominator or nonfinite draw makes the affected interval non-estimable; the command does not silently discard or replace failed draws. It retains estimates and intervals for other computable cells, but every contrast row in that family receives family_status INCOMPLETE_NONESTIMABLE_CELL_PRESENT, and RUN_METADATA.json reports FAIL_OR_NONESTIMABLE_REQUIRES_REVIEW. Such output is not a complete interval family. When every cell is estimable, contrasts.csv reports family_status COMPLETE_ALL_INTERVALS and RUN_METADATA.json reports PASS_COMPUTATION_UNCALIBRATED_INTERVALS. These statuses describe the saved-input rescoring path. They are not claims of calibrated interval coverage or end-to-end scientific reproduction.

## Synthetic-demo outputs

~~~bash
python scripts/run_synthetic_demo.py --output-dir demo_output
~~~

The demo output directory contains:

~~~text
DEMO_RECEIPT.json
inputs/
  synthetic_binary_predictions.csv
  synthetic_count_predictions.csv
  synthetic_draw_plan.npz
primary_like/
  descriptive_metrics.csv
  contrasts.csv
  draw_diagnostics.csv
representation_like/
  descriptive_metrics.csv
  contrasts.csv
  draw_diagnostics.csv
~~~

The demo calls the shared calculation functions with a small fabricated design. It does not execute the fixed-study input readers or the three fixed-study CLIs, and DEMO_RECEIPT.json records fixed_study_contract_checked as false.
