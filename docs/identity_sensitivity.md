# Fixed-OOF identity sensitivity analysis

`scripts/identity_sensitivity.py` recomputes four descriptive evaluation subsets from fixed, already-saved out-of-fold (OOF) predictions. It does not fit a model, generate predictions, change folds, resample rows, bootstrap metrics, or download data.

The repository does not include or fetch the study inputs. Run the script only with separately supplied CSV files that you are authorized to use and that satisfy the contract below.

## Command

Python 3.11 or newer, NumPy, and pandas are required. The representative run used Python 3.12.14. From the repository root:

```bash
python scripts/identity_sensitivity.py \
  --predictions ../private_inputs/predictions.csv \
  --membership ../private_inputs/primary_split_membership.csv \
  --pair-map ../private_inputs/primary_pair_map.csv \
  --records ../private_inputs/records.csv \
  --output-dir ../local-results/identity
```

All five paths are required. The default output is aggregate-only. Add `--write-membership` only when a local row-level audit file is needed:

```bash
python scripts/identity_sensitivity.py \
  --predictions ../private_inputs/predictions.csv \
  --membership ../private_inputs/primary_split_membership.csv \
  --pair-map ../private_inputs/primary_pair_map.csv \
  --records ../private_inputs/records.csv \
  --output-dir ../local-results/identity \
  --write-membership
```

Inputs and the optional `membership.csv` output can contain restricted record, pair, component, POI, identity-hash, and assay-token values. Keep them in an access-controlled, ignored output directory and do not commit them. The script reads only three columns from `--records`; it does not read or write SMILES.

## Minimal input schemas

Additional columns are allowed and ignored. Identifier fields are read as opaque, nonempty strings. Integer and numeric fields must parse exactly as described and may not be blank or nonfinite.

### `--predictions`

| Column | Required meaning |
|---|---|
| `pair_id` | Primary-pair identifier. |
| `comparison_pair_key` | Comparison/context key copied from membership. |
| `component_id` | Split-component identifier copied from membership. |
| `record_id_i` | Oriented endpoint-i record identifier. |
| `record_id_j` | Oriented endpoint-j record identifier. |
| `true_delta` | Numeric OOF truth on the **j minus i** convention. It must be nonzero. |
| `prediction` | Finite saved prediction on the same **j minus i** convention. |
| `model` | One of the eight model/control names listed under saved streams. |
| `model_seed` | Integer saved-stream seed. |
| `split_seed` | Integer split seed. |
| `fold` | Integer held-out fold. |

Every `(pair_id, split_seed, fold, model, model_seed)` row must be unique. Within a base `(pair_id, split_seed, fold)` key, `true_delta` must be identical across all saved streams.

### `--membership`

| Column | Required meaning |
|---|---|
| `split_seed` | Integer split seed. |
| `heldout_fold` | Integer fold containing the pair for that split. |
| `pair_id` | Primary-pair identifier. |
| `comparison_pair_key` | Comparison/context key. |
| `component_id` | Split-component identifier. |
| `record_id_i` | Oriented endpoint-i record identifier. |
| `record_id_j` | Oriented endpoint-j record identifier. |

Each `(split_seed, pair_id)` must be unique. Every split must contain the same 874 `pair_id` values exactly once, with every fold represented.

### `--pair-map`

| Column | Required meaning |
|---|---|
| `pair_id` | Unique primary-pair identifier. Exactly 874 rows are required. |
| `comparison_pair_key` | Comparison/context key matching membership. |
| `record_id_i` | Oriented endpoint-i record identifier matching membership. |
| `record_id_j` | Oriented endpoint-j record identifier matching membership. |
| `normalized_poi_id` | Nonempty normalized POI identifier used only for aggregate support counts. |
| `true_delta` | Finite numeric truth on the **i minus j** convention. |
| `is_cliff` | The case-insensitive text `true`; the table must contain only primary cliffs. |

The orientation is deliberately opposite to the prediction file. For every joined row, the script requires:

```text
predictions.true_delta = j - i = -pair_map.true_delta
```

### `--records`

| Column | Required meaning |
|---|---|
| `record_id` | Unique endpoint record identifier. Extra, unused records are allowed. |
| `identity_hash` | Canonical-isomeric identity hash: exactly 64 uppercase hexadecimal characters. |
| `assay_id_or_text` | Either `MISSING_ASSAY_ID` or `RAWASSAY:` followed by 64 uppercase hexadecimal characters. |

`RAWASSAY` values are hashes of normalized recorded assay text. They do not establish ontology-level assay identity or experimental-protocol equivalence. The expected missing sentinel must occur among the primary-pair endpoints; this catches supplying a record table with a different assay encoding.

## Join and fixed-study contract

The joins and redundant-key checks are part of the analysis contract:

| From | To | Join key | Additional equality required |
|---|---|---|---|
| membership | pair map | `pair_id` | `comparison_pair_key`, `record_id_i`, and `record_id_j` |
| membership endpoint i | records | `record_id_i = record_id` | Exactly one record row per identifier |
| membership endpoint j | records | `record_id_j = record_id` | Exactly one record row per identifier |
| predictions | enriched membership | `(pair_id, split_seed, fold)` with `fold = heldout_fold` | `comparison_pair_key`, `component_id`, `record_id_i`, `record_id_j`, and the truth-sign relation above |

The fixed design expects five split seeds and five folds:

```text
split_seed:    20260624, 20260625, 20260626, 20260724, 20260801
heldout_fold:  0, 1, 2, 3, 4
```

Each split contains the same 874 primary cliffs. For a test row in `(split_seed, heldout_fold)`, the training population used for the identity recurrence counts is exactly the same split's primary-cliff rows whose `heldout_fold` differs. Compatible graph edges, standalone records, and any other rows are not added to this training population.

The prediction file must contain all 12 fixed saved streams for every OOF key:

| `model` | Required `model_seed` values | Streams per OOF key |
|---|---|---:|
| `zero_delta` | `-1` | 1 |
| `train_mean` | `-1` | 1 |
| `source_prior` | `-1` | 1 |
| `nearest_neighbor` | `-1` | 1 |
| `ridge` | `-1` | 1 |
| `random_forest` | `20260624`, `20260724`, `20260801` | 3 |
| `xgboost` | `20260624`, `20260724`, `20260801` | 3 |
| `permuted_xgboost` | `20260803` | 1 |

The base OOF key universe must match membership exactly. Missing streams, extra streams, duplicate stream rows, or predictions attached to a different fold fail with an explicit error.

## Subsets and metrics

The script derives these four evaluation flags independently for every pair-by-split row:

| Subset | Definition |
|---|---|
| `all_oof` | Every fixed OOF membership row. |
| `no_unordered_identity_pair_in_training` | The unordered pair of endpoint identity hashes occurs zero times among the same split's other-fold primary pairs. |
| `both_endpoint_identities_absent_from_training` | Each endpoint identity hash occurs zero times among both endpoints of the same split's other-fold primary pairs. |
| `both_assay_ids_known` | Both endpoint assay tokens match the `RAWASSAY:<64 uppercase hex>` grammar. |

Filtering changes evaluation membership only. Predictions and folds remain fixed.

For each subset and model, all retained pair-by-split prediction rows from that model's saved streams are pooled directly. Three-stream models therefore contribute three prediction rows per retained OOF key; the script does not first average per-seed metrics.

- Direction accuracy is the mean of `(prediction != 0) and (sign(prediction) == sign(true_delta))`; a zero prediction is incorrect.
- MAE is the mean absolute error in pDC50.
- RMSE is the square root of the mean squared error in pDC50.

These are descriptive summaries. Subset differences do not prove the full workflow is leakage-free, establish independent-target or external-deployment generalization, isolate a causal effect of identity recurrence or assay missingness, or support inferential model comparisons.

## Outputs

The output directory receives three aggregate files by default:

| File | Contents |
|---|---|
| `metrics.csv` | One row per subset and model with model streams, base and prediction support, pooled direction accuracy, MAE, RMSE, and estimability status. |
| `support.csv` | Aggregate support at overall, split/fold, and model scopes, including unique-count and training-recurrence counts. No identifier values are emitted. |
| `RUN_METADATA.json` | Aggregate input row counts, fixed design, stream inventory, subset support, assay-token census, output list, and interpretation limits. It records no input paths or file hashes. |

With `--write-membership`, the script additionally writes `membership.csv`. That file is row-level and contains record, pair, component, POI, identity-hash, and assay-token values plus recurrence counts and subset flags. It is intended only for an authorized local audit and should remain untracked.

Validation failures exit with status 2 and name the broken contract, such as a missing column, duplicated key, incomplete endpoint join, unexpected fold or stream inventory, inconsistent redundant identifier, or reversed truth sign. No output should be interpreted unless the command finishes with a JSON summary whose `status` is `PASS`.
