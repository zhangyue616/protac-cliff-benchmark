# Evaluation from existing predictions

`scripts/evaluate_predictions.py` recomputes the paper's six fixed XGBoost-versus-random-forest point contrasts from an existing out-of-fold (OOF) prediction table. It does not derive benchmark records, build folds, fit models, or generate predictions.

The default command is:

```bash
python scripts/evaluate_predictions.py \
  --predictions ../private_inputs/predictions.csv \
  --output-dir ../local-results/evaluation
```

Paths are supplied at run time. The repository contains no prediction rows, molecular structures, record mappings, or machine-specific source paths.

## Prediction input

The CSV may contain additional columns and model streams. The evaluator reads these columns:

| Column | Meaning |
| --- | --- |
| `pair_id` | Stable cliff-pair identifier |
| `comparison_pair_key` | Key used for the equal-comparison-key estimand |
| `component_id` | Connected-component identifier used to retain the original aggregation order |
| `true_delta` | Nonzero observed ordered-pair difference in the prediction table's `j - i` orientation |
| `prediction` | Existing OOF predicted difference |
| `model` | Model identity; the evaluated values are `xgboost` and `random_forest` |
| `model_seed` | Estimator seed |
| `split_seed` | Split seed |
| `fold` | Test fold |

The script is intentionally tied to the frozen evaluation grid. It requires model seeds `20260624`, `20260724`, and `20260801`; split seeds `20260624`, `20260625`, `20260626`, `20260724`, and `20260801`; and folds 0–4. Each of the two model streams contains 13,110 rows across the three model seeds, or 4,370 rows per model seed. After pairing the streams, each model seed must also have 4,190 equal-comparison-key units. The script rejects duplicate or asymmetric OOF keys, cross-seed membership drift, model disagreements in truth or identifiers, blank identifiers, zero truth differences, and non-finite numeric values.

The evaluator assumes the supplied prediction table already uses the frozen `j - i` truth and prediction orientation. The source pair mapping records the opposite `i - j` convention, and this aggregate-only script does not receive enough source fields to verify orientation independently.

## Metric and contrast definitions

For each model seed, pair-level direction accuracy is the mean of

```text
(prediction != 0) and (sign(prediction) == sign(true_delta))
```

Pair-level MAE is the mean absolute error. Pair-level RMSE is the square root of the mean squared error over all 4,370 OOF pair rows for that model seed.

The equal-comparison-key calculation first averages direction indicators, absolute losses, and squared losses within each `(split_seed, model_seed, comparison_pair_key)` group. Those 4,190 key-level values are then weighted equally within each model seed. Equal-key RMSE takes the square root only after averaging the key-level mean squared losses. It is therefore not a mean of per-split or per-key RMSE values.

The six comparisons are the Cartesian product of two estimands (`pair_id` and `equal_comparison_key`) and three metrics (`direction_accuracy`, `delta_mae`, and `delta_rmse`). A positive contrast always favors XGBoost:

- direction accuracy: `xgboost - random_forest`
- MAE and RMSE: `random_forest - xgboost`

Each contrast is formed separately for each of the three model seeds. The reported point estimate is the arithmetic mean of those three seed contrasts. This order preserves the formal primary evaluation; the script does not replace it with a single metric pooled across model seeds or with a later identity-sensitivity subset calculation.

## Aggregate outputs

The default command writes three CSV files:

| File | Rows | Contents |
| --- | ---: | --- |
| `seed_model_metrics.csv` | 36 | Model metric and support for each seed, estimand, and metric |
| `seed_contrasts.csv` | 18 | Paired XGBoost-versus-random-forest contrast for each seed and family |
| `mean_contrasts.csv` | 6 | Arithmetic mean of the three seed contrasts |

No row-level prediction, pair, component, or target identifiers are written.

## Optional replay of the existing target plan

The portable script can apply the sealed supported-target multiplicity matrix. It cannot create a draw plan. Supply all three optional inputs together:

```bash
python scripts/evaluate_predictions.py \
  --predictions ../private_inputs/predictions.csv \
  --output-dir ../local-results/target_replay \
  --pair-target-map ../private_inputs/pair_target_map.csv \
  --key-target-map ../private_inputs/key_target_map.csv \
  --target-multiplicities ../private_inputs/target_multiplicities.npy
```

The pair map must contain `pair_id`, `comparison_pair_key`, `component_id`, `normalized_poi_id`, and `target_index`. The key map must contain `comparison_pair_key`, `component_id`, `normalized_poi_id`, and `target_index`. Extra columns are allowed. The maps remain local inputs and are not copied to the output directory.

The script requires the maps to cover exactly the evaluated pair and comparison-key universes; missing entries and unused extra entries are rejected. Within the bound pair universe and the bound equal-key universe separately, `normalized_poi_id` and `target_index` must form a two-way one-to-one mapping over exactly 17 targets. The two bound universes must have the same complete POI/index mapping, with indices 0–16 assigned in lexicographic `normalized_poi_id` order. The supplied NumPy matrix must be the sealed 50,000 × 17 `uint16` target-multiplicity artifact identified by the digest guard in `scripts/evaluate_predictions.py`; every row must sum to 17 tickets.

Matrix column `i` is interpreted as `target_index == i` in the two supplied maps. The script validates the matrix identity and the maps' internal agreement and lexical order, but it does not establish the provenance of those local mappings. Matrix identity alone does not prove that separately supplied maps are the frozen scientific mappings.

For each supplied replicate, one target-multiplicity row is applied synchronously to both models, both estimands, all three metrics, and all three model seeds. Loss sums and support counts are weighted before MAE or direction accuracy is formed; RMSE is the square root of the weighted mean squared error. The same contrast signs and three-seed arithmetic mean are then used. The six-row `target_plan_intervals.csv` reports the linear quantiles at `0.05 / (2 × 6)` and `1 - 0.05 / (2 × 6)`. Bounds are emitted only if all 50,000 supplied replicates are estimable across the complete six-comparison family.

This optional path replays an existing target plan. It does not generate a new target bootstrap or alter the frozen predictions.

## What is outside this script

This evaluator does not replay the primary 2,000 × 77 component family. The separate `scripts/recompute_primary_component_metrics.py` command can recompute that family's intervals from authorized saved predictions and the saved component plan. Those inputs are not included in the repository. The target-plan generator, target mappings, target multiplicity matrix, benchmark derivation, feature generation, training code, fitted models, and prediction tables are also absent. Consequently, the default point-estimate command does not reproduce the primary confidence intervals, and this repository alone is not a complete end-to-end reproduction of the study.

The repository environment requires Python 3.11 or newer and pins NumPy 2.3.5 and pandas 3.0.1. The evaluator does not require RDKit, scikit-learn, or XGBoost because it consumes predictions rather than producing them.
