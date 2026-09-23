# Aggregate results

These files are reader-facing projections of the accepted local analysis outputs. They contain aggregate numbers only: no molecular structures, SMILES, pair or record identifiers, source paths, credentials, or internal review logs are included.

| File | Contents |
| --- | --- |
| [primary_contrasts.csv](primary_contrasts.csv) | Six XGBoost-versus-random-forest contrasts from the primary 77-component, B = 2,000 analysis. |
| [target_sensitivity_contrasts.csv](target_sensitivity_contrasts.csv) | The same six point estimands with intervals from the post hoc 17-supported-target, B = 50,000 sensitivity. |
| [identity_metrics.csv](identity_metrics.csv) | Thirty-two descriptive rows: four frozen-OOF subsets by eight model or control identities. |
| [identity_support.csv](identity_support.csv) | Aggregate support for the four frozen-OOF subsets. |
| [representation_summary.json](representation_summary.json) | Support, parameters, and outcomes for the 16 binary-fingerprint-equal pair records. |
| [supplemental_contrasts.csv](supplemental_contrasts.csv) | Six E1 identity-disjoint contrasts and 18 E2 folded-count-versus-binary contrasts. |
| [supplemental_descriptive_metrics.csv](supplemental_descriptive_metrics.csv) | Descriptive E1 and E2 metrics under pair and equal-group weighting. |

## Direction and interval interpretation

In the primary and target-sensitivity tables, positive values favor XGBoost: direction accuracy is XGBoost minus random forest, while MAE and RMSE are random forest minus XGBoost. Only the two primary MAE intervals have positive lower bounds. All six target-sensitivity intervals include zero. The target analysis changes the resampling unit, sampling universe, and replicate count together, so the interval change cannot be attributed to target clustering alone.

In E1, positive values again favor XGBoost. Direction accuracy is XGBoost minus random forest; MAE and RMSE are random forest minus XGBoost. All six adjusted intervals include zero.

In E2, positive values favor folded counts. Direction accuracy is count minus binary; MAE and RMSE are binary minus count. All 18 adjusted intervals include zero. All six XGBoost point estimates across three metrics and two weighting schemes are negative, while the other model and metric combinations are mixed. These results do not establish equivalence or a universal representation ranking.

The checked-in intervals are multiplicity-adjusted percentile summaries with uncalibrated coverage. Replicate draws, split copies, and model-seed rows are not independent observations.

## Descriptive subsets and representation audit

The frozen-OOF identity table pools retained prediction rows within each model. Random forest and XGBoost have three saved streams; the other model or control identities have one. Subset changes are descriptive because support and POI composition change, and the predictions were not refit.

The representation diagnoses are hierarchical. FOLDING_SUPPORTED means non-chiral sparse feature-ID supports differ despite equal 2,048-bit binary fingerprints. MULTIPLICITY_LOSS_SUPPORTED means those supports are equal but counts differ. CHIRALITY_AWARE_ONLY_SEPARATION means non-chiral counts are equal and chirality-aware counts differ. Sparse feature IDs remain hashed, and separation does not demonstrate predictive benefit or causality.

See the [reproduction guide](../docs/reproduction.md) and [input contracts](../docs/input_contracts.md) for the computations that can be rerun from separately authorized analysis-ready inputs.
