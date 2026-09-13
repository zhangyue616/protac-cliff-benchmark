# Aggregate results

These files are reader-facing projections of the accepted local analysis outputs. They contain aggregate numbers only; no structures, SMILES, pair IDs, record IDs, source paths, credentials, or internal audit logs are included.

| File | Contents |
| --- | --- |
| `primary_contrasts.csv` | Six XGBoost-versus-random-forest contrasts from the primary 77-component, B = 2,000 bootstrap. |
| `target_sensitivity_contrasts.csv` | The same six point estimands with intervals from the post hoc 17-supported-target, B = 50,000 sensitivity. |
| `identity_metrics.csv` | Thirty-two descriptive rows: four fixed-OOF subsets by eight model/control identities. |
| `identity_support.csv` | Overall support counts for the four fixed-OOF subsets. |
| `representation_summary.json` | Aggregate support, parameters, and outcomes for the 16-pair-row representation audit. |

For both contrast tables, positive values favor XGBoost: direction accuracy uses XGBoost minus random forest, while MAE and RMSE use random forest minus XGBoost. The bounds are multiplicity-adjusted over the six-row family. In the primary component specification, only the two MAE intervals have positive lower bounds. In the target sensitivity, all six intervals include zero. The target sensitivity changes the resampling unit, sampling universe, and replicate count together, so the different interval conclusions cannot be attributed to target clustering alone.

The identity metrics pool retained prediction rows within each model. Random forest and XGBoost each have three saved streams, while the other model/control identities have one; metrics are not first averaged per seed. Subset changes are descriptive because support and POI composition change. They do not isolate effects of identity recurrence or recorded-assay completeness.

The representation diagnoses are hierarchical. `FOLDING_SUPPORTED` means the non-chiral sparse feature-ID supports differ despite equal 2,048-bit binary fingerprints. `MULTIPLICITY_LOSS_SUPPORTED` means those supports are equal but counts differ. `CHIRALITY_AWARE_ONLY_SEPARATION` means the non-chiral counts are equal and the chirality-aware counts differ. Sparse feature IDs remain hashed, and separation does not demonstrate predictive benefit or causality.

The scripts can regenerate selected aggregates only when the required, separately authorized inputs are available. See [`../docs/reproduction.md`](../docs/reproduction.md) and [`../docs/data_access.md`](../docs/data_access.md).
