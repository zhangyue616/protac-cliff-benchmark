# Analysis definitions

This page records the sign, weighting, aggregation, and interval conventions needed to interpret the checked-in result tables and the public recomputation commands.

## Prediction target and metrics

Stable outcome-blind endpoint order defines each pair as (i, j). Under the study contract, the prediction target is:

~~~text
true_delta = pDC50(j) - pDC50(i)
~~~

The fixed-study commands require nonzero, stream-consistent true_delta values. A prediction CSV alone cannot prove that its caller constructed them with the required j-minus-i orientation.

Direction accuracy is one when a nonzero prediction has the same sign as true_delta and zero otherwise. A zero prediction is counted as incorrect. MAE is mean absolute prediction error in pDC50 units. RMSE is the square root of mean squared prediction error in pDC50 units.

## Weighting

Pair weighting gives every directed pair row equal weight within the relevant stream.

Equal comparison-group weighting first averages row-level direction indicators, absolute errors, or squared errors within each comparison group. It then gives every group equal weight. RMSE takes the square root after the group-level mean squared error has been averaged.

The two weightings use the same frozen predictions. They differ only in aggregation.

## Model-seed aggregation

Metrics are calculated within a model-seed stream after pooling its five split copies. When a model has multiple saved seeds, stream-level metrics or contrasts are averaged with equal weight across those seeds. Rows from different seeds are not pooled before RMSE is calculated.

Random forest and XGBoost each have three saved model-seed streams in the primary and E1 analyses. Ridge and the other model or control identities have one. E2 compares binary and folded-count predictions within the same model and seed before averaging seed-level contrasts.

## Contrast directions

| Family | Metric | Stored contrast | Positive means |
| --- | --- | --- | --- |
| Primary, target sensitivity, E1 | Direction accuracy | XGBoost minus random forest | Higher XGBoost direction accuracy |
| Primary, target sensitivity, E1 | MAE or RMSE | Random forest minus XGBoost | Lower XGBoost error |
| E2 folded count versus binary | Direction accuracy | Count minus binary | Higher count direction accuracy |
| E2 folded count versus binary | MAE or RMSE | Binary minus count | Lower count error |

## Resampling families

### Primary family

The primary analysis uses B = 2,000 retained resamples of a 77-component container. Forty components contain primary pairs and 37 are primary-empty but remain eligible to be drawn. One multiplicity vector is applied synchronously across models, model seeds, weightings, metrics, and split copies. The six contrasts form one Bonferroni family that was fixed after early findings from the same data; the adjusted percentile intervals have uncalibrated coverage.

### Supported-target sensitivity

The post hoc sensitivity uses B = 50,000 resamples of the 17 targets represented by primary pairs. Target weights are shared across models, seeds, weightings, metrics, and splits. Pair and equal-group weighting are retained; target-level scores are not averaged. Relative to the primary analysis, the resampling unit, universe, and replicate count all change.

### E1: global identity-disjoint refitting

E1 uses predictions from refits on folds rebuilt after globally merging components that share endpoint molecular identities. It retains all 874 pairs and uses 55 components, 29 with primary-pair support and 26 primary-empty. The six XGBoost-versus-random-forest contrasts use B = 2,000 synchronous component resamples and form a separate Bonferroni family.

### E2: folded-count versus binary representation

E2 keeps the original 874 pairs and folds. It compares non-chiral radius-2, 2,048-position folded counts with the original binary representation for random forest, XGBoost, and ridge. The 18 contrasts use B = 50,000 synchronous resamples of the original 77-component container and form their own Bonferroni family.

## Interpretation boundary

The tables report multiplicity-adjusted two-sided percentile intervals whose coverage was not calibrated. Replicate draws, split copies, model-seed streams, and repeated appearances of a pair are not independent observations. An interval spanning zero permits either contrast direction under that analysis; it does not establish equivalence. Differences between analysis families were not directly tested and cannot be assigned to one changed design feature when several features changed together.
