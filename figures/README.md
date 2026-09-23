# Figure assets

These are the figure files used by the current manuscript. Figures 1–5 are available as publication PNGs and vector sources; Figure S1 is the raster image used in the Supporting Information.

| Figure | Available files | What the figure shows |
| --- | --- | --- |
| Figure 1 | [PNG](Figure1.png), [SVG](Figure1.svg), [PDF](Figure1.pdf) | Retrospective signed-difference workflow: graph-constrained folds, measured-cliff selection, signed fingerprint inputs, held-out scoring, and the two supplemental refit families. |
| Figure 2 | [PNG](Figure2.png), [SVG](Figure2.svg), [PDF](Figure2.pdf) | Descriptive out-of-fold absolute-error distributions and pooled DA, MAE, and RMSE for eight models and controls. The rows contain repeated split and model-seed predictions and do not establish a universal ranking. |
| Figure 3 | [PNG](Figure3.png), [SVG](Figure3.svg), [PDF](Figure3.pdf) | Saved component-bootstrap distributions and multiplicity-adjusted intervals for the six XGBoost–random-forest contrasts. DA is XGBoost minus random forest; MAE and RMSE are random forest minus XGBoost, so positive values favor XGBoost. See the [evaluation notes](../docs/evaluation.md). |
| Figure 4 | [PNG](Figure4.png), [SVG](Figure4.svg), [PDF](Figure4.pdf) | Primary-population accounting, overlap-policy retention, and overlapping population groups. The OOF-support bins are `<0.5`, `0.5–<0.6`, `0.6–<0.7`, `0.7–<0.8`, and `≥0.8`, with 3,919, 87, 6, 20, and 338 rows. Retention and support counts are not performance estimates. |
| Figure 5 | [PNG](Figure5.png), [SVG](Figure5.svg), [PDF](Figure5.pdf) | Two separate families: six identity-disjoint XGBoost-versus-random-forest contrasts from 2,000 draws over 55 components, and 18 folded-count-versus-binary contrasts from 50,000 draws over 77 components. Pair and equal-group weights use the same frozen predictions within each model or representation. The multiplicity-adjusted, two-sided percentile intervals are uncalibrated; all 24 span zero, which does not establish equivalence. Exact aggregate values are in [supplemental_contrasts.csv](../results/supplemental_contrasts.csv). |
| Figure S1 | [PNG](FigureS1.png) | Empirical distribution of binary Morgan-fingerprint Tanimoto similarity for the 874 measured cliff pairs. The final SI contains this raster image; a matching final vector source is not included here. |

The Figure 1 and Figure 3 PNG resolution metadata is set to 600 dpi. This changes only the PNG `pHYs` chunk; compressed image data and pixels are unchanged. The Figure 3 SVG and PDF retain the final vector crop repair, including the complete right-hand labels.

Figure 6 is intentionally absent. It contains third-party molecular structures and record-level annotations whose redistribution rights have not been closed. Its omission does not remove any aggregate result used by Figures 1–5 or Figure S1.
