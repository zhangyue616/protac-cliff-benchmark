# Released numerical artifacts

`predictions/` contains the saved model outputs for the primary, E1 and E2 analyses. The compressed CSV files retain predictions, analysis keys and split annotations. They omit source structures, measured activities and textual source fields. The primary cache also retains its calculated nearest-training support values and support bins.

| Cache | Rows | Saved streams |
| --- | ---: | ---: |
| primary.csv.gz | 52,440 | 12 |
| E1.csv.gz | 52,440 | 12 |
| E2.csv.gz | 30,590 | 7 |

Run `scripts/prepare_frozen_predictions.py` after data reconstruction to attach pDC50(j) minus pDC50(i) from the reconstructed records. Do not use the legacy `true_delta` field in the pair-construction table: its stored orientation is the reverse. This preparation also constructs the 17-target mapping used for the supported-target sensitivity.

`plans/` contains the saved resampling inputs:

| File | Shape | Meaning |
| --- | --- | --- |
| primary.npz | 2,000 × 77 | Original component multiplicities, including 37 primary-empty components. |
| E1.npz | 2,000 × 55 | Identity-disjoint component multiplicities, including 26 primary-empty components. |
| E2.npz | 50,000 × 77 | Synchronous count/binary component multiplicities. |
| target_multiplicities.npy | 50,000 × 17 | Separate supported-target sensitivity. |

The primary NPZ losslessly stores the retained long-form multiplicities. It contains no invented generator seed. The original ordered-draw and probability-vector records were not preserved. The E1 and E2 plans retain their recorded PCG64 seeds. None of these files establishes calibrated interval coverage or an analysis plan registered before data inspection.
