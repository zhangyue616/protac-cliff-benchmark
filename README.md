# Predicting signed pDC50 differences in PROTAC activity-cliff pairs: a retrospective comparison of machine-learning models

This repository contains the runnable aggregate-analysis code, checked-in numerical results, and manuscript figures for a retrospective benchmark of signed pDC50 differences within **874 measured PROTAC activity-cliff pairs from 420 records**. The six-contrast family was fixed after early findings from the same data; its multiplicity-adjusted percentile intervals have uncalibrated coverage.

The task begins after both endpoint activities are known and a pair meets the operational cliff definition. It evaluates local signed-difference resolution; it does not evaluate prospective cliff discovery, compound ranking, target-held-out deployment, or mechanism prediction.

![Study workflow](figures/Figure1.png)

## Main findings

The primary family compares XGBoost with random forest under pair weighting and equal comparison-group weighting. Direction accuracy uses XGBoost minus random forest; MAE and RMSE use random forest minus XGBoost. Positive values therefore favor XGBoost in every row.

| Weighting | Metric | Contrast | Multiplicity-adjusted interval |
| --- | --- | ---: | ---: |
| Pair | Direction accuracy | 0.032 | [-0.043, 0.098] |
| Pair | MAE | 0.140 | [0.006, 0.330] |
| Pair | RMSE | 0.040 | [-0.107, 0.172] |
| Equal comparison group | Direction accuracy | 0.025 | [-0.044, 0.093] |
| Equal comparison group | MAE | 0.137 | [0.005, 0.336] |
| Equal comparison group | RMSE | 0.039 | [-0.107, 0.190] |

Only the two MAE intervals have positive lower bounds under the sealed 77-component, B = 2,000 analysis. All six post hoc 17-target intervals include zero. That sensitivity changes the resampling unit, sampling universe, and replicate count together, so it does not isolate target clustering as the cause of the interval change.

The two supplemental families are also bounded. All six global identity-disjoint XGBoost-versus-random-forest intervals include zero. Folded counts distinguish 14 of the 16 binary-fingerprint-equal pair records. In the separate 874-pair prediction comparison, all 18 count-versus-binary intervals include zero. XGBoost has worse point estimates on all three metrics under both weighting schemes. Zero-spanning intervals do not establish equivalence.

Full-precision values, support counts, and interpretation notes are in [results](results/).

## Repository map

| Path | Contents |
| --- | --- |
| [scripts/run_synthetic_demo.py](scripts/run_synthetic_demo.py) | Deterministic offline toy workflow for checking local dependencies and shared aggregation code. |
| [scripts/recompute_primary_component_metrics.py](scripts/recompute_primary_component_metrics.py) | Rescores frozen primary predictions with the saved 2,000 × 77 component plan. |
| [scripts/run_supplemental_e1.py](scripts/run_supplemental_e1.py) | Rescores frozen E1 identity-disjoint predictions with the saved 2,000 × 55 component plan. |
| [scripts/run_supplemental_e2.py](scripts/run_supplemental_e2.py) | Rescores paired folded-count and binary predictions with the saved 50,000 × 77 component plan. |
| [results](results/) | Checked-in aggregate tables at stored precision. |
| [figures](figures/) | Figures 1–5 and Figure S1 in reviewable formats. |
| [docs](docs/) | Reproduction guide, exact input contracts, data-access boundary, and licensing status. |

Figure 6 is not included because it contains molecular structures and record identifiers whose redistribution status has not been cleared.

## Install

Use Python 3.11 or newer in an isolated environment. On Windows PowerShell:

~~~bash
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
~~~

On POSIX shells:

~~~bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
~~~

The manuscript records separate original study environments for model fitting and RDKit-based structure analyses. Installing this repository's lightweight requirements does not recreate those environments or rerun model training.

In the commands below, `python` means the interpreter inside that environment: `.\.venv\Scripts\python.exe` on Windows or `.venv/bin/python` on POSIX.

## Run the offline synthetic demo

~~~bash
python scripts/run_synthetic_demo.py --output-dir demo_output
~~~

The demo generates deterministic toy inputs and exercises the shared aggregation, weighting, within-seed RMSE, and count/binary pairing code without downloading data. A successful run checks this toy software path. It does not run the fixed-study input contract.

**The demo is not a scientific reproduction.** It does not use the 874 study pairs, rebuild the dataset, construct the reported folds, train a model, recreate frozen predictions, or reproduce a manuscript result.

## Recompute from authorized study inputs

Keep restricted inputs and generated real-data outputs outside the Git worktree.

~~~bash
python scripts/recompute_primary_component_metrics.py --predictions <csv> --draw-plan <npz-or-csv> --output-dir <dir>
python scripts/run_supplemental_e1.py --predictions <csv> --draw-plan <npz-or-csv> --output-dir <dir>
python scripts/run_supplemental_e2.py --count-predictions <csv> --binary-predictions <csv> --draw-plan <npz-or-csv> --output-dir <dir>
~~~

These commands consume analysis-ready tables. They do not download source data, construct the 874-pair population, regenerate graph-constrained folds, fit models, or recreate omitted prediction streams. See [input contracts](docs/input_contracts.md) for required columns, fixed identifiers, aggregation order, and output schemas. See the [reproduction guide](docs/reproduction.md) for what each command does and does not verify.

## Additional bounded analyses

Three specialized tools remain available for narrower retained analyses:

| Tool | Boundary |
| --- | --- |
| [evaluate_predictions.py](scripts/evaluate_predictions.py) | Recomputes fixed-prediction point metrics and contrasts and can replay the separate saved 17-target plan. It does not reproduce the primary component plan or fit models. |
| [identity_sensitivity.py](scripts/identity_sensitivity.py) | Filters frozen OOF predictions into identity-recurrence and recorded-assay subsets without refitting or changing folds. |
| [representation_audit.py](scripts/representation_audit.py) | Regenerates the bounded 16-pair binary/sparse Morgan audit from authorized structures. It does not train a model or test predictive gain and requires the separate RDKit environment. |

Their commands and schemas are documented in [fixed-prediction evaluation](docs/evaluation.md), [identity sensitivity](docs/identity_sensitivity.md), and [representation audit](docs/representation_audit.md).

## Data and reproduction boundary

The study used a locally frozen 4,184-row TACK DC50 snapshot. A currently accessible fixed revision was later verified to match the retained bytes. That fixed commit is a recovery anchor; it does not identify the unknown revision or authoritative retrieval time of the original download, nor does it recover upstream row-level provenance.

The repository does not include the structures, row-level pair and record maps, split memberships, model-ready features, fitted estimators, complete prediction streams, or retained resampling inputs required for end-to-end third-party reproduction. The checked-in aggregate results remain inspectable, the synthetic path is runnable offline, and the real-data commands can recompute bounded summaries when separately authorized inputs are supplied. These are narrower claims than full study reproduction.

See [data access and provenance](docs/data_access.md) for the fixed source revision and omitted-input boundary.

## License and release status

No repository-wide software license has been selected. The repository has no tagged release, archive identifier, or publication DOI. Third-party and source-derived materials remain subject to their own terms; the presence of aggregate results or manuscript figures does not clear their underlying inputs for reuse. See [licensing status](docs/licensing.md).
