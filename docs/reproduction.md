# Reproduction guide

The repository exposes four distinct levels of checking. They answer different questions and should not be described as interchangeable.

| Level | Runnable from this repository alone? | What it checks |
| --- | --- | --- |
| Inspect checked-in results and figures | Yes | The reported aggregate values, directions, support summaries, and manuscript visual assets. |
| Run the synthetic demo | Yes | Local dependencies and shared aggregation, weighting, RMSE, and pairing code on deterministic toy data. |
| Recompute from study inputs | Only with separately authorized inputs | The documented aggregate summaries from analysis-ready frozen predictions or resampling rows. |
| Reproduce the study end to end | No | Dataset construction, pair selection, graph-constrained folds, model fitting, prediction generation, retained draw generation, and all source-level provenance. |

## Environment

Use Python 3.11 or newer. Create an isolated environment from the repository root. On Windows PowerShell:

~~~bash
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
~~~

On POSIX shells:

~~~bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
~~~

The public aggregate commands use the lightweight NumPy/pandas environment in requirements.txt. The manuscript's original model-fitting and RDKit structure-analysis environments are separate historical environments; they are not recreated by this installation.

In the commands below, `python` means the interpreter inside that environment: `.\.venv\Scripts\python.exe` on Windows or `.venv/bin/python` on POSIX.

## Offline synthetic demo

~~~bash
python scripts/run_synthetic_demo.py --output-dir demo_output
~~~

The command creates deterministic toy inputs, calls the shared aggregation and pairing functions directly, and writes its outputs beneath demo_output. It does not execute the three fixed-study CLIs, access the network, or read any omitted study file.

A passing demo supports only the shared software path it exercises. It does not validate the fixed-study input contracts. The generated rows are synthetic, the support is intentionally small, and the output values are not expected to match any checked-in scientific result.

## Primary component summary

~~~bash
python scripts/recompute_primary_component_metrics.py --predictions <csv> --draw-plan <npz-or-csv> --output-dir <dir>
~~~

This command recomputes the six primary contrast estimates and multiplicity-adjusted interval bounds from a supplied analysis-ready prediction CSV and saved draw plan. It does not generate the 77-component plan, reconstruct missing ordered draws, infer the original probability law, or establish that the primary plan was fixed before results were inspected.

The scientific primary analysis used B = 2,000 resamples of all 77 components, including 37 components without primary-pair support. The six contrasts share synchronous weights and form one Bonferroni family. See [analysis definitions](analysis_definitions.md) and [input contracts](input_contracts.md).

## Supplemental E1: global identity-disjoint analysis

~~~bash
python scripts/run_supplemental_e1.py --predictions <csv> --draw-plan <npz-or-csv> --output-dir <dir>
~~~

The command aggregates supplied E1 prediction rows and recomputes the descriptive metrics and six XGBoost-versus-random-forest contrasts. It does not rebuild the identity-disjoint graph, assign folds, fit the 200 models, or generate predictions.

The reported E1 analysis contains 874 pairs across five split seeds. Its 55-component resampling container includes 29 supported and 26 primary-empty components. All six checked-in adjusted intervals include zero.

## Supplemental E2: folded count versus binary

~~~bash
python scripts/run_supplemental_e2.py --count-predictions <csv> --binary-predictions <csv> --draw-plan <npz-or-csv> --output-dir <dir>
~~~

The command aggregates supplied binary and folded-count prediction rows and recomputes descriptive metrics and the 18 representation contrasts. It does not calculate fingerprints from structures, fit the 175 models, or regenerate predictions.

The reported E2 analysis retains the original 874 pairs and folds and resamples the 77-component container for B = 50,000 replicates. All 18 checked-in adjusted intervals include zero.

## Additional bounded analyses

The repository retains three specialized commands alongside the main rescoring path.

### Fixed-prediction point metrics and target-plan replay

~~~bash
python scripts/evaluate_predictions.py --predictions <csv> --output-dir <dir>
~~~

This command calculates fixed-prediction point metrics and XGBoost-versus-random-forest contrasts. With a pair-target map, key-target map, and saved 50,000 × 17 target-multiplicity plan, it can also replay the post hoc supported-target intervals. It does not fit a model or reproduce the primary 77-component plan. See [fixed-prediction evaluation](evaluation.md).

### Frozen-prediction identity and assay subsets

~~~bash
python scripts/identity_sensitivity.py --predictions <csv> --membership <csv> --pair-map <csv> --records <csv> --output-dir <dir>
~~~

This command filters frozen predictions into four descriptive subsets. It neither refits models nor changes folds, and subset differences do not isolate a causal identity or assay effect. See [identity sensitivity](identity_sensitivity.md).

### Bounded representation audit

Install the matched RDKit dependency in a Python 3.11 environment:

~~~bash
python -m pip install -r requirements-rdkit.txt
~~~

Then run:

~~~bash
python scripts/representation_audit.py --pairs <csv> --records <csv> --primary-pair-map <csv> --output-dir <dir>
~~~

This command regenerates binary and sparse Morgan representations for the fixed 16-pair audit from authorized structures. It performs no fit, prediction, or bootstrap and does not test predictive benefit. See [representation audit](representation_audit.md).

## Inputs and outputs

The three main fixed-study rescoring commands validate the conditions listed for their actual implementations; their accepted columns, fixed sets, row identities, output files, and ordering are documented in [input contracts](input_contracts.md). The three additional tools have their own narrower contracts in [fixed-prediction evaluation](evaluation.md), [identity sensitivity](identity_sensitivity.md), and [representation audit](representation_audit.md). Do not interpret a partial output from a command that exits with an error.

Keep restricted inputs and generated outputs outside the repository, for example:

~~~text
../private_inputs/
../local-results/
~~~

No command downloads or fabricates a missing study input.

## Result comparison

For a real-data verification, compare the generated aggregate CSVs with the corresponding checked-in tables in [results](../results/). Match rows by their semantic keys before comparing numerical fields; file order alone is not a scientific identity check.

The checked-in primary, target-sensitivity, fixed-OOF subset, representation-audit, E1, and E2 tables come from accepted local analysis outputs. A matching aggregate recomputation validates the supplied analysis-ready inputs and aggregation path. It does not validate earlier data construction, model fitting, source provenance, redistribution rights, or a fresh end-to-end installation.
