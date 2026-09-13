# PROTAC Cliff Benchmark

Code and aggregate evidence for stress-testing signed pDC50 differences within **874 already-observed operational PROTAC cliff pairs from 420 records**.

This private repository accompanies the unpublished working manuscript, *Recorded-context stress testing of whole-molecule models on operational pDC50 cliffs in PROTAC degraders*. The manuscript and repository do not yet have a publication DOI.

## Research question

Given a pair already known from measured values to satisfy an operational whole-molecule cliff definition, how well does a fixed chemistry-only representation estimate the signed local pDC50 difference, and how sensitive are the conclusions to weighting, dependence units, recurrent molecular identities, and representation compression?

This is an evaluation of local difference resolution after cliff membership is known. It is not a prospective cliff-discovery, compound-ranking, or mechanism-prediction system.

Pairs share a POI and an equality token built from recorded source, article, assay, cell, and E3 fields, with explicit missing sentinels. Equality of this token means that recorded fields and fallback values match. It does **not** show that the experiments used an identical or semantically equivalent protocol; equal missing sentinels can match.

## What is in this repository

| Path | Purpose |
| --- | --- |
| [`scripts/evaluate_predictions.py`](scripts/evaluate_predictions.py) | Aggregate metrics and fixed XGBoost-versus-random-forest point contrasts from saved predictions; optional replay of an existing target multiplicity plan. |
| [`scripts/identity_sensitivity.py`](scripts/identity_sensitivity.py) | Four fixed-OOF identity and recorded-assay subsets, evaluated without refitting. |
| [`scripts/representation_audit.py`](scripts/representation_audit.py) | Binary/sparse Morgan comparison for the 16 selected binary-equal primary pair rows. |
| [`results/`](results/) | Reader-facing aggregate result projections. |
| [`docs/`](docs/) | Exact input contracts, commands, data-access limits, reproduction status, and licensing status. |

No source structures, SMILES, pair IDs, record IDs, prediction rows, split memberships, resampling plans, credentials, or internal review logs are included.

## Main numerical findings

The primary fixed family compares XGBoost with random forest under pair-row and equal-comparison-key estimands. Positive contrasts favor XGBoost: accuracy is XGBoost minus random forest; MAE and RMSE are random forest minus XGBoost.

| Estimand | Metric | Point estimate | Primary adjusted interval, B = 2,000 |
| --- | --- | ---: | ---: |
| Pair row | Direction accuracy | 0.032265 | [-0.042964, 0.098007] |
| Pair row | MAE | 0.139883 | [0.005703, 0.329848] |
| Pair row | RMSE | 0.040420 | [-0.106584, 0.171994] |
| Equal comparison key | Direction accuracy | 0.025219 | [-0.043766, 0.092955] |
| Equal comparison key | MAE | 0.137045 | [0.004724, 0.336475] |
| Equal comparison key | RMSE | 0.038659 | [-0.106875, 0.189682] |

Only the two MAE intervals have positive adjusted lower bounds under the primary 77-component specification. In the post hoc 17-supported-target, B = 50,000 sensitivity, all six adjusted intervals include zero. The two analyses change the resampling unit, sampling universe, and replicate count together, so the different interval conclusions cannot be attributed to target clustering alone.

Across the 4,370 frozen OOF pair-by-split rows, ridge had the strongest descriptive absolute metrics among the fitted models (direction accuracy 0.655606, MAE 1.104988, RMSE 1.294739), followed in direction accuracy by XGBoost at 0.638673. Ridge was not part of the fixed inferential family, so these values do not establish inferential superiority.

The fixed-prediction identity analysis retained 4,038 rows after excluding unordered molecular-identity pairs present in the corresponding training folds and 3,771 rows when both endpoint identities had to be absent from training. POI support fell from 17 to 14 and 13, respectively. Ridge direction accuracy changed from 0.655606 to 0.632244 and 0.623177; XGBoost changed from 0.638673 to 0.616147 and 0.607178. These are descriptive subset changes, not causal effects of recurrence. A separate recorded-assay-known subset retained 2,985 rows across seven POIs; it does not isolate an effect of assay missingness.

All 16 selected primary pair rows were equal under the non-chiral radius-2, 2,048-bit binary Morgan fingerprint and represented 13 unordered identity pairs. Non-folded sparse feature-ID support distinguished 6 rows; sparse counts distinguished another 8; chirality-aware sparse counts distinguished the remaining 2 rows, which were one identity pair observed across two POIs. Sparse feature IDs remain hashed. Separability under another representation does not imply improved prediction or identify the cause of an activity difference.

Full-precision values and support are in [`results/`](results/).

## Running the included analyses

The required inputs are intentionally external to the repository. The examples assume authorized files in `../private_inputs/` and write outputs to `../local-results/`.

Install the base numerical dependencies in an isolated environment:

```bash
python -m pip install -r requirements.txt
```

Recompute aggregate point metrics and contrasts from fixed predictions:

```bash
python scripts/evaluate_predictions.py --predictions ../private_inputs/predictions.csv --output-dir ../local-results/evaluation
```

Recompute fixed-OOF identity and recorded-assay sensitivity:

```bash
python scripts/identity_sensitivity.py --predictions ../private_inputs/predictions.csv --membership ../private_inputs/primary_split_membership.csv --pair-map ../private_inputs/primary_pair_map.csv --records ../private_inputs/records.csv --output-dir ../local-results/identity
```

For the representation audit, use the separate matched Python 3.11.15 / RDKit 2026.03.3 environment and install `requirements-rdkit.txt`, then run:

```bash
python scripts/representation_audit.py --pairs ../private_inputs/pairs.csv --records ../private_inputs/records.csv --primary-pair-map ../private_inputs/primary_pair_map.csv --output-dir ../local-results/representation_audit
```

The [reproduction guide](docs/reproduction.md) records the optional target-plan command and the exact verification completed during repository assembly. Each script's documentation gives its actual minimum input columns and output schema.

## Data and reproduction status

The aggregate results are available, but the study inputs are not bundled because structure-containing and row-level derivatives have not been cleared for unrestricted redistribution. The locally used TACK-derived snapshot also lacks a preserved upstream tag or commit and authoritative retrieval time. The included scripts do not download or fabricate replacements.

During repository assembly, the identity-sensitivity script was run once against the authorized frozen inputs. It reproduced all 96 stored metric values and all 24 projected support counts exactly in the recorded Python 3.12.14 / NumPy 2.3.5 / pandas 3.0.1 environment. The fixed-prediction evaluator, target-plan replay, and representation audit received source and syntax review but were not scientifically rerun. No new environment was installed or tested.

These facts support bounded local verification, not complete third-party reproduction. See [data access](docs/data_access.md) and [licensing status](docs/licensing.md).

## Related work and citation

- [TACK preprint, arXiv:2605.19579](https://arxiv.org/abs/2605.19579) describes a neighboring PROTAC endpoint-prediction resource and task.
- [Dablander et al., *Journal of Cheminformatics* 15, 47 (2023)](https://doi.org/10.1186/s13321-023-00708-w) provides a direct pair-level activity-cliff precedent.
- [MoleculeACE, *Journal of Chemical Information and Modeling* 62, 5938–5951 (2022)](https://doi.org/10.1021/acs.jcim.2c01073) is a broader activity-cliff benchmark.

There is no tagged release, archive identifier, or publication DOI for this repository. For private review, cite the repository URL and the exact Git commit inspected: `https://github.com/zhangyue616/protac-cliff-benchmark`.

## Use of AI tools

OpenAI Codex assisted with computational experimental design and protocol development, data processing, writing and executing analysis code, implementing and executing model training and prediction, statistical and sensitivity analyses, numerical verification, figure preparation, and writing and editing. Anthropic Claude assisted with review and editing of methods, code, and manuscript text. These tools did not generate the underlying experimental measurements; those measurements came from the cited source resources, and no new wet-laboratory experiments were performed. Responsibility for the study and reported content remains with the human authors.

No repository-wide software license has been granted yet. See [`docs/licensing.md`](docs/licensing.md).
