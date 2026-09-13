# Reproduction guide

This repository separates three questions that require different evidence:

1. Can a reader inspect the reported aggregate results? Yes; see [`../results/`](../results/).
2. Can the included scripts recompute selected analyses when authorized local inputs are supplied? Yes, within the contracts documented below.
3. Does this repository alone provide complete third-party reproduction of the study? No. The required prediction, membership, mapping, structure, and resampling-plan inputs are not distributed here, and no fresh-environment reproduction has been completed.

All commands below are run from the repository root and keep inputs and generated outputs outside the Git worktree.

## Environments

The base numerical environment used for the verified identity-sensitivity run was Python 3.12.14, NumPy 2.3.5, and pandas 3.0.1. Install the pinned Python packages into an isolated environment with:

```bash
python -m pip install -r requirements.txt
```

The representation audit has a separate matched environment: Python 3.11.15 with RDKit 2026.03.3. In an appropriate Python 3.11 environment, install:

```bash
python -m pip install -r requirements-rdkit.txt
```

These commands describe dependencies; this repository has not been validated by creating and testing a new environment from scratch.

## Fixed-prediction evaluation

[`scripts/evaluate_predictions.py`](../scripts/evaluate_predictions.py) recomputes point metrics and XGBoost-versus-random-forest contrasts from fixed saved predictions:

```bash
python scripts/evaluate_predictions.py --predictions ../private_inputs/predictions.csv --output-dir ../local-results/evaluation
```

The default outputs are `seed_model_metrics.csv` (36 rows), `seed_contrasts.csv` (18 rows), and `mean_contrasts.csv` (6 rows). This path does not train models, generate predictions, construct splits, generate a resampling plan, or reproduce the primary B = 2,000 by 77-component bootstrap intervals.

An optional branch can apply an already-existing 50,000 by 17 `uint16` target-multiplicity plan. All three optional inputs must be supplied together:

```bash
python scripts/evaluate_predictions.py --predictions ../private_inputs/predictions.csv --output-dir ../local-results/evaluation-target --pair-target-map ../private_inputs/pair_target_map.csv --key-target-map ../private_inputs/key_target_map.csv --target-multiplicities ../private_inputs/target_multiplicities.npy
```

That branch adds `target_plan_intervals.csv` with six rows. It replays the supplied plan; it does not create random draws. See [`evaluation.md`](evaluation.md) for the required columns, weighting, contrast signs, and plan contract.

## Fixed-OOF identity and recorded-assay sensitivity

[`scripts/identity_sensitivity.py`](../scripts/identity_sensitivity.py) filters frozen OOF predictions into four descriptive subsets without refitting:

```bash
python scripts/identity_sensitivity.py --predictions ../private_inputs/predictions.csv --membership ../private_inputs/primary_split_membership.csv --pair-map ../private_inputs/primary_pair_map.csv --records ../private_inputs/records.csv --output-dir ../local-results/identity
```

The default outputs are aggregate `metrics.csv`, `support.csv`, and `RUN_METADATA.json`. The optional `--write-membership` flag emits a row-level file with restricted identifiers and hashes; keep it outside the repository. See [`identity_sensitivity.md`](identity_sensitivity.md) for schemas and fixed-study checks.

This is the one analysis path executed during initial repository assembly. In the recorded Python 3.12.14 / NumPy 2.3.5 / pandas 3.0.1 environment, the run completed with 4,370 OOF membership rows and 52,440 saved-prediction rows. Its 32 model-by-subset rows contained 96 metric values identical to the frozen accepted aggregate CSV, and the four subsets' 24 retained support counts also matched exactly. No row-level membership output was requested, and all run outputs remained outside the repository.

## Representation audit

[`scripts/representation_audit.py`](../scripts/representation_audit.py) regenerates the fixed binary and sparse Morgan representations from authorized local structure data:

```bash
python scripts/representation_audit.py --pairs ../private_inputs/pairs.csv --records ../private_inputs/records.csv --primary-pair-map ../private_inputs/primary_pair_map.csv --output-dir ../local-results/representation_audit
```

It writes aggregate JSON and CSV only. See [`representation_audit.md`](representation_audit.md) for the three minimal input schemas and interpretation limits.

## Initial verification status

| Path | Verification during repository assembly |
| --- | --- |
| Identity/recorded-assay sensitivity | One representative scientific run passed and exactly matched the frozen aggregate metrics and support described above. |
| Fixed-prediction point metrics and contrasts | Source review and Python syntax validation only; no scientific run in this assembly. |
| Optional target-plan replay | Source review and Python syntax validation only; no replay in this assembly. |
| Representation audit | Source review and Python syntax validation only; no scientific run in this assembly. |
| Primary component-bootstrap intervals | Aggregate results included; the original B = 2,000 by 77-component bootstrap implementation is outside this repository. |
| Fresh-environment reproduction | Not performed. |

The first row validates a bounded code path against frozen aggregates. It does not supply the omitted inputs, validate a new installation, reproduce training, or close the complete-reproduction gap.
