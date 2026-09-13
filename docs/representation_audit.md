# Representation audit

This script reproduces the bounded post hoc audit of primary cliff-pair rows that are distinct molecular identities but have source Morgan Tanimoto exactly equal to 1. It does not fit a model, generate predictions, recompute benchmark metrics, or perform bootstrap analysis.

## Scientific definition

For each selected endpoint, the script regenerates the following RDKit representations:

1. A non-chiral binary Morgan fingerprint with radius 2 and 2,048 bits. The legacy `AllChem.GetMorganFingerprintAsBitVect` result is checked against `rdFingerprintGenerator.GetMorganGenerator(...).GetFingerprint`.
2. The support (feature-ID set) and counts from the matching non-chiral sparse Morgan count fingerprint. These feature IDs are not folded to 2,048 bins.
3. Sparse Morgan counts with the same settings except `includeChirality=True`.

Each pair receives the first applicable descriptive diagnosis:

- `FOLDING_SUPPORTED`: the binary fingerprints are equal but the sparse non-chiral feature-ID sets differ.
- `MULTIPLICITY_LOSS_SUPPORTED`: the sparse non-chiral feature-ID sets are equal but their counts differ.
- `CHIRALITY_AWARE_ONLY_SEPARATION`: the non-chiral sparse counts are equal but the chirality-aware sparse counts differ.
- `UNRESOLVED_AT_TESTED_REPRESENTATIONS`: none of the tested representations separates the pair.

The intended benchmark inputs yield 16 selected pair rows, 25 endpoint records, and 13 unordered molecular-identity pairs. Their pair-row diagnoses are 6 folding-supported, 8 multiplicity-loss-supported, 2 chirality-aware-only, and 0 unresolved. The two chirality-aware-only rows represent one molecular-identity pair observed across two POIs. These are semantic reproduction checks; the portable script does not use frozen file digests or private source paths.

## Runtime

The portable script uses Python 3.11-compatible syntax. The original audit environment was specifically Python 3.11.15 with RDKit 2026.03.3. RDKit behavior can change between releases, so merely running on a later version is not evidence of a matched-environment reproduction.

## Input schema

The three inputs are local CSV files. They are not included in this repository because the record table contains molecular structures and redistribution remains subject to data access and licensing decisions. Extra columns are allowed and ignored.

`--pairs` requires:

| Column | Meaning |
| --- | --- |
| `pair_id` | Unique pair-row identifier. |
| `record_id_i`, `record_id_j` | Endpoint record identifiers that join to `--records`. |
| `identity_hash_i`, `identity_hash_j` | Endpoint molecular-identity hashes. |
| `normalized_poi_id` | Normalized POI identifier used only for the aggregate cross-POI check. |
| `tanimoto` | Source reference-binary Tanimoto; selection uses exact decimal equality to 1. |
| `is_cliff` | Boolean (`True`/`False` or `1`/`0`). |

`--records` requires:

| Column | Meaning |
| --- | --- |
| `record_id` | Unique record identifier. |
| `canonical_isomeric_smiles_full` | RDKit-canonical isomeric SMILES used to regenerate representations. |
| `identity_hash` | Uppercase SHA-256 of the canonical-isomeric SMILES text. |

`--primary-pair-map` requires only `pair_id`. Membership defines the primary set. The script requires those IDs to equal the rows marked `is_cliff=True` in `--pairs`, then preserves primary-map row order when selecting the exact-Tanimoto-one rows.

## Command

From the repository root, replace the three input placeholders with authorized local files:

```bash
python scripts/representation_audit.py --pairs ../private_inputs/pairs.csv --records ../private_inputs/records.csv --primary-pair-map ../private_inputs/primary_pair_map.csv --output-dir ../local-results/representation_audit
```

The output directory is created if needed. The default outputs are:

- `representation_audit_summary.json`: structured parameters, aggregate counts, runtime versions, and interpretation boundaries.
- `representation_audit_summary.csv`: a long-form projection of numeric aggregate counts.

Neither output contains SMILES, record IDs, pair IDs, sparse feature IDs, input paths, credentials, or internal audit logs. The repository does not provide a record-level output mode.

## Interpretation limits

Sparse Morgan feature IDs are hashed and are not guaranteed collision-free. A changed representation making a pair distinguishable does not show that it improves prediction. Chirality-aware separation does not assign the activity difference to chirality, and none of the diagnoses establishes mechanism, transferability, causal SAR, or a named substructure effect.

Without authorized copies of all three input tables and the matched RDKit environment, this command cannot provide a complete third-party reproduction of the audit.
