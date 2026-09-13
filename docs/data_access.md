# Data access

This repository intentionally contains code and aggregate numerical results, not the structure-containing or row-level study inputs.

## Inputs that are not distributed here

The analyses require some or all of the following local tables:

- fixed out-of-fold prediction streams;
- primary pair and split-membership maps;
- record-to-identity and recorded-assay-token mappings;
- the compatible-pair table used by the representation audit;
- canonical-isomeric SMILES for the selected endpoint records;
- optional target mappings and the pre-existing target multiplicity plan.

These inputs can contain structures, source-derived fields, pair and record identifiers, identity hashes, assay-text hashes, or detailed memberships. They have not been cleared for unrestricted redistribution. The locally used TACK-derived snapshot also lacks a preserved upstream tag or commit and authoritative retrieval time, so this repository does not present it as a uniquely recoverable public source snapshot.

## What is available

The [`results/`](../results/) directory contains aggregate projections sufficient to inspect the reported numerical conclusions. The scripts document the exact columns required to recompute their respective outputs from authorized local inputs. No script downloads or fabricates missing data.

Use directories outside the repository for restricted inputs and generated outputs, for example:

```text
../private_inputs/
../local-results/
```

The relative names are conventions only; no example data are bundled. If access is granted separately, the recipient remains responsible for following the terms of the original data sources and any conditions attached to the supplied derivative files.

## Reproduction boundary

Code availability and aggregate-result availability do not by themselves provide complete third-party reproduction. That would additionally require authorized input delivery, source-specific redistribution clearance, a matched runtime, and a fresh-environment run. None of those broader conditions is implied by this private repository.
