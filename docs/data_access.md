# Data access and provenance

This repository distributes code, aggregate result tables, and manuscript figures. It does not distribute the row-level scientific inputs used for the reported analyses.

## Source snapshot

The study started from a locally frozen 4,184-row TACK DC50 table. During manuscript preparation, the retained bytes were matched to this fixed Hugging Face revision:

[Fixed TACK DC50 snapshot](https://huggingface.co/datasets/ailab-bio/TACK/resolve/0ebfb3627cfeee826c3392586d46940402237b11/DC50/train-00000-of-00001.parquet)

That fixed revision is a recoverable byte-matching anchor. It does not recover the unknown revision or authoritative retrieval time of the original download, and it does not reconstruct upstream row-level curation, patent or table-copy lineage, or source-database snapshots that preceded the frozen input.

## Inputs omitted from this repository

The real-data recomputation commands require one or more authorized local tables containing analysis-ready rows. Depending on the command, these may include:

- fixed out-of-fold predictions and true signed pDC50 differences;
- pair, comparison-group, component, split, fold, model, model-seed, and representation labels;
- retained component-resampling values or multiplicities;
- record-to-identity, target, and recorded-assay mappings;
- molecular structures or canonical-isomeric SMILES.

The exact columns accepted by each public command are listed in [input contracts](input_contracts.md). No script downloads, reconstructs, or fabricates a missing scientific input.

These omitted files can contain structures, source-derived fields, stable identifiers, membership information, or hashes that remain subject to source-specific rights. If access is granted separately, keep inputs and generated real-data outputs outside the Git worktree, for example:

```text
../private_inputs/
../local-results/
```

The repository's synthetic demo uses generated toy rows and does not require these inputs. Its output is not a substitute for the omitted study data.

## Figure boundary

Figures 1–5 and Figure S1 are included as manuscript-level assets. Figure 6 is omitted because it contains molecular structures and record identifiers whose redistribution status has not been cleared. The omission does not change the aggregate numerical results in [`../results/`](../results/).

## What the repository can establish

The checked-in aggregate tables permit inspection of the reported numerical conclusions. With separately authorized analysis-ready inputs, the included commands can recompute the documented aggregate summaries. Complete third-party reproduction would additionally require the omitted inputs, source-specific redistribution clearance, the original data-construction and model-fitting workflow, matched environments, and a fresh end-to-end run.
