# Data access and provenance

## Fixed public source

The TACK-based model comparisons start from a 4,184-row DC50 table available at a [fixed Hugging Face revision](https://huggingface.co/datasets/ailab-bio/TACK/resolve/0ebfb3627cfeee826c3392586d46940402237b11/DC50/train-00000-of-00001.parquet). `scripts/fetch_source.py` downloads that file and checks its bytes against the retained study input.

This revision was identified after the original analysis as an exact byte match. It does not recover the original download revision or time, upstream curation decisions, or patent/table-copy lineage. Public access is not a blanket redistribution license for upstream sources.

The source table is downloaded to the reader's output directory and is not mirrored in Git. `reproduce_construction.py` supplies the actual eligibility, canonicalization, context, pair-selection, graph and fold-assignment rules. It reconstructs molecular structures, activities, metadata, comparison groups, record/pair maps and model features locally. Eligibility uses the stored concentration/operator fields; it does not substitute inequality bounds or convert other units.

## Author-generated artifacts

[reproduction/](../reproduction/) contains the saved model predictions and sampling multiplicities. The prediction CSVs omit source structures, source activity measurements and source text. They retain author-generated identifiers, predictions and split annotations. `prepare_frozen_predictions.py` obtains their signed targets from reconstructed records as pDC50(j) minus pDC50(i), rather than copying the oppositely oriented legacy pair-table field.

Long identifiers in these files bind rows, groups and components across the computation. They are necessary analysis keys, not an additional layer of release checksums. Human-facing documentation avoids duplicating file digests; the downloader retains the one required fixed-source integrity check.

## Historical source diagnostics

The reported nonself-overlap diagnostic used three additional local anchors: a 15,502-row PROTAC-DB-derived table, a 1,429-row external-record derivative and a 17-row Wurz-family structure ledger. All three molecular-graph anchors derive from PROTAC-DB. They are not included, and the TACK-based commands do not substitute TACK for these nonself anchors.

On 23 September 2026, the file length and ETag at the official [PROTAC-DB download endpoint](https://cadd.zju.edu.cn/protacdb/downloads) matched the retained workbook's length and SHA-512. This was a header check, not a fresh download and byte comparison. The endpoint is not versioned. The provider requires acceptance of its access agreement and restricts redistribution of both raw and derivative data. A matching manual download may therefore be obtainable now, but a durable fixed-version link or permission to archive the snapshot remains to be secured. No agreement is accepted on the reader's behalf.

This is a remaining reproducibility boundary for that source-overlap diagnostic. Publishing the model code does not close it. The missing original draw-order/probability-vector records and upstream retrieval history are separate provenance limits; no script fabricates them.

If you independently obtain the matching workbook under the provider's terms, `scripts/prepare_protacdb.py --input-xlsx <file> --output-dir <local-directory>` verifies its fixed hash and converts it locally. It neither downloads the file nor accepts an agreement. The workbook and generated CSV must stay outside the repository. The conversion was checked with Python 3.12.14, pandas 3.0.1 and openpyxl 3.1.5; optional conversion dependencies are in [environment/protacdb-requirements.txt](../environment/protacdb-requirements.txt). A different snapshot is rejected rather than silently substituted.

Using the model-fitting environment, the conditional local audit is:

~~~bash
python scripts/audit_source_overlap.py --construction-dir <construction> --protacdb-csv <local-protac.csv> --predictions <primary-predictions.csv> --output-dir <local-audit>
~~~

It rebuilds the PROTAC-DB structure-overlap membership and rescores the retained 589-pair subset. With the author's retained workbook, all 874 keep/exclude decisions matched the original nonself union. The external and Wurz anchors added no pair outside that union, but this script does not recreate their separate labels, DOI matches or manual adjudications. This local verification does not establish access for another reader. Its row-level output must also remain local unless the provider grants redistribution permission.

## Figures

Figures 1–5 and Figure S1 are included as manuscript assets. Figure 6 is not redistributed because source-structure redistribution has not been cleared. Its cases can be examined after local reconstruction, and the bounded full-graph mapping is reproduced by `audit_structure.py`. The unified command does not rebuild final figure layouts.

See [reproduction coverage](reproduction.md) for the analyses actually executed and [source terms](licensing.md) for the software/data distinction.
