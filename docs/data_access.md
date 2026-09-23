# Data access and provenance

## Fixed public source

The TACK-based model comparisons start from a 4,184-row DC50 table available at a [fixed Hugging Face revision](https://huggingface.co/datasets/ailab-bio/TACK/resolve/0ebfb3627cfeee826c3392586d46940402237b11/DC50/train-00000-of-00001.parquet). `scripts/fetch_source.py` downloads that file and checks its bytes against the retained study input.

This revision was identified after the original analysis as an exact byte match. It does not recover the original download revision or time, upstream curation decisions, or patent/table-copy lineage. Public access is not a blanket redistribution license for upstream sources.

The source table is downloaded to the reader's output directory and is not mirrored in Git. `reproduce_construction.py` supplies the actual eligibility, canonicalization, context, pair-selection, graph and fold-assignment rules. It reconstructs molecular structures, activities, metadata, comparison groups, record/pair maps and model features locally. Eligibility uses the stored concentration/operator fields; it does not substitute inequality bounds or convert other units.

## Author-generated artifacts

[reproduction/](../reproduction/) contains the saved model predictions and sampling multiplicities. The prediction CSVs omit source structures, source activity measurements and source text. They retain author-generated identifiers, predictions and split annotations. `prepare_frozen_predictions.py` obtains their signed targets from reconstructed records as pDC50(j) minus pDC50(i), rather than copying the oppositely oriented legacy pair-table field.

Long identifiers in these files bind rows, groups and components across the computation. They are necessary analysis keys, not an additional layer of release checksums. Human-facing documentation avoids duplicating file digests; the downloader retains the one required fixed-source integrity check.

## Historical source diagnostics

The reported nonself-overlap diagnostic used three additional local anchors: a 15,502-row PROTAC-DB-derived table, a 1,429-row external-record derivative and a 17-row Wurz-family structure ledger. All three molecular-graph anchors derive from PROTAC-DB. They are not included, and the unified TACK-based commands do not substitute TACK for these nonself anchors.

The reader acquisition route is the official [PROTAC-DB downloads page](https://cadd.zju.edu.cn/protacdb/downloads). Each reader must review and accept the provider's terms personally, download the PROTAC XLSX to a private path outside this repository, and keep both the workbook and row-level derivatives local. The repository scripts perform no network request and accept no agreement. They provide fixed-version validation, conversion, overlap reconstruction and rescoring without distributing a mirror.

On 23 September 2026, a fresh HEAD request to the download endpoint returned status 200; its `Content-Length` of 6,268,531 bytes and ETag matched the retained workbook's length and SHA-512. This was a metadata check only: the response body was not downloaded or byte-verified. The endpoint is unversioned and may change, so every reader's workbook must pass the study-version SHA-256 check below. There is no project-controlled durable archive of the fixed workbook.

This reader-download route does not require the project to mirror or archive PROTAC-DB. Permission would still be required before redistributing the workbook or its row-level derivatives. Exact future availability remains dependent on the unversioned provider endpoint. The missing original draw-order/probability-vector records and upstream retrieval history are separate provenance limits; no script fabricates them.

After downloading, use a separate conversion environment to verify and convert the workbook:

~~~bash
python -m pip install -r environment/protacdb-requirements.txt
python scripts/prepare_protacdb.py --input-xlsx <downloaded-protac.xlsx> --output-dir <private-protacdb-dir>
~~~

`prepare_protacdb.py` requires the fixed workbook length of 6,268,531 bytes and SHA-256 `4E3A7ECC74A24E26877D319B18937E2A81161A43B4BD1A2C6A1C2BAE0FCB263D`. It then applies the retained first-worksheet conversion and requires 15,502 rows, 89 columns and the fixed output CSV hash. A different or drifted snapshot is rejected rather than silently substituted. The conversion was checked with Python 3.12.14, pandas 3.0.1 and openpyxl 3.1.5.

After completing the cached TACK reproduction, use the model-fitting environment for the local source-overlap audit:

~~~bash
python scripts/audit_source_overlap.py --construction-dir <reproduction-output>/construction --protacdb-csv <private-protacdb-dir>/protac.csv --predictions <reproduction-output>/frozen/primary_predictions.csv --output-dir <private-audit-dir>
~~~

It rebuilds the PROTAC-DB structure-overlap membership, enforces the fixed 874-pair census and 589 retained / 285 excluded split, and rescores predictions. With the author's retained workbook, all 874 keep/exclude decisions matched the original nonself union. The external and Wurz anchors added no pair outside that union, but this script does not recreate their separate labels, DOI matches or manual adjudications. Each reader must complete the download and checksum validation independently. Row-level audit output must remain local unless the provider grants redistribution permission.

## Figures

Figures 1–5 and Figure S1 are included as manuscript assets. Figure 6 is not redistributed because source-structure redistribution has not been cleared. Its cases can be examined after local reconstruction, and the bounded full-graph mapping is reproduced by `audit_structure.py`. The unified command does not rebuild final figure layouts.

See [reproduction coverage](reproduction.md) for the analyses actually executed and [source terms](licensing.md) for the software/data distinction.
