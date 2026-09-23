# Licensing and source terms

The authors selected the [MIT License](../LICENSE) for the author-written code and its accompanying software documentation on 23 September 2026. This covers the reconstruction, model-fitting, statistical-analysis and verification code in this repository.

The repository also supplies author-generated predictions, split identifiers, resampling multiplicities and aggregate results so that readers can examine and recompute the reported analyses. These files contain no source SMILES or measured activities. Their identifiers bind rows across the analysis.

Third-party software, publications and data retain their own terms. In particular, the MIT software license does not grant new rights to TACK, PROTAC-DB or their upstream sources. The download script retrieves the fixed TACK snapshot directly from its public host; it does not mirror the source table in this repository. For the PROTAC-DB audit, each reader reviews the provider's terms and downloads the workbook from the official page; the repository scripts only verify, convert and analyze it locally. Provider metadata is not a substitute for permission to redistribute third-party data.

The PROTAC-DB workbook and row-level nonself-overlap derivatives are not redistributed. A reader with the exact matching workbook can run the documented local reconstruction, but the official endpoint is unversioned and the project has no durable archive of that workbook. These access, version and redistribution boundaries are described in [data access](data_access.md) and remain separate from the unified TACK-based model comparisons.

Figures 1–5 and Figure S1 are manuscript assets supplied for inspection. Figure 6 is not redistributed here; its molecular structures can be inspected in the corresponding source records after local reconstruction.
