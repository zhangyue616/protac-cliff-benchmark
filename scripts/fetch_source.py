#!/usr/bin/env python3
"""Download the publicly hosted, fixed TACK DC50 snapshot used for reproduction.

This is a recovered byte-matching source anchor. It does not establish the
date or revision of the study's original download. Source terms remain with
the source provider; the repository's software license does not relicense data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

URL = (
    "https://huggingface.co/datasets/ailab-bio/TACK/resolve/"
    "0ebfb3627cfeee826c3392586d46940402237b11/"
    "DC50/train-00000-of-00001.parquet"
)
SHA256 = "2893ce64fe1d5ad50f6bf97aa63fde9597d11e6a1b388cfe30a7cfc79bb06875"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.resolve()
    if destination.exists():
        if digest(destination) != SHA256:
            raise RuntimeError("Existing file differs from the fixed source; it was not overwritten.")
        print("Fixed source already present and verified.")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    if partial.exists():
        raise RuntimeError("Partial download exists; inspect it before starting another attempt.")
    request = urllib.request.Request(URL, headers={"User-Agent": "protac-cliff-benchmark-reproduction/1"})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("xb") as output:
        for block in iter(lambda: response.read(1024 * 1024), b""):
            output.write(block)
    if digest(partial) != SHA256:
        raise RuntimeError("Downloaded bytes differ from the fixed input; partial file retained for inspection.")
    partial.rename(destination)
    receipt = {"source_url": URL, "sha256": SHA256,
               "original_download_revision_recovered": False,
               "status": "VERIFIED_FIXED_SOURCE_BYTES"}
    destination.with_suffix(destination.suffix + ".source.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print("Downloaded and verified the fixed TACK source.")


if __name__ == "__main__":
    main()
