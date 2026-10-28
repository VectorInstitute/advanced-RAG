#!/usr/bin/env python3
"""Ingest the FinanceBench filings into one section folder each.

Fetches the 84 filings the 150 questions cite (verified against their pinned
digests), converts each with Docling's table-structure model, and writes the
harness's folder layout: ``document.json``, ``ingestion.json``, and zero-based
``pages/pNNN.md``. Each parse is keyed by the source digest and the parser
version in the artifact cache, so a second run is a cache hit and a changed
parser rebuilds.

Nothing here is committed. The PDFs live in the corpus cache and the section
folders under the cache root, because the corpus is licensed CC BY-NC 4.0.

Needs the ``docling`` extra. Conversion runs on CPU and takes an hour or two for
all 84 filings, so split it across processes with ``--shard`` if you have cores.

Examples
--------
Ingest every filing, in two processes::

    $ python ingest_financebench.py --shard 0/2 &
    $ python ingest_financebench.py --shard 1/2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from aieng.advanced_rag.cache import TieredResultCache
from aieng.advanced_rag.cache.paths import root as cache_root
from aieng.advanced_rag.corpora.financebench import PDF_SHA256, fetch_financebench
from aieng.advanced_rag.ingestion import PARSER_VERSION, TableStorage, ingest_document
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


RI_ID = "ri3"


def default_root() -> Path:
    """Return where the section folders go, under the shared cache root."""
    return Path(cache_root()) / "ingestion" / "financebench"


def build_converter() -> DocumentConverter:
    """Return a Docling converter with table structure on and OCR off.

    Filings are born-digital PDFs, so OCR would only slow the run and add noise.
    """
    options = PdfPipelineOptions(do_ocr=False, do_table_structure=True)
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})


def parse_shard(text: str) -> tuple[int, int]:
    """Parse ``i/n`` into a zero-based shard index and the shard count."""
    index, _, count = text.partition("/")
    shard, total = int(index), int(count)
    if not 0 <= shard < total:
        raise argparse.ArgumentTypeError(f"shard {text!r} is not i/n with 0 <= i < n")
    return shard, total


def table_count(directory: Path) -> int:
    """Return how many tables a folder's ``document.json`` holds."""
    payload: dict[str, Any] = json.loads((directory / "document.json").read_text())
    return len(payload.get("tables", []))


def summarise(root: Path) -> dict[str, Any]:
    """Count what each ingested folder holds, from the files alone.

    Counts carry no filing text, so the result is safe to commit.
    """
    documents: dict[str, dict[str, int]] = {}
    for manifest_path in sorted(root.glob("*/ingestion.json")):
        manifest = json.loads(manifest_path.read_text())
        directory = manifest_path.parent
        documents[directory.name] = {
            "pages": len(manifest["pages"]),
            "tables": table_count(directory),
            "failures": len(manifest["failures"]),
        }
    return {
        "parser_version": PARSER_VERSION,
        "documents": len(documents),
        "pages": sum(d["pages"] for d in documents.values()),
        "tables": sum(d["tables"] for d in documents.values()),
        "failures": sum(d["failures"] for d in documents.values()),
        "per_document": documents,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Ingest the filings in this shard and print one summary line each."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--docs", nargs="*", default=None, help="document names")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard", type=parse_shard, default=(0, 1))
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument(
        "--summarise",
        type=Path,
        default=None,
        metavar="OUT",
        help="write per-document counts from the existing folders to OUT and exit",
    )
    args = parser.parse_args(argv)

    if args.summarise is not None:
        summary = summarise(args.root or default_root())
        args.summarise.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(
            f"{summary['documents']} documents, {summary['pages']} pages, "
            f"{summary['tables']} tables, {summary['failures']} failures"
        )
        return 0

    names = sorted(args.docs if args.docs else PDF_SHA256)
    shard, total = args.shard
    names = [name for i, name in enumerate(names) if i % total == shard]
    if args.limit is not None:
        names = names[: args.limit]

    root = args.root or default_root()
    paths = fetch_financebench(names)
    converter = build_converter()
    cache = TieredResultCache()

    def convert(path: Path) -> Any:
        return converter.convert(path).document

    began = time.perf_counter()
    totals = {"documents": 0, "pages": 0, "tables": 0, "failures": 0, "hits": 0}
    for name in names:
        started = time.perf_counter()
        result = ingest_document(
            paths.pdfs[name],
            root=root,
            convert=convert,
            cache=cache,
            ri_id=RI_ID,
            storage=TableStorage.MARKDOWN,
            doc_id=name,
            force_refresh=args.force_refresh,
        )
        tables = table_count(result.directory)
        totals["documents"] += 1
        totals["pages"] += len(result.pages)
        totals["tables"] += tables
        totals["failures"] += len(result.failures)
        totals["hits"] += int(result.cache_hit)
        print(
            f"{name}: {len(result.pages)} pages, {tables} tables, "
            f"{len(result.failures)} failures, "
            f"{'cache hit' if result.cache_hit else 'parsed'} "
            f"in {time.perf_counter() - started:.1f}s"
        )
        for failure in result.failures:
            print(f"  failure: {failure}")
    print(
        f"shard {shard}/{total}: {totals['documents']} documents, "
        f"{totals['pages']} pages, {totals['tables']} tables, "
        f"{totals['failures']} failures, {totals['hits']} cache hits, "
        f"{time.perf_counter() - began:.0f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
