"""Convert a document once, and write the section folder two RIs read.

The harness takes the converter as an injected callable, the way
``AdvancedRagSpine`` takes retrieval. ``docling`` itself is not a dependency of
this package: the real converter is ``pip install docling``, and it is injected
as a callable that returns a ``DoclingDocument``, usually built from
``DocumentConverter().convert(path).document``. The document *types* come from
``docling-core``, which the ``filings`` extra already installs and which
:mod:`aieng.advanced_rag.retrieval.chunkers` already imports. So a reference
implementation that only reads already-converted documents installs no PDF
model stack, and the harness is testable against committed Docling JSON without
one either.

The on-disk shape is the contract. Two reference implementations read it, table
reasoning and agent-native retrieval, so what :func:`ingest_document` writes
matters more than how it got there:

.. code-block:: text

    <root>/<DOC_ID>/document.json        the converted DoclingDocument
    <root>/<DOC_ID>/ingestion.json       the harness manifest
    <root>/<DOC_ID>/pages/p000.md        one file per page, in reading order
    <root>/<DOC_ID>/tables/t000.png      kept table images, when the setting
                                         keeps them

One folder per document with section files inside, rather than one file per
document, because the agent-native RI navigates pages as a directory listing
and the table RI opens single pages. ``document.json`` is the converter's own
export and nothing else, so a notebook loads it with
``DoclingDocument.model_validate``.

That last point is a deliberate departure from the earlier
``framework_lab/financebench.py``, which writes one ``document.json`` holding
``{"document": ..., "pages": ..., "docling_document": ...}``. Nesting the export
under a key means a reader cannot validate the file as a ``DoclingDocument``, so
the export and the harness's own manifest are separate files here. The page
files keep that code's zero-based numbering, because FinanceBench's page-level
gold evidence is zero-based and a 1-based name would put every page-level match
out by one.

Every parse is keyed by the source file's SHA-256 and the parser version
through :class:`~aieng.advanced_rag.cache.TieredResultCache`. A changed source
or a bumped :data:`PARSER_VERSION` is a different key, so it rebuilds rather
than serving output the current code would not write. An unchanged pair is a hit
and is reported to the meter, which keeps a warm re-run comparable with the cold
run it repeats. The cached payload is a tar of the
document folder, because the cache stores one opaque file per key and the
folder is what has to come back intact.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import tempfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aieng.advanced_rag.cache import (
    CacheKey,
    CacheTier,
    JsonValue,
    TieredResultCache,
)
from aieng.advanced_rag.ingestion.tables import (
    TableStorage,
    TableValueCheck,
    check_values_on_page,
    render_table_block,
)
from aieng.advanced_rag.provenance import Provenance
from aieng.advanced_rag.retrieval.chunkers import Item, serialise, table_aware_chunks
from aieng.advanced_rag.retrieval.corpus import Chunk
from aieng.advanced_rag.retrieval.tables import TableGrid


#: Bumped whenever a change here would produce a different folder from the same
#: source. It is part of the cache key, so a bump invalidates every parse rather
#: than serving output the current code would not write.
PARSER_VERSION = 1

#: The artifact type every ingestion key uses, so one RI's parses are findable
#: as a group in the cache tree.
ARTIFACT_TYPE = "docling-ingestion"

DOCUMENT_FILENAME = "document.json"
MANIFEST_FILENAME = "ingestion.json"
PAGES_DIRNAME = "pages"
TABLES_DIRNAME = "tables"

#: A converted document: ``Path`` in, ``DoclingDocument`` out. Typed loosely
#: because ``docling-core`` is an optional extra and the annotation has to
#: resolve without it.
DoclingConverter = Callable[[Path], Any]

#: ``(document, table_item)`` in, PNG bytes out, or None when this table cannot
#: be rendered. Injected for the same reason the converter is: cropping a table
#: out of a page needs the page raster, which only the real converter has.
TableImageRenderer = Callable[[Any, Any], bytes | None]


@dataclass(frozen=True)
class IngestionResult:
    """One converted document, its folder, and what the parse found.

    Parameters
    ----------
    doc_id : str
        Identifier of the document, which is also its folder name.
    directory : Path
        ``<root>/<doc_id>``, holding the files named in the module docstring.
    storage : TableStorage
        The table setting this parse ran with.
    source_sha256 : str
        Digest of the source file, which is part of the cache key.
    parser_version : int
        The parser version this parse ran with, also part of the key.
    pages : tuple of int
        The page numbers written, ascending.
    cache_hit : bool
        True when the converter was not called because the key resolved.
    tier : str
        Which cache tier answered, ``"local"`` or ``"global"``.
    table_checks : tuple of TableValueCheck
        One entry per table, only when the setting runs the check. Read
        ``failed_checks`` for the ones that did not pass.
    failures : tuple of str
        Everything that went wrong without being fatal: a failed value check, a
        table whose image could not be rendered, a desynchronised table
        reference. The folder is still written.
    """

    doc_id: str
    directory: Path
    storage: TableStorage
    source_sha256: str
    parser_version: int
    pages: tuple[int, ...]
    cache_hit: bool
    tier: CacheTier
    table_checks: tuple[TableValueCheck, ...]
    failures: tuple[str, ...]

    @property
    def failed_checks(self) -> tuple[TableValueCheck, ...]:
        """Return the value checks that did not pass."""
        return tuple(check for check in self.table_checks if not check.passed)

    def page_file(self, page: int) -> Path:
        """Return the path of one page's markdown file.

        Parameters
        ----------
        page : int
            The page number, counted from 1 as Docling counts it.

        Returns
        -------
        Path
            ``<directory>/pages/pNNN.md``, where ``NNN`` is one less than
            ``page``, because the page files are zero-based. The file exists
            only when ``page`` is in :attr:`pages`.
        """
        return self.directory / PAGES_DIRNAME / _page_filename(page)


def source_sha256(path: Path) -> str:
    """Return the SHA-256 of a source file, read in blocks.

    Parameters
    ----------
    path : Path
        The source document.

    Returns
    -------
    str
        The hex digest, which goes into the cache key's configuration so a
        re-converted source is a different artifact rather than a stale hit.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ingest_document(
    source: Path,
    *,
    root: Path,
    convert: DoclingConverter,
    cache: TieredResultCache,
    ri_id: str,
    storage: TableStorage = TableStorage.MARKDOWN,
    doc_id: str | None = None,
    producer_id: str = "docling",
    parser_version: int = PARSER_VERSION,
    render_table_image: TableImageRenderer | None = None,
    max_chars: int = 1200,
    force_refresh: bool = False,
) -> IngestionResult:
    """Convert one document and write its folder under ``root``.

    The folder is rewritten from the cached payload on every call, including a
    hit, so switching the table setting cannot leave a file from the previous
    setting behind.

    Parameters
    ----------
    source : Path
        The source document, typically a PDF.
    root : Path
        The directory the per-document folders live under.
    convert : callable
        ``Path`` in, ``DoclingDocument`` out. For the real converter, install
        ``docling`` and pass
        ``lambda path: DocumentConverter().convert(path).document``.
    cache : TieredResultCache
        The artifact cache that keys the parse and stores the payload.
    ri_id : str
        The reference implementation this parse belongs to, for the cache key.
    storage : TableStorage, optional
        What to keep for each table. Defaults to the markdown grid alone.
    doc_id : str or None, optional
        Folder name. Defaults to the source file's stem.
    producer_id : str, optional
        Converter identity, for the cache key. Change it when you change
        converters, which a parser version bump would not express.
    parser_version : int, optional
        Defaults to :data:`PARSER_VERSION`. Part of the cache key.
    render_table_image : callable or None, optional
        ``(document, table_item)`` in, PNG bytes or None out. Required for any
        setting that keeps images; without it every table falls back to its
        grid and records a failure.
    max_chars : int, optional
        Chunk budget handed to ``table_aware_chunks``, which decides when an
        oversized table is split into row groups that repeat their headers.
    force_refresh : bool, optional
        Reconvert and overwrite the cached payload even on a hit.

    Returns
    -------
    IngestionResult
        The folder that was written, and what the parse found.
    """
    resolved_id = doc_id or source.stem
    digest = source_sha256(source)
    key = CacheKey(
        ri_id=ri_id,
        artifact_type=ARTIFACT_TYPE,
        producer_id=producer_id,
        config={
            "doc_id": resolved_id,
            "max_chars": max_chars,
            "parser_version": parser_version,
            "source_sha256": digest,
            "table_storage": storage.value,
        },
    )

    def compute(payload_path: Path) -> JsonValue:
        with tempfile.TemporaryDirectory() as staging:
            folder = Path(staging) / resolved_id
            manifest = _build_folder(
                folder,
                source=source,
                doc_id=resolved_id,
                digest=digest,
                parser_version=parser_version,
                storage=storage,
                convert=convert,
                render_table_image=render_table_image,
                max_chars=max_chars,
            )
            with tarfile.open(payload_path, "w") as archive:
                archive.add(folder, arcname=resolved_id)
        return manifest

    result = cache.load_or_compute_artifact(
        key,
        compute,
        media_type="application/x-tar",
        force_refresh=force_refresh,
    )
    directory = root / resolved_id
    _unpack(result.path, root, resolved_id)
    return _result_from_manifest(
        result.metadata,
        doc_id=resolved_id,
        directory=directory,
        storage=storage,
        digest=digest,
        parser_version=parser_version,
        cache_hit=result.cache_hit,
        tier=result.tier,
    )


def ingest_corpus(
    sources: Iterable[Path],
    **options: Any,
) -> list[IngestionResult]:
    """Convert several documents into one root, in the order given.

    Parameters
    ----------
    sources : iterable of Path
        The source documents.
    **options
        Passed through to :func:`ingest_document`.

    Returns
    -------
    list of IngestionResult
        One result per source. A table that fails its value check or whose
        image cannot be rendered is recorded in that document's ``failures``,
        so one bad table does not end the corpus.
    """
    return [ingest_document(source, **options) for source in sources]


def _page_index(page_no: int) -> int:
    """Return the zero-based page index for a Docling page number.

    Docling numbers pages from 1, and the package keeps that meaning everywhere
    ``Provenance.page`` is read. The page files on disk are zero-based instead,
    because that is the identity FinanceBench's page-level gold evidence uses:
    the agentic cases name pages as ``financebench:<doc>:p<index>`` with the
    index zero-based, so a 1-based file name would put every page-level gold
    match out by one. The floor at zero mirrors the prior art, which clamps an
    unexpected page number rather than writing a negative index.
    """
    return max(0, page_no - 1)


def _page_filename(page_no: int) -> str:
    """Return a page's file name, zero-padded so a listing sorts correctly."""
    return f"p{_page_index(page_no):03d}.md"


def _tables_in_order(document: Any) -> dict[str, Any]:
    """Return the tables ``serialise`` keeps, keyed by ``self_ref``, in order.

    A table is recognised by its reference rather than by ``isinstance`` against
    ``docling_core``'s ``TableItem``, for the same reason ``TableGrid`` reads its
    cells with ``getattr``: this module has to import and be testable without
    the converter's types resolving to anything in particular.

    ``serialise`` returns ``Item`` objects without the element they came from, so
    the elements are collected by walking the document again under the same
    condition ``serialise`` drops an item on, an empty rendering.
    """
    tables: dict[str, Any] = {}
    for element, _level in document.iterate_items():
        ref = str(getattr(element, "self_ref", ""))
        if not ref.startswith("#/tables/") or not hasattr(element, "data"):
            continue
        if TableGrid.from_table(element).to_markdown().strip():
            tables[ref] = element
    return tables


def _pages_by_ref(doc_id: str, exported: dict[str, Any]) -> dict[str, int]:
    """Return each element's page, by ``self_ref``.

    ``Provenance.from_docling`` already reads ``prov[0].page_no`` and is the
    package's one answer to "which page is this element on", so the page numbers
    the manifest records come from it rather than from a second reader of the
    same field.
    """
    pages: dict[str, int] = {}
    for group in ("texts", "tables", "pictures", "groups"):
        for element in exported.get(group) or []:
            if not isinstance(element, dict):
                continue
            provenance = Provenance.from_docling(doc_id, element)
            if provenance.chunk_id is not None and provenance.page is not None:
                pages[str(provenance.chunk_id)] = int(provenance.page)
    return pages


def _item_page(items: Sequence[Item], fallback: int) -> list[int]:
    """Return one page number per item, carrying the last known page forward.

    An item with no provenance belongs to the page the reading order put it
    after, which is a better guess than dropping it out of the page files.
    """
    out: list[int] = []
    current = fallback
    for item in items:
        if item.page is not None:
            current = int(item.page)
        out.append(current)
    return out


def _page_of_chunk(chunk: Chunk, items: Sequence[Item], pages: Sequence[int]) -> int:
    """Return the page of the first item the chunk overlaps."""
    for item, page in zip(items, pages):
        if chunk.start < item.end and item.start < chunk.end:
            return page
    return pages[0] if pages else 1


def _page_text(items: Sequence[Item], pages: Sequence[int], page: int) -> str:
    """Return a page's prose, with every table on it left out.

    The value check compares a table against this string, so including the
    table's own rendering would make the check pass by construction.
    """
    return "\n".join(
        item.text
        for item, item_page in zip(items, pages)
        if item_page == page and item.kind != "table"
    )


@dataclass
class _Parse:
    """What one parse accumulates while its folder is written.

    Two lists and a map that every table can add to, carried as one object so
    the helpers below do not each need three out-parameters.

    Parameters
    ----------
    failures : list of str
        Everything that went wrong without being fatal.
    checks : list of TableValueCheck
        One entry per table, only for the setting that runs the check.
    images : dict
        Kept image paths relative to the document folder, by table reference.
    """

    failures: list[str] = field(default_factory=list)
    checks: list[TableValueCheck] = field(default_factory=list)
    images: dict[str, JsonValue] = field(default_factory=dict)


def _keep_image(
    folder: Path,
    element: Any,
    ref: str,
    index: int,
    *,
    document: Any,
    render_table_image: TableImageRenderer | None,
    parse: _Parse,
) -> str | None:
    """Render one table image, and return its path relative to a page file.

    Returns None when no renderer was injected or the renderer declined. A
    large table splits into several page blocks that all point at the one
    image, so a reference already rendered is not rendered again.
    """
    name = f"t{index:03d}.png"
    relative = f"{TABLES_DIRNAME}/{name}"
    if ref in parse.images:
        return f"../{relative}"
    payload = render_table_image(document, element) if render_table_image else None
    if not payload:
        parse.failures.append(f"{ref} produced no image; kept as markdown instead")
        return None
    (folder / TABLES_DIRNAME).mkdir(parents=True, exist_ok=True)
    (folder / TABLES_DIRNAME / name).write_bytes(payload)
    parse.images[ref] = relative
    return f"../{relative}"


def _run_check(
    doc_id: str,
    ref: str,
    grid: TableGrid,
    page_text: str,
    page: int,
    *,
    parse: _Parse,
) -> None:
    """Check one table against its page text, recording a failure if it fails.

    A failed check is recorded and the parse continues, because one table whose
    numbers are not in its page text must not end the ingestion of a corpus.
    """
    if any(check.table_ref == ref for check in parse.checks):
        return
    check = check_values_on_page(doc_id, ref, grid, page_text, page=page)
    parse.checks.append(check)
    if not check.passed:
        parse.failures.append(
            f"{ref} on page {page}: "
            f"{', '.join(check.missing_values)} absent from the page text"
        )


def _build_folder(
    folder: Path,
    *,
    source: Path,
    doc_id: str,
    digest: str,
    parser_version: int,
    storage: TableStorage,
    convert: DoclingConverter,
    render_table_image: TableImageRenderer | None,
    max_chars: int,
) -> JsonValue:
    """Convert the source and write the whole document folder.

    Returns the manifest, which is both written into the folder and stored as
    the cache envelope's metadata, so a hit can rebuild the result object
    without reading the payload back.
    """
    document = convert(source)
    exported = document.export_to_dict()
    _, items = serialise(document)

    pages_by_ref = _pages_by_ref(doc_id, exported)
    first_page = min(pages_by_ref.values()) if pages_by_ref else 1
    item_pages = _item_page(items, first_page)

    table_items = [item for item in items if item.kind == "table"]
    elements_by_ref = _tables_in_order(document)
    refs = list(elements_by_ref)
    parse = _Parse()
    if len(refs) != len(table_items):
        parse.failures.append(
            f"table references desynchronised: {len(refs)} in the document, "
            f"{len(table_items)} in the serialised items; tables are referenced "
            "by position instead"
        )
        refs = [f"#/tables/{index}" for index in range(len(table_items))]
    ref_of_span = {(item.start, item.end): ref for item, ref in zip(table_items, refs)}

    folder.mkdir(parents=True, exist_ok=True)
    (folder / DOCUMENT_FILENAME).write_text(
        json.dumps(exported, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    blocks_by_page: dict[int, list[str]] = {}
    for chunk in table_aware_chunks(doc_id, items, max_chars=max_chars):
        page = _page_of_chunk(chunk, items, item_pages)
        ref = ref_of_span.get((chunk.start, chunk.end))
        if ref is None:
            blocks_by_page.setdefault(page, []).append(chunk.text)
            continue

        element = elements_by_ref.get(ref)
        grid = TableGrid.from_table(element) if element is not None else None
        if grid is None:
            parse.failures.append(f"{ref} has no table element; kept as markdown")
            blocks_by_page.setdefault(page, []).append(chunk.text)
            continue

        image_path = (
            _keep_image(
                folder,
                element,
                ref,
                refs.index(ref),
                document=document,
                render_table_image=render_table_image,
                parse=parse,
            )
            if storage.keeps_image
            else None
        )
        if storage.describes:
            _run_check(
                doc_id,
                ref,
                grid,
                _page_text(items, item_pages, page),
                page,
                parse=parse,
            )

        blocks_by_page.setdefault(page, []).append(
            render_table_block(
                storage,
                chunk.text,
                table_ref=ref,
                page=page,
                grid=grid,
                image_path=image_path,
            )
        )

    (folder / PAGES_DIRNAME).mkdir(parents=True, exist_ok=True)
    for page in sorted(blocks_by_page):
        (folder / PAGES_DIRNAME / _page_filename(page)).write_text(
            "\n\n".join(blocks_by_page[page]).strip() + "\n",
            encoding="utf-8",
        )

    page_numbers: list[JsonValue] = [int(page) for page in sorted(blocks_by_page)]
    # Element-level pages, so a reference implementation can cite the element a
    # page file's text came from without reopening document.json.
    element_pages: dict[str, JsonValue] = {
        ref: int(page) for ref, page in pages_by_ref.items()
    }
    manifest: dict[str, JsonValue] = {
        "doc_id": doc_id,
        "element_pages": element_pages,
        "failures": list(parse.failures),
        "page_files": {
            str(page): f"{PAGES_DIRNAME}/{_page_filename(page)}"
            for page in sorted(blocks_by_page)
        },
        "pages": page_numbers,
        "parser_version": parser_version,
        "source": source.name,
        "source_sha256": digest,
        "table_checks": [check.as_json() for check in parse.checks],
        "table_images": dict(parse.images),
        "table_storage": storage.value,
    }
    (folder / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _unpack(payload: Path, root: Path, doc_id: str) -> None:
    """Replace ``<root>/<doc_id>`` with the folder held in the cached tar.

    The folder is removed first, because a second run under a different table
    setting writes a different set of files and a leftover image from the
    previous setting would read as part of this parse.
    """
    root.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(root / doc_id, ignore_errors=True)
    with tarfile.open(payload, "r") as archive:
        members = list(archive.getmembers())
        for member in members:
            name = Path(member.name)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError(f"cached payload holds an unsafe path: {member.name}")
            if name.parts[0] != doc_id:
                raise ValueError(
                    f"cached payload holds {member.name}, which is outside {doc_id}"
                )
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"cached payload holds a special file: {member.name}")
        # `filter` landed in 3.11.4 and this package supports 3.11.0, so the
        # member validation above is the guarantee and the filter is a bonus.
        options: dict[str, Any] = {}
        if hasattr(tarfile, "data_filter"):
            options["filter"] = "data"
        archive.extractall(root, members=members, **options)  # noqa: S202


def _result_from_manifest(
    metadata: JsonValue,
    *,
    doc_id: str,
    directory: Path,
    storage: TableStorage,
    digest: str,
    parser_version: int,
    cache_hit: bool,
    tier: CacheTier,
) -> IngestionResult:
    """Rebuild the result object from the manifest the cache stored."""
    fields = metadata if isinstance(metadata, dict) else {}
    raw_pages = fields.get("pages")
    raw_checks = fields.get("table_checks")
    raw_failures = fields.get("failures")
    return IngestionResult(
        doc_id=doc_id,
        directory=directory,
        storage=storage,
        source_sha256=digest,
        parser_version=parser_version,
        pages=tuple(
            int(page)
            for page in (raw_pages if isinstance(raw_pages, list) else [])
            if isinstance(page, int)
        ),
        cache_hit=cache_hit,
        tier=tier,
        table_checks=tuple(
            TableValueCheck.from_json(entry)
            for entry in (raw_checks if isinstance(raw_checks, list) else [])
        ),
        failures=tuple(
            str(entry)
            for entry in (raw_failures if isinstance(raw_failures, list) else [])
        ),
    )


__all__ = [
    "ARTIFACT_TYPE",
    "DOCUMENT_FILENAME",
    "MANIFEST_FILENAME",
    "PAGES_DIRNAME",
    "PARSER_VERSION",
    "TABLES_DIRNAME",
    "DoclingConverter",
    "IngestionResult",
    "TableImageRenderer",
    "ingest_corpus",
    "ingest_document",
    "source_sha256",
]
