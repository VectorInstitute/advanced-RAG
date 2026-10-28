"""The harness: the folder it writes, the key it writes under, and the check.

Every test here runs against the committed Docling JSON under ``fixtures``, with
the converter injected as a callable that loads it. No test in this module
touches a PDF or a converter model, which is the point of injecting the
converter at all. The one test that drives the real converter is marked
``integration_test``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from aieng.advanced_rag.cache import TieredResultCache
from aieng.advanced_rag.ingestion.docling import (
    DOCUMENT_FILENAME,
    MANIFEST_FILENAME,
    PARSER_VERSION,
    IngestionResult,
    ingest_corpus,
    ingest_document,
    source_sha256,
)
from aieng.advanced_rag.ingestion.tables import TableStorage
from aieng.advanced_rag.metering import UsageMeter


FIXTURE = Path(__file__).parent / "fixtures" / "acme-10k.docling.json"

# Enough of a PNG to be recognisable in a committed fixture folder. Nothing in
# the harness decodes it; the renderer is injected exactly so nothing has to.
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"fake-table-raster"


class FakeConverter:
    """Load the committed Docling JSON, and count how often it was asked to."""

    def __init__(self, fixture: Path = FIXTURE) -> None:
        self.fixture = fixture
        self.calls = 0

    def __call__(self, source: Path) -> Any:
        from docling_core.types.doc.document import DoclingDocument

        self.calls += 1
        return DoclingDocument.model_validate(
            json.loads(self.fixture.read_text(encoding="utf-8"))
        )


def fake_image(document: Any, table: Any) -> bytes | None:
    return PNG_BYTES


@pytest.fixture
def source(tmp_path: Path) -> Path:
    path = tmp_path / "acme-10k.pdf"
    path.write_bytes(b"%PDF-1.7 stand-in for the source bytes\n")
    return path


@pytest.fixture
def cache(tmp_path: Path) -> TieredResultCache:
    return TieredResultCache(
        local_dir=tmp_path / "cache" / "local",
        global_dir=tmp_path / "cache" / "global",
    )


@pytest.fixture
def ingest(
    tmp_path: Path, source: Path, cache: TieredResultCache
) -> Callable[..., IngestionResult]:
    """Return a one-argument ingest call, so a test varies only what it tests."""

    def run(**overrides: Any) -> IngestionResult:
        options: dict[str, Any] = {
            "root": tmp_path / "docling",
            "convert": FakeConverter(),
            "cache": cache,
            "ri_id": "ri-ingestion",
            "render_table_image": fake_image,
        }
        options.update(overrides)
        return ingest_document(options.pop("source", source), **options)

    return run


def relative_files(directory: Path) -> set[str]:
    return {
        str(path.relative_to(directory))
        for path in directory.rglob("*")
        if path.is_file()
    }


def test_the_folder_holds_the_document_the_manifest_and_one_file_per_page(
    ingest: Callable[..., IngestionResult],
) -> None:
    result = ingest()
    assert result.pages == (1, 2)
    assert relative_files(result.directory) == {
        DOCUMENT_FILENAME,
        MANIFEST_FILENAME,
        "pages/p000.md",
        "pages/p001.md",
    }
    assert result.page_file(1).is_file()


def test_the_document_file_is_the_converter_export_and_nothing_else(
    ingest: Callable[..., IngestionResult],
) -> None:
    from docling_core.types.doc.document import DoclingDocument

    result = ingest()
    payload = json.loads((result.directory / DOCUMENT_FILENAME).read_text())
    assert DoclingDocument.model_validate(payload).name == "acme-10k"


def test_a_table_keeps_the_section_heading_above_it(
    ingest: Callable[..., IngestionResult],
) -> None:
    # `table_aware_chunks` is reused for exactly this, so the page file carries
    # the heading the table sat under rather than a bare grid.
    page = ingest().page_file(1).read_text()
    assert "Segment results\n| " in page


def test_pages_come_from_the_provenance_the_converter_recorded(
    ingest: Callable[..., IngestionResult],
) -> None:
    result = ingest()
    assert "4,812" in result.page_file(1).read_text()
    assert "Results of operations" in result.page_file(2).read_text()
    assert "4,812" not in result.page_file(2).read_text()


@pytest.mark.parametrize(
    ("storage", "expected"),
    [
        (TableStorage.MARKDOWN, {"grid"}),
        (
            TableStorage.MARKDOWN_WITH_IMAGE,
            {"image", "header", "description", "grid", "reference"},
        ),
        (TableStorage.IMAGE_ONLY, {"image", "reference"}),
        (TableStorage.BOTH, {"image", "grid", "reference"}),
    ],
)
def test_the_four_settings_write_four_different_folders(
    ingest: Callable[..., IngestionResult],
    storage: TableStorage,
    expected: set[str],
) -> None:
    result = ingest(storage=storage)
    files = relative_files(result.directory)
    page = result.page_file(1).read_text()
    present = {
        "image": "tables/t000.png" in files,
        "header": "<!-- table ref=#/tables/0" in page,
        "description": "Table of 3 data rows" in page,
        "grid": "| Instruments | 4,812 |" in page,
        "reference": "![#/tables/0](../tables/t000.png)" in page,
    }
    assert {name for name, found in present.items() if found} == expected


def test_switching_the_setting_does_not_leave_the_old_files_behind(
    ingest: Callable[..., IngestionResult],
) -> None:
    both = ingest(storage=TableStorage.BOTH)
    assert "tables/t000.png" in relative_files(both.directory)
    markdown = ingest(storage=TableStorage.MARKDOWN)
    assert "tables/t000.png" not in relative_files(markdown.directory)


def test_a_second_run_with_nothing_changed_is_a_cache_hit(
    tmp_path: Path, source: Path, cache: TieredResultCache
) -> None:
    converter = FakeConverter()
    options: dict[str, Any] = {
        "root": tmp_path / "docling",
        "convert": converter,
        "cache": cache,
        "ri_id": "ri-ingestion",
        "render_table_image": fake_image,
    }
    first = ingest_document(source, **options)
    second = ingest_document(source, **options)

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert converter.calls == 1
    # The folder is rewritten from the payload on a hit, so the contract the
    # reference implementations read exists either way.
    assert relative_files(second.directory) == relative_files(first.directory)


def test_a_cache_hit_reports_one_hit_to_the_meter(
    tmp_path: Path, source: Path, cache: TieredResultCache
) -> None:
    options: dict[str, Any] = {
        "root": tmp_path / "docling",
        "convert": FakeConverter(),
        "cache": cache,
        "ri_id": "ri-ingestion",
        "render_table_image": fake_image,
    }
    meter = UsageMeter()
    with meter.question("cold"):
        ingest_document(source, **options)
    with meter.question("warm"):
        ingest_document(source, **options)

    assert meter.per_question()["cold"].cache_hits == 0
    assert meter.per_question()["warm"].cache_hits == 1
    assert meter.totals().cache_hits == 1


def test_a_changed_source_hash_rebuilds(
    tmp_path: Path, source: Path, cache: TieredResultCache
) -> None:
    converter = FakeConverter()
    options: dict[str, Any] = {
        "root": tmp_path / "docling",
        "convert": converter,
        "cache": cache,
        "ri_id": "ri-ingestion",
        "render_table_image": fake_image,
    }
    first = ingest_document(source, **options)
    source.write_bytes(b"%PDF-1.7 the source was re-exported\n")
    second = ingest_document(source, **options)

    assert second.source_sha256 != first.source_sha256
    assert second.source_sha256 == source_sha256(source)
    assert second.cache_hit is False
    assert converter.calls == 2


def test_a_bumped_parser_version_rebuilds(
    tmp_path: Path, source: Path, cache: TieredResultCache
) -> None:
    converter = FakeConverter()
    options: dict[str, Any] = {
        "root": tmp_path / "docling",
        "convert": converter,
        "cache": cache,
        "ri_id": "ri-ingestion",
        "render_table_image": fake_image,
    }
    first = ingest_document(source, **options)
    second = ingest_document(source, parser_version=PARSER_VERSION + 1, **options)

    assert first.cache_hit is False
    assert second.cache_hit is False
    assert second.parser_version == PARSER_VERSION + 1
    assert converter.calls == 2


def test_the_check_reports_a_table_whose_values_are_not_on_its_page(
    ingest: Callable[..., IngestionResult],
) -> None:
    # Page 1 restates every segment figure in prose, so that table passes. The
    # operations table on page 2 has no such restatement, so it fails.
    result = ingest(storage=TableStorage.MARKDOWN_WITH_IMAGE)
    assert {check.table_ref for check in result.table_checks} == {
        "#/tables/0",
        "#/tables/1",
    }
    failed = result.failed_checks
    assert [check.table_ref for check in failed] == ["#/tables/1"]
    assert failed[0].page == 2
    assert failed[0].missing_values == ("1,587", "2,031")
    assert any("#/tables/1 on page 2" in failure for failure in result.failures)


def test_a_failed_check_does_not_stop_the_folder_being_written(
    ingest: Callable[..., IngestionResult],
) -> None:
    result = ingest(storage=TableStorage.MARKDOWN_WITH_IMAGE)
    assert result.failed_checks
    assert result.page_file(2).is_file()
    assert json.loads((result.directory / MANIFEST_FILENAME).read_text())["failures"]


def test_the_check_only_runs_for_the_setting_that_makes_claims(
    ingest: Callable[..., IngestionResult],
) -> None:
    for storage in (
        TableStorage.MARKDOWN,
        TableStorage.IMAGE_ONLY,
        TableStorage.BOTH,
    ):
        assert ingest(storage=storage).table_checks == ()


def test_a_table_with_no_image_records_a_failure_and_keeps_the_grid(
    ingest: Callable[..., IngestionResult],
) -> None:
    result = ingest(storage=TableStorage.IMAGE_ONLY, render_table_image=None)
    assert "| Instruments | 4,812 |" in result.page_file(1).read_text()
    assert any("produced no image" in failure for failure in result.failures)


def test_the_failures_of_one_table_do_not_end_the_corpus(
    tmp_path: Path, source: Path, cache: TieredResultCache
) -> None:
    second_source = tmp_path / "acme-10q.pdf"
    second_source.write_bytes(b"%PDF-1.7 a second filing\n")
    results = ingest_corpus(
        [source, second_source],
        root=tmp_path / "docling",
        convert=FakeConverter(),
        cache=cache,
        ri_id="ri-ingestion",
        storage=TableStorage.MARKDOWN_WITH_IMAGE,
        render_table_image=fake_image,
    )
    assert [result.doc_id for result in results] == ["acme-10k", "acme-10q"]
    assert all(result.failed_checks for result in results)
    assert all(result.page_file(1).is_file() for result in results)


@pytest.mark.integration_test
def test_the_real_converter_fits_the_injected_callable(
    tmp_path: Path, cache: TieredResultCache
) -> None:
    converter_module = pytest.importorskip("docling.document_converter")

    source = tmp_path / "note.html"
    source.write_text(
        "<html><body><h1>Segment results</h1>"
        "<table><tr><th>Segment</th><th>2022</th></tr>"
        "<tr><td>Instruments</td><td>4,812</td></tr></table>"
        "</body></html>",
        encoding="utf-8",
    )
    converter = converter_module.DocumentConverter()
    result = ingest_document(
        source,
        root=tmp_path / "docling",
        convert=lambda path: converter.convert(path).document,
        cache=cache,
        ri_id="ri-ingestion",
    )
    assert (result.directory / DOCUMENT_FILENAME).is_file()
    assert result.pages
