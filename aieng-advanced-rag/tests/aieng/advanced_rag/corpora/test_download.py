"""The per-RI fetch command: what each RI pulls, and how a failure is reported."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from aieng.advanced_rag.corpora import download
from aieng.advanced_rag.corpora.download import CORE, ITEMS, RIS, Item, main, plan, run


@pytest.fixture
def fake_items(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, int]:
    """Replace every fetcher with one that writes a small file and counts calls."""
    calls: dict[str, int] = {}

    def fetcher(name: str):
        def fetch_one() -> list[Path]:
            calls[name] = calls.get(name, 0) + 1
            if name == "serverfault":
                raise RuntimeError("extracting the dump needs the 7z program")
            path = tmp_path / f"{name}.json"
            path.write_text(name)
            return [path]

        return fetch_one

    fakes = {
        name: Item(name, item.description, fetcher(name))
        for name, item in ITEMS.items()
    }
    monkeypatch.setattr(download, "ITEMS", fakes)
    return calls


def test_core_leaves_out_the_large_downloads():
    assert "financebench-pdfs" not in CORE
    assert "serverfault" not in CORE
    assert "bge-small" in CORE


def test_every_ri_names_only_known_items_and_the_embedding_model():
    for names in RIS.values():
        assert set(names) <= set(ITEMS)
        assert "bge-small" in names


def test_a_plan_is_ordered_and_has_no_repeats():
    names = plan(["2", "4"])

    assert names == ["multihop-rag", "musique", "serverfault", "bge-small"]


def test_an_ri_folder_name_or_path_means_the_same_as_its_number():
    assert plan(["table_structured_reasoning"]) == plan(["3"])
    assert plan(["implementations/table_structured_reasoning/"]) == plan(["3"])


def test_an_item_name_fetches_only_that_item():
    assert plan(["cuad"]) == ["cuad"]


def test_core_and_an_ri_combine():
    assert set(plan(["3"], core=True)) == set(CORE) | {"financebench-pdfs"}


def test_an_unknown_name_is_refused():
    with pytest.raises(ValueError, match="unknown RI"):
        plan(["7"])


def test_each_path_is_printed_with_its_digest(fake_items, tmp_path):
    lines: list[str] = []

    status = run(["cuad"], out=lines.append)

    digest = hashlib.sha256(b"cuad").hexdigest()
    assert status == 0
    assert lines[0] == f"ok   cuad: {tmp_path / 'cuad.json'} sha256={digest}"


def test_one_failure_does_not_stop_the_rest_and_sets_the_status(fake_items):
    lines: list[str] = []

    status = run(["serverfault", "bge-small"], out=lines.append)

    assert status == 1
    assert fake_items == {"serverfault": 1, "bge-small": 1}
    assert lines[0].startswith("FAIL serverfault: RuntimeError")
    assert lines[-1] == "1 of 2 failed: serverfault"


def test_the_command_needs_something_to_fetch(capsys):
    with pytest.raises(SystemExit) as raised:
        main([])

    assert raised.value.code == 2
    assert "--core" in capsys.readouterr().err


def test_list_fetches_nothing(fake_items, capsys):
    assert main(["--list"]) == 0
    assert fake_items == {}
    assert "RI 4" in capsys.readouterr().out
