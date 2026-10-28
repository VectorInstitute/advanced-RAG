"""Data profiles: loading and checking a profile, and the tools it generates."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aieng.advanced_rag.agents.corpus_tools import (
    STANDARD_ARMS,
    CorpusToolServer,
    ToolKind,
)
from aieng.advanced_rag.agents.profile import (
    ProfileError,
    ProfileTools,
    build_server,
    index_python,
    load_profile,
    module_name,
    parse_sections,
    profile_from_dict,
)


PROFILE = """
name: shop
description: Support tickets and the shop's code.
files:
  - glob: "tickets/*.md"
    kind: document
    sections: markdown
  - glob: "docs/*.rst"
    kind: document
    sections: rst
  - glob: "src/shop/*.py"
    kind: code
    language: python
identifiers:
  - name: order_id
    pattern: 'ORD-\\d{5}'
    description: an order number
    examples: [ORD-00001]
tables:
  - name: tickets
    file: tickets.csv
    description: one row per ticket
    doc_column: doc_id
    columns:
      - {name: doc_id, type: text, description: "the ticket's document"}
      - {name: priority, type: integer, description: "1 is urgent"}
      - {name: product, type: text}
code:
  language: python
  package_root: src
"""

TICKET_A = """# Refund for ORD-12345

## Problem
The customer was charged twice for ORD-12345.

## Resolution
Refunded one charge. See also ORD-12345 and ORD-99999.
"""

TICKET_B = """# Late delivery

## Problem
Order ORD-54321 arrived late.
"""

DOCS = """Shipping
========

Rates
-----

Standard shipping costs 5 dollars.

Returns
-------

Returns are free within 30 days.
"""

CODE = '''"""Pricing rules."""

TAX_RATE = 0.13


class Basket:
    """A customer's basket."""

    def total(self, discount: float = 0.0) -> float:
        """Return the total after discount."""
        return 0.0


def shipping(weight: int) -> int:
    """Return the shipping cost."""
    return 5
'''


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    return write_shop(tmp_path)


def write_shop(tmp_path: Path) -> Path:
    """Write the shop folder the profile describes, and return it."""
    (tmp_path / "tickets").mkdir()
    (tmp_path / "tickets" / "a.md").write_text(TICKET_A)
    (tmp_path / "tickets" / "b.md").write_text(TICKET_B)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "shipping.rst").write_text(DOCS)
    (tmp_path / "src" / "shop").mkdir(parents=True)
    (tmp_path / "src" / "shop" / "pricing.py").write_text(CODE)
    (tmp_path / "tickets.csv").write_text(
        "doc_id,priority,product\ntickets/a.md,1,mug\ntickets/b.md,3,lamp\n"
    )
    (tmp_path / "profile.yaml").write_text(PROFILE)
    return tmp_path


@pytest.fixture
def tools(folder: Path) -> ProfileTools:
    profile = load_profile(folder / "profile.yaml", root=folder)
    return ProfileTools(profile, profile.corpus())


def session(tools: ProfileTools, *, generic: bool = True):
    server = CorpusToolServer(tools.corpus)
    tools.register(server)
    return server.session(tools.arm(STANDARD_ARMS["C5"], generic=generic))


def test_a_profile_loads_and_names_its_structures(folder: Path) -> None:
    profile = load_profile(folder / "profile.yaml", root=folder)
    assert profile.structures == ("identifiers", "tables", "sections", "code")
    assert [d.doc_id for d in profile.corpus().documents] == [
        "docs/shipping.rst",
        "src/shop/pricing.py",
        "tickets/a.md",
        "tickets/b.md",
    ]
    assert str(folder) not in json.dumps(profile.to_dict())


def test_a_relative_root_in_the_file_is_read_against_the_file(folder: Path) -> None:
    text = (folder / "profile.yaml").read_text() + "root: .\n"
    (folder / "rooted.yaml").write_text(text)
    assert load_profile(folder / "rooted.yaml").root == folder / "."


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"files": [{"glob": "*.md", "kind": "video"}]}, "kind"),
        ({"files": [{"glob": "*.md", "sections": "(unclosed"}]}, "sections"),
        ({"identifiers": [{"name": "x", "pattern": "a*"}]}, "empty string"),
        ({"identifiers": [{"name": "bad name", "pattern": "a"}]}, "letters"),
        ({"tables": [{"name": "t", "file": "t.xlsx"}]}, ".csv"),
        ({"tables": [{"name": "t", "file": "t.csv", "colums": []}]}, "unknown keys"),
        ({"descripton": "typo"}, "unknown keys"),
    ],
)
def test_a_malformed_profile_is_refused_with_the_reason(
    change: dict, message: str
) -> None:
    record = {"name": "x", "files": [{"glob": "*.md"}], **change}
    with pytest.raises(ProfileError, match=message):
        profile_from_dict(record)


def test_an_unquoted_comma_in_a_flow_mapping_is_caught(tmp_path: Path) -> None:
    path = tmp_path / "p.yaml"
    path.write_text(
        "name: x\nfiles: [{glob: '*.md'}]\ntables:\n  - name: t\n    file: t.csv\n"
        "    columns:\n      - {name: a, description: one, two}\n"
    )
    with pytest.raises(ProfileError, match="Quote a value"):
        load_profile(path)


def test_markdown_sections_skip_code_fences_and_nest_by_level() -> None:
    text = "# Top\nintro\n```\n# not a heading\n```\n## Child\nbody\n# Next\n"
    sections = parse_sections(text, "markdown")
    assert [(s.title, s.level, s.start_line, s.end_line) for s in sections] == [
        ("Top", 1, 1, 7),
        ("Child", 2, 6, 7),
        ("Next", 1, 8, 8),
    ]


def test_rst_sections_take_levels_from_the_order_underlines_appear() -> None:
    sections = parse_sections(DOCS, "rst")
    assert [(s.title, s.level) for s in sections] == [
        ("Shipping", 1),
        ("Rates", 2),
        ("Returns", 2),
    ]


def test_a_regex_section_style_uses_its_title_group() -> None:
    text = "ITEM 1. Business\nwords\nITEM 7. Discussion\nmore\n"
    sections = parse_sections(text, r"^ITEM \d+\. (?P<title>.+)$")
    assert [s.title for s in sections] == ["Business", "Discussion"]


def test_python_symbols_are_indexed_without_running_the_file() -> None:
    symbols = index_python("src/shop/pricing.py", CODE, "shop.pricing")
    assert {(s.qualname, s.kind) for s in symbols} == {
        ("shop.pricing.TAX_RATE", "variable"),
        ("shop.pricing.Basket", "class"),
        ("shop.pricing.Basket.total", "method"),
        ("shop.pricing.shipping", "function"),
    }
    assert index_python("x.py", "def broken(:\n", "x") == []
    assert module_name("src/shop/__init__.py", "src") == "shop"


def test_the_tools_appear_only_for_declared_structures(tools: ProfileTools) -> None:
    assert tools.names == (
        "find_identifier",
        "list_identifiers",
        "sql",
        "outline",
        "open_section",
        "find_symbol",
        "open_symbol",
    )
    bare = profile_from_dict(
        {"name": "b", "files": [{"glob": "tickets/*.md"}]}, root=tools.profile.root
    )
    assert ProfileTools(bare, bare.corpus()).names == ()


def test_find_identifier_is_exact_and_suggests_near_values(tools: ProfileTools) -> None:
    s = session(tools)
    found = s.call("find_identifier", {"value": "ORD-12345"})
    assert found["total_documents"] == 1
    assert found["matches"][0]["doc_id"] == "tickets/a.md"
    assert found["matches"][0]["count"] == 3
    missing = s.call("find_identifier", {"value": "ORD-123"})
    assert missing["total_documents"] == 0
    assert missing["similar"][0]["value"] == "ORD-12345"
    assert s.trace.events[0].returned_ids == ("tickets/a.md",)


def test_list_identifiers_counts_documents_per_value(tools: ProfileTools) -> None:
    s = session(tools)
    listed = s.call("list_identifiers", {"kind": "order_id"})
    assert listed["total_distinct"] == 3
    assert {v["value"] for v in listed["values"]} == {
        "ORD-12345",
        "ORD-99999",
        "ORD-54321",
    }
    assert "error" in s.call("list_identifiers", {"kind": "nope"})


def test_sql_reads_the_typed_tables_and_records_the_documents(
    tools: ProfileTools,
) -> None:
    s = session(tools)
    result = s.call(
        "sql", {"query": "SELECT doc_id, priority FROM tickets WHERE priority < 2"}
    )
    assert result["rows"] == [["tickets/a.md", 1]]
    assert s.trace.events[-1].returned_ids == ("tickets/a.md",)


@pytest.mark.parametrize(
    "query",
    [
        "DELETE FROM tickets",
        "DROP TABLE tickets",
        "CREATE TABLE x (a)",
        "ATTACH DATABASE 'x.db' AS x",
        "PRAGMA query_only = 0",
    ],
)
def test_sql_refuses_anything_but_a_read(tools: ProfileTools, query: str) -> None:
    s = session(tools)
    assert "error" in s.call("sql", {"query": query})
    assert s.call("sql", {"query": "SELECT count(*) FROM tickets"})["rows"] == [[2]]


def test_sql_stops_a_runaway_query(
    tools: ProfileTools, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("aieng.advanced_rag.agents.profile.SQL_TIMEOUT", 0.05)
    s = session(tools)
    endless = (
        "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n) "
        "SELECT count(*) FROM n"
    )
    assert "error" in s.call("sql", {"query": endless})


def test_the_sql_description_carries_the_schema(tools: ProfileTools) -> None:
    pytest.importorskip("google.adk", reason="the agents extra is not installed")
    from aieng.advanced_rag.agents.toolset import tool_declarations  # noqa: PLC0415

    s = session(tools)
    (declared,) = tool_declarations([next(t for t in s.tools if t.__name__ == "sql")])
    assert "tickets (2 rows): one row per ticket" in declared["description"]
    assert "priority integer -- 1 is urgent" in declared["description"]
    assert declared["parameters_json_schema"]["required"] == ["query"]


def test_outline_and_open_section_read_by_heading(tools: ProfileTools) -> None:
    s = session(tools)
    outline = s.call("outline", {"doc_id": "tickets/a.md"})
    assert [x["title"] for x in outline["sections"]] == [
        "Refund for ORD-12345",
        "Problem",
        "Resolution",
    ]
    section = s.call("open_section", {"doc_id": "tickets/a.md", "section": "resol"})
    assert section["section"] == "Resolution"
    assert "Refunded one charge" in section["text"]
    assert s.trace.read_ids == ["tickets/a.md"]
    assert s.opened_any
    by_number = s.call("open_section", {"doc_id": "docs/shipping.rst", "section": "3"})
    assert by_number["section"] == "Returns"
    assert "error" in s.call("open_section", {"doc_id": "tickets/a.md", "section": "x"})


def test_open_section_stops_at_the_output_cap(tools: ProfileTools) -> None:
    server = CorpusToolServer(tools.corpus)
    tools.register(server)
    arm = tools.arm(STANDARD_ARMS["C5"]).with_(output_cap_tokens=10)
    result = server.session(arm).call(
        "open_section", {"doc_id": "tickets/a.md", "section": "1"}
    )
    assert "continue with open" in result["note"]


def test_symbol_tools_find_and_read_definitions(tools: ProfileTools) -> None:
    s = session(tools)
    found = s.call("find_symbol", {"name": "total"})
    assert found["symbols"][0]["qualname"] == "shop.pricing.Basket.total"
    assert "discount" in found["symbols"][0]["signature"]
    opened = s.call("open_symbol", {"name": "shop.pricing.Basket.total"})
    assert "return 0.0" in opened["text"]
    assert opened["doc_id"] == "src/shop/pricing.py"
    assert "error" in s.call("open_symbol", {"name": "nothing_here"})


def test_the_description_names_tools_only_when_asked(tools: ProfileTools) -> None:
    with_tools = tools.describe()
    without = tools.describe(tools=False)
    assert "find_identifier" in with_tools
    assert "find_identifier" not in without
    assert "order_id (an order number, for example ORD-00001)" in without
    prompt = tools.instruction("Answer from {corpus}.")
    assert prompt.format(corpus="the shop").startswith("Answer from the shop.")


def test_the_arm_keeps_the_base_limits_and_lists_the_extra_tools(
    tools: ProfileTools,
) -> None:
    arm = tools.arm(STANDARD_ARMS["C7"], name="P7")
    assert arm.tools == ("search", "grep", "open")
    assert arm.extra_tools == tools.names
    assert arm.search_k == 20
    assert arm.to_dict()["extra_tools"] == list(tools.names)
    only = tools.arm(STANDARD_ARMS["C7"], generic=False)
    assert only.tools == ("open",)


def test_an_arm_without_extra_tools_keeps_its_old_record() -> None:
    assert "extra_tools" not in STANDARD_ARMS["C5"].to_dict()


def test_the_server_refuses_an_unregistered_or_clashing_tool(
    tools: ProfileTools,
) -> None:
    server = CorpusToolServer(tools.corpus)
    with pytest.raises(ValueError, match="does not have"):
        server.session(tools.arm(STANDARD_ARMS["C5"]))
    with pytest.raises(ValueError, match="already"):
        server.register(ToolKind("grep", lambda session: print))
    tools.register(server)
    with pytest.raises(ValueError, match="already"):
        tools.register(server)


def test_build_server_registers_everything(folder: Path) -> None:
    profile = load_profile(folder / "profile.yaml", root=folder)
    server, tools = build_server(profile)
    assert set(server.tool_kinds) == set(tools.names)
