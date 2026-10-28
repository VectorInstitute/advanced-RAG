"""Describe a corpus in a short profile, and turn the profile into tools.

``search``, ``grep``, and ``open`` treat every corpus as a pile of text. Most
team corpora are not: posts carry tags and scores, filings carry tables and
numbered items, a repository carries definitions. A :class:`DataProfile` says
which of those a corpus holds, in a YAML file a team writes in a few minutes, and
:class:`ProfileTools` builds one tool family per declared structure:

* ``identifiers`` (named regular expressions such as error codes or file paths)
  give ``find_identifier`` and ``list_identifiers``, over an index built once.
* ``tables`` (CSV or JSON Lines files with described columns) give ``sql``, a
  read-only SQLite query over the tables loaded in memory. The tool's
  description carries the schema and the column descriptions.
* ``sections`` on a file rule (Markdown or reStructuredText headings, or a
  regular expression) give ``outline`` and ``open_section``.
* ``code`` (Python) gives ``find_symbol`` and ``open_symbol``, over an index of
  classes, functions, methods, and module-level names read with :mod:`ast`.

The tools register on a
:class:`~aieng.advanced_rag.agents.corpus_tools.CorpusToolServer` as
:class:`~aieng.advanced_rag.agents.corpus_tools.ToolKind` entries, so they
log to the same :class:`~aieng.advanced_rag.agents.corpus_tools.ToolTrace` as
the built-in tools, and an arm offers them by listing their names in
``extra_tools``:

    profile = load_profile("profiles/serverfault.yaml", root=folder)
    corpus = profile.corpus()
    tools = ProfileTools(profile, corpus)
    server = CorpusToolServer(corpus, search=DenseSearch(corpus, embedder))
    tools.register(server)
    arm = tools.arm(STANDARD_ARMS["C7"], name="P7")

The profile format is documented in ``profiles/template.yaml`` under
``implementations/agent_native_retrieval``.
"""

from __future__ import annotations

import ast
import csv
import json
import re
import sqlite3
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from .corpus_tools import (
    CHARS_PER_TOKEN,
    ArmConfig,
    Corpus,
    CorpusToolServer,
    ToolKind,
    ToolSession,
)


FileKind = Literal["document", "code", "text"]
ColumnType = Literal["text", "integer", "real"]

#: Names of every tool a profile can generate, by the structure that adds it.
PROFILE_TOOL_NAMES: Mapping[str, tuple[str, ...]] = {
    "identifiers": ("find_identifier", "list_identifiers"),
    "tables": ("sql",),
    "sections": ("outline", "open_section"),
    "code": ("find_symbol", "open_symbol"),
}

#: Rows ``sql`` returns before it stops and says how many it left out.
SQL_MAX_ROWS = 50

#: Seconds a query may run before SQLite is told to stop it.
SQL_TIMEOUT = 5.0

_SECTION_STYLES = ("markdown", "rst", "none")
_RST_UNDERLINE = re.compile(r"^([=\-~^\"'`#*+:.])\1{2,}\s*$")
_MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ProfileError(ValueError):
    """A profile is malformed: a missing key, a bad pattern, or an unknown kind."""


# ---------------------------------------------------------------------------
# The profile


@dataclass(frozen=True)
class FileRule:
    """Which files are documents, and how to read them.

    Parameters
    ----------
    glob : str
        A pattern relative to the profile's root, such as ``docs/**/*.md``.
    kind : str, optional
        ``document`` (prose with headings), ``code`` (source files), or
        ``text`` (anything else).
    sections : str, optional
        How a document splits into sections: ``markdown`` (``#`` headings),
        ``rst`` (underlined titles), ``none``, or a regular expression whose
        match starts a section. A named group ``title`` gives the section's
        title; the whole match is the title otherwise.
    language : str, optional
        For ``code`` files. Only ``python`` gets a symbol index.
    """

    glob: str
    kind: FileKind = "document"
    sections: str = "none"
    language: str = ""


@dataclass(frozen=True)
class IdentifierKind:
    """A named pattern for the identifiers a corpus uses.

    Parameters
    ----------
    name : str
        A short label the agent passes to ``list_identifiers``.
    pattern : str
        A Python regular expression. The whole match is the identifier.
    description : str, optional
        What the identifiers are, for the tool description.
    examples : tuple of str, optional
        A few real values, which help the model recognise the kind.
    ignore_case : bool, optional
        Treat values that differ only in case as one identifier.
    """

    name: str
    pattern: str
    description: str = ""
    examples: tuple[str, ...] = ()
    ignore_case: bool = False


@dataclass(frozen=True)
class Column:
    """One column of a declared table.

    Parameters
    ----------
    name : str
        The column header in the file, and the SQL column name.
    type : str, optional
        ``text``, ``integer``, or ``real``. Values that do not convert are
        stored as ``NULL``.
    description : str, optional
        What the column holds, shown to the model with the schema.
    """

    name: str
    type: ColumnType = "text"
    description: str = ""


@dataclass(frozen=True)
class TableSpec:
    """A table the agent can query with SQL.

    Parameters
    ----------
    name : str
        The SQL table name.
    file : str
        A ``.csv``, ``.tsv``, or ``.jsonl`` file relative to the root.
    description : str, optional
        What one row is.
    columns : tuple of Column, optional
        The columns to load, with types and descriptions. Every column of the
        file is loaded as text when this is empty.
    doc_column : str, optional
        A column holding corpus document identifiers, so the trace records
        which documents a query named.
    """

    name: str
    file: str
    description: str = ""
    columns: tuple[Column, ...] = ()
    doc_column: str = ""


@dataclass(frozen=True)
class CodeSpec:
    """How to index source code.

    Parameters
    ----------
    language : str, optional
        Only ``python`` is indexed.
    package_root : str, optional
        A leading folder to drop from module names, such as ``src``.
    """

    language: str = "python"
    package_root: str = ""


@dataclass(frozen=True)
class DataProfile:
    """What a corpus holds, as a team describes it.

    Load one with :func:`load_profile`. See the module docstring for the tools
    each part generates.

    Parameters
    ----------
    name : str
        A label for reports and cache keys.
    description : str
        One paragraph for the agent: what the documents are and who wrote them.
    files : tuple of FileRule
        Which files are documents. A file matching several rules takes the
        first.
    identifiers : tuple of IdentifierKind, optional
        Identifier patterns.
    tables : tuple of TableSpec, optional
        Tables for SQL.
    code : CodeSpec or None, optional
        Source code to index.
    root : Path or None, optional
        The folder the paths are relative to.
    max_file_bytes : int, optional
        Larger files are skipped.
    """

    name: str
    description: str
    files: tuple[FileRule, ...]
    identifiers: tuple[IdentifierKind, ...] = ()
    tables: tuple[TableSpec, ...] = ()
    code: CodeSpec | None = None
    root: Path | None = None
    max_file_bytes: int = 5_000_000

    def with_root(self, root: Path | str) -> DataProfile:
        """Return the same profile over another folder."""
        return DataProfile(
            self.name,
            self.description,
            self.files,
            self.identifiers,
            self.tables,
            self.code,
            Path(root),
            self.max_file_bytes,
        )

    def _root(self) -> Path:
        if self.root is None:
            raise ProfileError(f"profile {self.name!r} has no root folder")
        return self.root

    def rules(self) -> dict[str, FileRule]:
        """Return each file under the root with the first rule whose glob finds it.

        Matching uses :meth:`pathlib.Path.glob`, the same call that reads the
        corpus, so a rule covers exactly the files it contributed.
        """
        root = self._root()
        found: dict[str, FileRule] = {}
        for rule in self.files:
            for path in root.glob(rule.glob):
                if path.is_file():
                    found.setdefault(path.relative_to(root).as_posix(), rule)
        return found

    def corpus(self) -> Corpus:
        """Read the declared files under the root, one document per file.

        Returns
        -------
        Corpus
            Identifiers are paths relative to the root, in sorted order.
        """
        return Corpus.from_folder(
            self._root(),
            patterns=[rule.glob for rule in self.files],
            max_bytes=self.max_file_bytes,
            name=self.name,
        )

    @property
    def structures(self) -> tuple[str, ...]:
        """Return which tool families the profile declares, in a fixed order."""
        found: list[str] = []
        if self.identifiers:
            found.append("identifiers")
        if self.tables:
            found.append("tables")
        if any(rule.sections != "none" for rule in self.files):
            found.append("sections")
        if self.code is not None and self.code.language == "python":
            found.append("code")
        return tuple(found)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record, for cache keys and reports.

        The root is left out, so the record is the same on every machine.
        """
        return {
            "name": self.name,
            "description": self.description,
            "files": [vars(rule) for rule in self.files],
            "identifiers": [
                {**vars(kind), "examples": list(kind.examples)}
                for kind in self.identifiers
            ],
            "tables": [
                {
                    "name": t.name,
                    "file": t.file,
                    "description": t.description,
                    "doc_column": t.doc_column,
                    "columns": [vars(c) for c in t.columns],
                }
                for t in self.tables
            ],
            "code": vars(self.code) if self.code else None,
            "max_file_bytes": self.max_file_bytes,
        }


_KEYS: Mapping[str, frozenset[str]] = {
    "the profile": frozenset(
        {
            "name",
            "description",
            "files",
            "identifiers",
            "tables",
            "code",
            "root",
            "max_file_bytes",
        }
    ),
    "files": frozenset({"glob", "kind", "sections", "language"}),
    "identifiers": frozenset(
        {"name", "pattern", "description", "examples", "ignore_case"}
    ),
    "tables": frozenset({"name", "file", "description", "columns", "doc_column"}),
    "columns": frozenset({"name", "type", "description"}),
    "code": frozenset({"language", "package_root"}),
}


def _known(record: Any, part: str, where: str) -> Mapping[str, Any]:
    # A misspelt key, or a YAML flow mapping split by an unquoted comma, would
    # otherwise be dropped without a word.
    if not isinstance(record, Mapping):
        raise ProfileError(f"{where} must be a mapping")
    unknown = sorted(set(record) - _KEYS[part])
    if unknown:
        raise ProfileError(
            f"{where} has unknown keys {unknown}; expected some of {sorted(_KEYS[part])}. "
            "Quote a value that contains a comma or a colon."
        )
    return record


def _require(record: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in record or record[key] in (None, ""):
        raise ProfileError(f"{where} needs {key!r}")
    return record[key]


def _check_name(name: str, where: str) -> str:
    if not _IDENT.match(str(name)):
        raise ProfileError(
            f"{where} name {name!r} must be letters, digits, and underscores"
        )
    return str(name)


def _file_rules(raws: Iterable[Mapping[str, Any]]) -> list[FileRule]:
    files: list[FileRule] = []
    for index, raw in enumerate(raws):
        where = f"files[{index}]"
        _known(raw, "files", where)
        rule = FileRule(
            glob=str(_require(raw, "glob", where)),
            kind=raw.get("kind", "document"),
            sections=str(raw.get("sections", "none")),
            language=str(raw.get("language", "")),
        )
        if rule.kind not in ("document", "code", "text"):
            raise ProfileError(
                f"{where} kind {rule.kind!r} is not document, code, or text"
            )
        if rule.sections not in _SECTION_STYLES:
            try:
                re.compile(rule.sections, re.MULTILINE)
            except re.error as error:
                raise ProfileError(
                    f"{where} sections is neither {_SECTION_STYLES} nor a valid "
                    f"regular expression: {error}"
                ) from error
        files.append(rule)
    return files


def _identifier_kinds(raws: Iterable[Mapping[str, Any]]) -> list[IdentifierKind]:
    kinds: list[IdentifierKind] = []
    for index, raw in enumerate(raws):
        where = f"identifiers[{index}]"
        _known(raw, "identifiers", where)
        kind = IdentifierKind(
            name=_check_name(_require(raw, "name", where), where),
            pattern=str(_require(raw, "pattern", where)),
            description=" ".join(str(raw.get("description", "")).split()),
            examples=tuple(str(e) for e in raw.get("examples") or ()),
            ignore_case=bool(raw.get("ignore_case", False)),
        )
        try:
            compiled = re.compile(kind.pattern)
        except re.error as error:
            raise ProfileError(f"{where} pattern does not compile: {error}") from error
        if compiled.match(""):
            raise ProfileError(f"{where} pattern matches the empty string")
        kinds.append(kind)
    return kinds


def _table_specs(raws: Iterable[Mapping[str, Any]]) -> list[TableSpec]:
    tables: list[TableSpec] = []
    for index, raw in enumerate(raws):
        where = f"tables[{index}]"
        _known(raw, "tables", where)
        columns = []
        for c_index, column in enumerate(raw.get("columns") or ()):
            c_where = f"{where}.columns[{c_index}]"
            _known(column, "columns", c_where)
            spec = Column(
                name=_check_name(_require(column, "name", c_where), c_where),
                type=column.get("type", "text"),
                description=" ".join(str(column.get("description", "")).split()),
            )
            if spec.type not in ("text", "integer", "real"):
                raise ProfileError(
                    f"{c_where} type {spec.type!r} is not text, integer, or real"
                )
            columns.append(spec)
        table = TableSpec(
            name=_check_name(_require(raw, "name", where), where),
            file=str(_require(raw, "file", where)),
            description=" ".join(str(raw.get("description", "")).split()),
            columns=tuple(columns),
            doc_column=str(raw.get("doc_column", "")),
        )
        if Path(table.file).suffix.lower() not in (".csv", ".tsv", ".jsonl"):
            raise ProfileError(f"{where} file must be .csv, .tsv, or .jsonl")
        if (
            table.doc_column
            and columns
            and table.doc_column not in {c.name for c in columns}
        ):
            raise ProfileError(
                f"{where} doc_column {table.doc_column!r} is not a declared column"
            )
        tables.append(table)
    return tables


def profile_from_dict(
    record: Mapping[str, Any], *, root: Path | str | None = None
) -> DataProfile:
    """Build a profile from a parsed YAML or JSON mapping, checking each part.

    Parameters
    ----------
    record : mapping
        The profile's keys: ``name``, ``description``, ``files``, and optionally
        ``identifiers``, ``tables``, ``code``, ``root``, and ``max_file_bytes``.
    root : Path, str, or None, optional
        Overrides ``root`` in the record.

    Returns
    -------
    DataProfile
        The checked profile.

    Raises
    ------
    ProfileError
        On a missing key, an unknown kind or section style, a pattern that does
        not compile, or a duplicate name.
    """
    _known(record, "the profile", "the profile")
    name = str(_require(record, "name", "the profile"))
    files = _file_rules(_require(record, "files", "the profile"))
    identifiers = _identifier_kinds(record.get("identifiers") or ())
    tables = _table_specs(record.get("tables") or ())
    for label, names in (
        ("identifier", [k.name for k in identifiers]),
        ("table", [t.name for t in tables]),
    ):
        if len(set(names)) != len(names):
            raise ProfileError(f"two {label} entries share a name")
    code_record = record.get("code")
    code = None
    if code_record:
        _known(code_record, "code", "code")
        code = CodeSpec(
            language=str(code_record.get("language", "python")),
            package_root=str(code_record.get("package_root", "")).strip("/"),
        )
    chosen_root = root if root is not None else record.get("root")
    return DataProfile(
        name=name,
        description=" ".join(str(record.get("description", "")).split()),
        files=tuple(files),
        identifiers=tuple(identifiers),
        tables=tuple(tables),
        code=code,
        root=Path(chosen_root) if chosen_root else None,
        max_file_bytes=int(record.get("max_file_bytes", 5_000_000)),
    )


def load_profile(path: Path | str, *, root: Path | str | None = None) -> DataProfile:
    """Read a profile from a ``.yaml``, ``.yml``, or ``.json`` file.

    A relative ``root`` in the file is read against the file's own folder.

    Parameters
    ----------
    path : Path or str
        The profile file.
    root : Path, str, or None, optional
        The folder the profile describes. Overrides the file's ``root``.

    Returns
    -------
    DataProfile
        The checked profile.

    Raises
    ------
    ProfileError
        If the file is not a mapping or a part of it is malformed.
    ImportError
        For a YAML file when PyYAML is not installed.
    """
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".json":
        record = json.loads(text)
    else:
        try:
            import yaml  # noqa: PLC0415 - only YAML profiles need it
        except ImportError as error:  # pragma: no cover - depends on the install
            raise ImportError(
                "reading a YAML profile needs PyYAML; install it or write the profile as JSON"
            ) from error
        record = yaml.safe_load(text)
    if not isinstance(record, Mapping):
        raise ProfileError(f"{source} does not hold a mapping")
    if root is None and record.get("root"):
        declared = Path(str(record["root"])).expanduser()
        root = declared if declared.is_absolute() else source.parent / declared
    return profile_from_dict(record, root=root)


# ---------------------------------------------------------------------------
# Indexes


@dataclass(frozen=True)
class Section:
    """One section of a document, by line.

    Parameters
    ----------
    number : int
        Position in the document's outline, from 1.
    level : int
        Heading depth, from 1.
    title : str
        The heading text.
    start_line, end_line : int
        The heading's line and the section's last line, counting from 1. A
        section runs to the next heading at its level or above.
    """

    number: int
    level: int
    title: str
    start_line: int
    end_line: int


def _markdown_heads(lines: Sequence[str]) -> list[tuple[int, int, str]]:
    heads: list[tuple[int, int, str]] = []
    fenced = False
    for number, line in enumerate(lines, 1):
        if _FENCE.match(line):
            fenced = not fenced
            continue
        match = None if fenced else _MARKDOWN_HEADING.match(line)
        if match:
            heads.append((number, len(match.group(1)), match.group(2).strip()))
    return heads


def _rst_heads(lines: Sequence[str]) -> list[tuple[int, int, str]]:
    # reStructuredText has no fixed heading levels: the first underline style a
    # document uses is level 1, the next new one level 2, and so on.
    heads: list[tuple[int, int, str]] = []
    levels: list[str] = []
    for index in range(len(lines) - 1):
        title, under = lines[index].rstrip(), lines[index + 1]
        if not title.strip() or _RST_UNDERLINE.match(title):
            continue
        match = _RST_UNDERLINE.match(under)
        if match and len(under.rstrip()) >= len(title.strip()):
            over = index > 0 and _RST_UNDERLINE.match(lines[index - 1]) is not None
            marker = ("o" if over else "u") + match.group(1)
            if marker not in levels:
                levels.append(marker)
            start = index if over else index + 1
            heads.append((start, levels.index(marker) + 1, title.strip()))
    return heads


def _regex_heads(lines: Sequence[str], pattern: str) -> list[tuple[int, int, str]]:
    compiled = re.compile(pattern, re.MULTILINE)
    heads: list[tuple[int, int, str]] = []
    for number, line in enumerate(lines, 1):
        match = compiled.search(line)
        if match:
            named = match.groupdict().get("title")
            heads.append((number, 1, (named or match.group(0)).strip()))
    return heads


def parse_sections(text: str, style: str) -> list[Section]:
    """Return a document's sections under one heading style.

    Parameters
    ----------
    text : str
        The document.
    style : str
        ``markdown``, ``rst``, ``none``, or a regular expression (see
        :class:`FileRule`).

    Returns
    -------
    list of Section
        In document order. Markdown headings inside code fences are ignored.
    """
    if style == "none":
        return []
    lines = text.splitlines()
    if style == "markdown":
        heads = _markdown_heads(lines)
    elif style == "rst":
        heads = _rst_heads(lines)
    else:
        heads = _regex_heads(lines, style)
    sections: list[Section] = []
    for position, (line, level, title) in enumerate(heads):
        end = len(lines)
        for later_line, later_level, _ in heads[position + 1 :]:
            if later_level <= level:
                end = later_line - 1
                break
        sections.append(Section(position + 1, level, title, line, max(end, line)))
    return sections


@dataclass(frozen=True)
class Symbol:
    """One Python definition.

    Parameters
    ----------
    qualname : str
        Module path and name, such as ``click.core.Command.invoke``.
    kind : str
        ``class``, ``function``, ``method``, or ``variable``.
    doc_id : str
        The file.
    start_line, end_line : int
        The definition's lines, decorators included.
    signature : str
        The ``def`` or ``class`` line, or the assignment, as written.
    summary : str
        The docstring's first line, or ``""``.
    """

    qualname: str
    kind: str
    doc_id: str
    start_line: int
    end_line: int
    signature: str
    summary: str = ""

    @property
    def name(self) -> str:
        """Return the last component of the qualified name."""
        return self.qualname.rsplit(".", 1)[-1]


def module_name(path: str, package_root: str = "") -> str:
    """Return the dotted module name for a Python file's relative path."""
    parts = list(PurePosixPath(path).with_suffix("").parts)
    if package_root:
        prefix = package_root.split("/")
        if parts[: len(prefix)] == prefix:
            parts = parts[len(prefix) :]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) or PurePosixPath(path).stem


def index_python(doc_id: str, text: str, module: str) -> list[Symbol]:
    """Return the classes, functions, methods, and module-level names in a file.

    The file is parsed with :func:`ast.parse` and never run. A file that does
    not parse yields no symbols.
    """
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return []
    lines = text.splitlines()

    def first_line(node: ast.AST) -> int:
        decorators = getattr(node, "decorator_list", [])
        return min([getattr(node, "lineno", 1)] + [d.lineno for d in decorators])

    def line_of(node: ast.AST) -> str:
        number = getattr(node, "lineno", 1)
        return lines[number - 1].strip() if 0 < number <= len(lines) else ""

    def summary(node: Any) -> str:
        doc = ast.get_docstring(node) if hasattr(node, "body") else None
        return doc.strip().splitlines()[0] if doc and doc.strip() else ""

    found: list[Symbol] = []

    def visit(body: Sequence[ast.stmt], prefix: str, in_class: bool) -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                qual = f"{prefix}.{node.name}"
                found.append(
                    Symbol(
                        qual,
                        "class",
                        doc_id,
                        first_line(node),
                        node.end_lineno or node.lineno,
                        line_of(node),
                        summary(node),
                    )
                )
                visit(node.body, qual, True)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                found.append(
                    Symbol(
                        f"{prefix}.{node.name}",
                        "method" if in_class else "function",
                        doc_id,
                        first_line(node),
                        node.end_lineno or node.lineno,
                        line_of(node),
                        summary(node),
                    )
                )
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                for target in targets:
                    if isinstance(target, ast.Name):
                        found.append(
                            Symbol(
                                f"{prefix}.{target.id}",
                                "variable",
                                doc_id,
                                node.lineno,
                                node.end_lineno or node.lineno,
                                line_of(node),
                            )
                        )

    visit(tree.body, module, False)
    return found


def _coerce(value: Any, kind: ColumnType) -> Any:
    if value is None or value == "":
        return None
    if kind == "text":
        return (
            value
            if isinstance(value, str)
            else json.dumps(value)
            if isinstance(value, (list, dict))
            else str(value)
        )
    try:
        if kind == "integer":
            return int(float(str(value).replace(",", "")))
        return float(str(value).replace(",", ""))
    except ValueError:
        return None


def _read_rows(path: Path) -> tuple[list[str], Iterable[Mapping[str, Any]]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        header = list(dict.fromkeys(k for r in records for k in r))
        return header, records
    handle = path.open(encoding="utf-8", newline="")
    reader = csv.DictReader(handle, delimiter="\t" if suffix == ".tsv" else ",")
    rows = list(reader)
    handle.close()
    return list(reader.fieldnames or []), rows


class SqlTables:
    """Declared tables loaded into an in-memory SQLite database, read-only.

    Parameters
    ----------
    tables : sequence of TableSpec
        The tables.
    root : Path
        The folder their files are relative to.

    Raises
    ------
    ProfileError
        If a declared column is missing from its file.
    """

    _ALLOWED = {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
        getattr(sqlite3, "SQLITE_RECURSIVE", 33),
    }

    def __init__(self, tables: Sequence[TableSpec], root: Path) -> None:
        self.tables = tuple(tables)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(":memory:", check_same_thread=False)
        self.columns: dict[str, tuple[Column, ...]] = {}
        self.rows: dict[str, int] = {}
        for table in self.tables:
            header, records = _read_rows(root / table.file)
            columns = table.columns or tuple(
                Column(h) for h in header if _IDENT.match(h)
            )
            missing = [c.name for c in columns if c.name not in header]
            if missing:
                raise ProfileError(f"table {table.name!r} file lacks columns {missing}")
            self.columns[table.name] = columns
            names = ", ".join(f'"{c.name}" {c.type.upper()}' for c in columns)
            self._db.execute(f'CREATE TABLE "{table.name}" ({names})')
            marks = ", ".join("?" for _ in columns)
            batch = [
                tuple(_coerce(r.get(c.name), c.type) for c in columns) for r in records
            ]
            self._db.executemany(f'INSERT INTO "{table.name}" VALUES ({marks})', batch)
            self.rows[table.name] = len(batch)
        self._db.commit()
        self._db.set_authorizer(self._authorize)
        self._deadline = 0.0
        self._db.set_progress_handler(self._progress, 10_000)

    def _authorize(self, action: int, *_: Any) -> int:
        return sqlite3.SQLITE_OK if action in self._ALLOWED else sqlite3.SQLITE_DENY

    def _progress(self) -> int:
        return 1 if self._deadline and time.monotonic() > self._deadline else 0

    def schema(self) -> str:
        """Return the tables, columns, types, and descriptions as text."""
        parts = []
        for table in self.tables:
            head = f"{table.name} ({self.rows[table.name]:,} rows)"
            if table.description:
                head += f": {table.description}"
            cols = [
                f"  {c.name} {c.type}"
                + (f" -- {c.description}" if c.description else "")
                for c in self.columns[table.name]
            ]
            parts.append("\n".join([head, *cols]))
        return "\n".join(parts)

    def query(
        self, sql: str, *, max_rows: int = SQL_MAX_ROWS, timeout: float = SQL_TIMEOUT
    ) -> tuple[list[str], list[tuple[Any, ...]], bool]:
        """Run one read-only statement.

        Returns
        -------
        tuple
            Column names, up to ``max_rows`` rows, and whether more rows were
            left out.

        Raises
        ------
        sqlite3.Error
            On a syntax error, a write, or a query that ran past ``timeout``.
        """
        with self._lock:
            self._deadline = time.monotonic() + timeout
            try:
                cursor = self._db.execute(sql)
                names = [d[0] for d in cursor.description or ()]
                rows = cursor.fetchmany(max_rows + 1)
            finally:
                self._deadline = 0.0
        return names, rows[:max_rows], len(rows) > max_rows


# ---------------------------------------------------------------------------
# The tools


def _cap_lines(
    lines: Sequence[str], first: int, last: int, cap_chars: int
) -> tuple[str, int]:
    shown: list[str] = []
    used = 0
    stop = last
    for number in range(first, last + 1):
        line = f"{number}: {lines[number - 1]}"
        if shown and used + len(line) > cap_chars:
            stop = number - 1
            break
        shown.append(line[:cap_chars])
        used += len(line) + 1
    return "\n".join(shown), stop


def _escape_braces(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")


@dataclass
class ProfileTools:
    """The tools a profile generates over one corpus.

    Building one indexes the corpus: identifiers, sections, and symbols are
    found once, and the tables are loaded into SQLite. Nothing calls a model.

    Parameters
    ----------
    profile : DataProfile
        The profile, with a root.
    corpus : Corpus
        The documents, normally ``profile.corpus()``.
    """

    profile: DataProfile
    corpus: Corpus
    identifier_index: dict[str, dict[str, dict[str, tuple[int, int]]]] = field(
        init=False, default_factory=dict
    )
    sections: dict[str, list[Section]] = field(init=False, default_factory=dict)
    symbols: list[Symbol] = field(init=False, default_factory=list)
    tables: SqlTables | None = field(init=False, default=None)
    rule_of: dict[str, FileRule] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        """Index the corpus for every structure the profile declares."""
        profile = self.profile
        for kind in profile.identifiers:
            flags = re.IGNORECASE if kind.ignore_case else 0
            compiled = re.compile(kind.pattern, flags)
            values: dict[str, dict[str, tuple[int, int]]] = {}
            for doc in self.corpus.documents:
                for match in compiled.finditer(doc.text):
                    value = match.group(0)
                    if not value.strip():
                        continue
                    key = value.lower() if kind.ignore_case else value
                    seen = values.setdefault(key, {})
                    if doc.doc_id in seen:
                        line, count = seen[doc.doc_id]
                        seen[doc.doc_id] = (line, count + 1)
                    else:
                        line = doc.text.count("\n", 0, match.start()) + 1
                        seen[doc.doc_id] = (line, 1)
            self.identifier_index[kind.name] = values
        self.rule_of = profile.rules()
        for doc in self.corpus.documents:
            rule = self.rule_of.get(doc.doc_id)
            if rule is None:
                continue
            if rule.sections != "none":
                found = parse_sections(doc.text, rule.sections)
                if found:
                    self.sections[doc.doc_id] = found
            if (
                "code" in profile.structures
                and rule.kind == "code"
                and doc.doc_id.endswith(".py")
            ):
                module = module_name(
                    doc.doc_id, profile.code.package_root if profile.code else ""
                )
                self.symbols.extend(index_python(doc.doc_id, doc.text, module))
        if profile.tables:
            self.tables = SqlTables(profile.tables, profile._root())

    # -- what the agent is told --------------------------------------------

    @property
    def names(self) -> tuple[str, ...]:
        """Return the generated tool names, in a fixed order."""
        return tuple(
            name for s in self.profile.structures for name in PROFILE_TOOL_NAMES[s]
        )

    def summary(self) -> dict[str, Any]:
        """Return counts of what the indexes hold. No corpus text."""
        return {
            "documents": len(self.corpus),
            "tools": list(self.names),
            "identifier_kinds": {
                name: {
                    "distinct": len(values),
                    "documents": len({d for docs in values.values() for d in docs}),
                }
                for name, values in self.identifier_index.items()
            },
            "tables": dict(self.tables.rows) if self.tables else {},
            "documents_with_sections": len(self.sections),
            "sections": sum(len(s) for s in self.sections.values()),
            "symbols": len(self.symbols),
        }

    def describe(self, *, tools: bool = True) -> str:
        """Return the paragraph the agent reads about the corpus.

        Parameters
        ----------
        tools : bool, optional
            Name the generated tools. ``False`` describes the data alone, for
            an arm that keeps the generic tools.
        """
        profile = self.profile
        lines = [profile.description] if profile.description else []
        counts: dict[str, int] = {}
        for doc in self.corpus.documents:
            rule = self.rule_of.get(doc.doc_id)
            label = rule.glob if rule else "other"
            counts[label] = counts.get(label, 0) + 1
        lines.append(
            "Documents: "
            + "; ".join(f"{n:,} files matching {g}" for g, n in counts.items())
            + ". A document's doc_id is its path."
        )
        if profile.identifiers:
            kinds = "; ".join(
                f"{k.name} ({k.description or k.pattern}"
                + (f", for example {', '.join(k.examples[:3])}" if k.examples else "")
                + ")"
                for k in profile.identifiers
            )
            tail = (
                " find_identifier and list_identifiers look them up." if tools else ""
            )
            lines.append(f"Identifier kinds in the text: {kinds}.{tail}")
        if self.tables is not None:
            tail = " The sql tool queries them." if tools else ""
            lines.append("Tables:\n" + self.tables.schema() + tail)
        if self.sections:
            styles = sorted({r.sections for r in profile.files if r.sections != "none"})
            tail = (
                " outline lists a document's sections and open_section reads one."
                if tools
                else ""
            )
            lines.append(
                f"{len(self.sections):,} documents have sections ({', '.join(styles)} headings).{tail}"
            )
        if self.symbols:
            tail = (
                " find_symbol and open_symbol look definitions up by name."
                if tools
                else ""
            )
            lines.append(
                f"The code defines {len(self.symbols):,} Python classes, functions, "
                f"methods, and module-level names.{tail}"
            )
        return "\n".join(lines)

    def instruction(self, base: str, *, tools: bool = True) -> str:
        """Return a system prompt with the corpus description appended.

        ``base`` is a :func:`~aieng.advanced_rag.agents.session_loop.run_session`
        instruction with a ``{corpus}`` placeholder, which stays in place.
        """
        return (
            base
            + "\n\nAbout the corpus:\n"
            + _escape_braces(self.describe(tools=tools))
        )

    # -- registration --------------------------------------------------------

    def kinds(self) -> list[ToolKind]:
        """Return a :class:`ToolKind` for every generated tool."""
        builders: dict[
            str, tuple[Callable[[ToolSession], Callable[..., dict[str, Any]]], bool]
        ] = {
            "find_identifier": (self._find_identifier, False),
            "list_identifiers": (self._list_identifiers, False),
            "sql": (self._sql, False),
            "outline": (self._outline, False),
            "open_section": (self._open_section, True),
            "find_symbol": (self._find_symbol, False),
            "open_symbol": (self._open_symbol, True),
        }
        return [ToolKind(n, builders[n][0], builders[n][1]) for n in self.names]

    def register(self, server: CorpusToolServer) -> None:
        """Register every generated tool on a server over the same corpus."""
        for kind in self.kinds():
            server.register(kind)

    def arm(
        self, base: ArmConfig, *, name: str | None = None, generic: bool = True
    ) -> ArmConfig:
        """Return an arm offering the generated tools.

        Parameters
        ----------
        base : ArmConfig
            The arm whose limits and guards to keep, such as ``C7``.
        name : str or None, optional
            The new arm's label. ``P`` and the base's name when omitted.
        generic : bool, optional
            Keep the base arm's own tools beside the generated ones.
        """
        return base.with_(
            name=name or f"P{base.name}",
            tools=base.tools if generic else ("open",),
            extra_tools=self.names,
            description=(
                f"{base.name} tools plus {', '.join(self.names)}"
                if generic
                else f"open plus {', '.join(self.names)}"
            ),
        )

    # -- identifier tools ----------------------------------------------------

    def _identifier_kind(self, value: str) -> list[tuple[IdentifierKind, str]]:
        found = []
        for kind in self.profile.identifiers:
            key = value.lower() if kind.ignore_case else value
            if key in self.identifier_index[kind.name]:
                found.append((kind, key))
        return found

    def _find_identifier(self, session: ToolSession) -> Callable[..., dict[str, Any]]:
        tools = self

        def find_identifier(value: str) -> dict[str, Any]:
            started = time.perf_counter()
            arguments = {"value": value}
            value = value.strip()
            hits = tools._identifier_kind(value)
            if not hits:
                needle = value.lower()
                close = sorted(
                    {
                        (kind.name, key)
                        for kind in tools.profile.identifiers
                        for key in tools.identifier_index[kind.name]
                        if needle and needle in key.lower()
                    },
                    key=lambda pair: (
                        -len(tools.identifier_index[pair[0]][pair[1]]),
                        pair[1],
                    ),
                )[:20]
                result: dict[str, Any] = {
                    "total_documents": 0,
                    "matches": [],
                    "note": "no declared identifier is exactly this value"
                    + (
                        "; identifiers containing it are listed"
                        if close
                        else "; try grep"
                    ),
                }
                if close:
                    result["similar"] = [
                        {
                            "kind": k,
                            "value": v,
                            "documents": len(tools.identifier_index[k][v]),
                        }
                        for k, v in close
                    ]
                session.record(
                    "find_identifier", arguments, result, total_hits=0, started=started
                )
                return result
            docs: dict[str, tuple[int, int]] = {}
            for kind, key in hits:
                for doc_id, (line, count) in tools.identifier_index[kind.name][
                    key
                ].items():
                    old = docs.get(doc_id)
                    docs[doc_id] = (
                        (min(line, old[0]), count + old[1]) if old else (line, count)
                    )
            ordered = sorted(docs, key=lambda d: (-docs[d][1], d))
            limit = session.arm.grep_max_results
            matches = []
            for doc_id in ordered[:limit]:
                doc = tools.corpus.get(doc_id)
                line, count = docs[doc_id]
                text = doc.text.splitlines()[line - 1] if doc else ""
                matches.append(
                    {
                        "doc_id": doc_id,
                        "title": doc.title if doc else doc_id,
                        "line": line,
                        "count": count,
                        "preview": text.strip()[: session.arm.grep_preview_chars],
                    }
                )
            result = {
                "kinds": [kind.name for kind, _ in hits],
                "total_documents": len(docs),
                "matches": matches,
            }
            if len(docs) > limit:
                result["note"] = (
                    f"showing the {limit} documents that mention it most, of {len(docs)}"
                )
            session.record(
                "find_identifier",
                arguments,
                result,
                returned=[str(m["doc_id"]) for m in matches],
                total_hits=len(docs),
                started=started,
            )
            return result

        kinds = "; ".join(
            f"{k.name}: {k.description or k.pattern}" for k in self.profile.identifiers
        )
        find_identifier.__doc__ = f"""Find the documents that contain one identifier, exactly.

        The corpus was indexed for these identifier kinds: {kinds}. The match is
        exact and whole, so it never confuses one identifier with a longer one
        that contains it. Documents that mention it most come first. A value that
        is not an indexed identifier returns similar indexed values instead.

        Parameters
        ----------
        value : str
            The identifier as it appears in the text.

        Returns
        -------
        dict
            ``total_documents`` and ``matches``, each with ``doc_id``, ``title``,
            ``line`` of the first mention, ``count``, and that line as ``preview``.
        """
        return find_identifier

    def _list_identifiers(self, session: ToolSession) -> Callable[..., dict[str, Any]]:
        tools = self
        names = [k.name for k in self.profile.identifiers]

        def list_identifiers(kind: str, contains: str = "") -> dict[str, Any]:
            started = time.perf_counter()
            arguments = {"kind": kind, "contains": contains}
            values = tools.identifier_index.get(kind)
            if values is None:
                result: dict[str, Any] = {
                    "error": f"unknown kind {kind!r}; kinds are {names}"
                }
                session.record(
                    "list_identifiers",
                    arguments,
                    result,
                    error=result["error"],
                    started=started,
                )
                return result
            needle = contains.lower().strip()
            chosen = (
                [v for v in values if needle in v.lower()] if needle else list(values)
            )
            chosen.sort(key=lambda v: (-len(values[v]), v))
            result = {
                "kind": kind,
                "total_distinct": len(chosen),
                "values": [
                    {"value": v, "documents": len(values[v])} for v in chosen[:50]
                ],
            }
            if len(chosen) > 50:
                result["note"] = (
                    "showing the 50 values in the most documents; narrow with contains"
                )
            session.record(
                "list_identifiers",
                arguments,
                result,
                total_hits=len(chosen),
                started=started,
            )
            return result

        list_identifiers.__doc__ = f"""List the distinct identifiers of one kind, with how many documents hold each.

        Kinds: {"; ".join(f"{k.name} ({k.description or k.pattern})" for k in self.profile.identifiers)}.
        Use it to see which identifiers exist before looking one up, or to count them.

        Parameters
        ----------
        kind : str
            One of: {", ".join(names)}.
        contains : str
            Keep only values containing this text, ignoring case. Empty keeps all.

        Returns
        -------
        dict
            ``total_distinct`` and up to 50 ``values`` with their document counts,
            most widespread first.
        """
        return list_identifiers

    # -- the SQL tool --------------------------------------------------------

    def _sql(self, session: ToolSession) -> Callable[..., dict[str, Any]]:
        tools = self
        tables = self.tables
        if tables is None:  # pragma: no cover - names omit sql without tables
            raise ProfileError("the profile declares no tables")
        doc_columns = {t.doc_column for t in self.profile.tables if t.doc_column}

        def sql(query: str) -> dict[str, Any]:
            started = time.perf_counter()
            arguments = {"query": query}
            try:
                names, rows, more = tables.query(query)
            except sqlite3.Error as error:
                message = str(error)
                if "interrupted" in message:
                    message = f"the query ran longer than {SQL_TIMEOUT:.0f} seconds and was stopped"
                elif "not authorized" in message:
                    message = "only SELECT queries are allowed"
                result: dict[str, Any] = {"error": message}
                session.record("sql", arguments, result, error=message, started=started)
                return result
            cap = session.arm.output_cap_tokens * CHARS_PER_TOKEN
            shown: list[list[Any]] = []
            used = 0
            for row in rows:
                cells = [
                    c if not isinstance(c, str) or len(c) <= 500 else c[:500] + "..."
                    for c in row
                ]
                size = len(json.dumps(cells, default=str))
                if shown and used + size > cap:
                    more = True
                    break
                shown.append(cells)
                used += size
            result = {"columns": names, "rows": shown}
            if more:
                result["note"] = (
                    f"showing the first {len(shown)} rows; add LIMIT, WHERE, or an aggregate"
                )
            if not shown:
                result["note"] = "no rows"
            returned = [
                str(cell)
                for row in shown
                for name, cell in zip(names, row, strict=False)
                if name in doc_columns
                and cell is not None
                and str(cell) in tools.corpus
            ]
            session.record(
                "sql",
                arguments,
                result,
                returned=list(dict.fromkeys(returned)),
                total_hits=len(shown),
                started=started,
            )
            return result

        sql.__doc__ = f"""Run one read-only SQLite SELECT over the corpus's tables.

        Use it for counts, filters, sorting, and exact values. Text comparisons
        are case-sensitive; use LIKE or lower() for case-insensitive matching.
        At most {SQL_MAX_ROWS} rows come back. The tables:
        {tables.schema().replace(chr(10), chr(10) + "        ")}

        Parameters
        ----------
        query : str
            One SELECT statement (WITH is allowed).

        Returns
        -------
        dict
            ``columns`` and ``rows``, or an ``error`` naming what went wrong.
        """
        return sql

    # -- section tools -------------------------------------------------------

    def _outline(self, session: ToolSession) -> Callable[..., dict[str, Any]]:
        tools = self

        def outline(doc_id: str, contains: str = "") -> dict[str, Any]:
            started = time.perf_counter()
            arguments = {"doc_id": doc_id, "contains": contains}
            if doc_id not in tools.corpus:
                result: dict[str, Any] = {
                    "error": f"no document {doc_id!r}; use a doc_id from a result"
                }
                session.record(
                    "outline", arguments, result, error=result["error"], started=started
                )
                return result
            sections = tools.sections.get(doc_id, [])
            needle = contains.lower().strip()
            chosen = (
                [s for s in sections if needle in s.title.lower()]
                if needle
                else sections
            )
            cap = session.arm.output_cap_tokens * CHARS_PER_TOKEN
            listed: list[dict[str, Any]] = []
            used = 0
            for s in chosen:
                entry = {
                    "number": s.number,
                    "level": s.level,
                    "title": s.title[:200],
                    "lines": f"{s.start_line}-{s.end_line}",
                }
                size = len(json.dumps(entry))
                if listed and used + size > cap:
                    break
                listed.append(entry)
                used += size
            result = {
                "doc_id": doc_id,
                "total_sections": len(chosen),
                "sections": listed,
            }
            if len(listed) < len(chosen):
                result["note"] = (
                    f"showing {len(listed)} of {len(chosen)}; filter with contains"
                )
            if not sections:
                result["note"] = "this document has no sections; read it with open"
            session.record(
                "outline", arguments, result, returned=[doc_id], started=started
            )
            return result

        outline.__doc__ = """List a document's sections: number, heading level, title, and line range.

        Parameters
        ----------
        doc_id : str
            A ``doc_id`` from an earlier result.
        contains : str
            Keep only sections whose title contains this text, ignoring case.

        Returns
        -------
        dict
            ``total_sections`` and ``sections``. Read one with open_section.
        """
        return outline

    def _open_section(self, session: ToolSession) -> Callable[..., dict[str, Any]]:
        tools = self

        def open_section(doc_id: str, section: str) -> dict[str, Any]:
            started = time.perf_counter()
            arguments = {"doc_id": doc_id, "section": section}
            doc = tools.corpus.get(doc_id)
            sections = tools.sections.get(doc_id, [])
            chosen = _pick_section(sections, section)
            if doc is None or chosen is None:
                message = (
                    f"no document {doc_id!r}"
                    if doc is None
                    else f"no section {section!r}; list them with outline"
                )
                result: dict[str, Any] = {"error": message}
                session.record(
                    "open_section", arguments, result, error=message, started=started
                )
                return result
            lines = doc.text.splitlines() or [""]
            cap = session.arm.output_cap_tokens * CHARS_PER_TOKEN
            text, stop = _cap_lines(lines, chosen.start_line, chosen.end_line, cap)
            result = {
                "doc_id": doc_id,
                "section": chosen.title,
                "number": chosen.number,
                "lines": f"{chosen.start_line}-{chosen.end_line}",
                "text": text,
            }
            if stop < chosen.end_line:
                result["note"] = (
                    f"output cap reached; continue with open(doc_id={doc_id!r}, "
                    f"start_line={stop + 1}, end_line={chosen.end_line})"
                )
            session.mark_read(doc_id, chosen.start_line, chosen.end_line)
            session.record(
                "open_section",
                arguments,
                result,
                returned=[doc_id],
                read=[doc_id],
                started=started,
            )
            return result

        open_section.__doc__ = """Read one section of a document in full, with line numbers.

        Parameters
        ----------
        doc_id : str
            A ``doc_id`` from an earlier result.
        section : str
            The section's number from outline, or its title (an exact title
            wins, then the first title containing the text, ignoring case).

        Returns
        -------
        dict
            The section's ``title``, ``lines``, and ``text``. A long section stops
            early with a ``note`` naming the line to continue from.
        """
        return open_section

    # -- code tools ----------------------------------------------------------

    def _find_symbol(self, session: ToolSession) -> Callable[..., dict[str, Any]]:
        tools = self

        def find_symbol(name: str) -> dict[str, Any]:
            started = time.perf_counter()
            arguments = {"name": name}
            found = _match_symbols(tools.symbols, name.strip())
            listed = [
                {
                    "qualname": s.qualname,
                    "kind": s.kind,
                    "doc_id": s.doc_id,
                    "lines": f"{s.start_line}-{s.end_line}",
                    "signature": s.signature[:200],
                    "summary": s.summary[:200],
                }
                for s in found[:30]
            ]
            result: dict[str, Any] = {"total": len(found), "symbols": listed}
            if len(found) > 30:
                result["note"] = (
                    "showing 30; use a qualified name such as module.Class.method"
                )
            if not found:
                result["note"] = "no definition matches; grep for uses instead"
            session.record(
                "find_symbol",
                arguments,
                result,
                returned=list(dict.fromkeys(s.doc_id for s in found[:30])),
                total_hits=len(found),
                started=started,
            )
            return result

        find_symbol.__doc__ = """Find Python definitions by name: classes, functions, methods, and module-level names.

        An exact qualified name wins, then definitions whose last name part
        equals ``name``, then names that end with it, then names containing it,
        ignoring case.

        Parameters
        ----------
        name : str
            A name such as ``invoke`` or a qualified one such as ``pkg.core.Command.invoke``.

        Returns
        -------
        dict
            ``total`` and ``symbols``, each with ``qualname``, ``kind``, ``doc_id``,
            ``lines``, the ``signature`` line, and the docstring's first line.
        """
        return find_symbol

    def _open_symbol(self, session: ToolSession) -> Callable[..., dict[str, Any]]:
        tools = self

        def open_symbol(name: str) -> dict[str, Any]:
            started = time.perf_counter()
            arguments = {"name": name}
            found = _match_symbols(tools.symbols, name.strip())
            exact = [s for s in found if s.qualname == name.strip()]
            pick = exact[0] if exact else (found[0] if len(found) == 1 else None)
            if pick is None:
                message = (
                    "no definition matches; use find_symbol"
                    if not found
                    else f"{len(found)} definitions match; pass one qualname"
                )
                result: dict[str, Any] = {"error": message}
                if found:
                    result["candidates"] = [s.qualname for s in found[:20]]
                session.record(
                    "open_symbol",
                    arguments,
                    result,
                    error=message,
                    started=started,
                    returned=list(dict.fromkeys(s.doc_id for s in found[:20])),
                )
                return result
            doc = tools.corpus.get(pick.doc_id)
            lines = doc.text.splitlines() if doc else [""]
            cap = session.arm.output_cap_tokens * CHARS_PER_TOKEN
            text, stop = _cap_lines(lines, pick.start_line, pick.end_line, cap)
            result = {
                "qualname": pick.qualname,
                "doc_id": pick.doc_id,
                "lines": f"{pick.start_line}-{pick.end_line}",
                "text": text,
            }
            if stop < pick.end_line:
                result["note"] = (
                    f"output cap reached; continue with open(doc_id={pick.doc_id!r}, "
                    f"start_line={stop + 1}, end_line={pick.end_line})"
                )
            session.mark_read(pick.doc_id, pick.start_line, pick.end_line)
            session.record(
                "open_symbol",
                arguments,
                result,
                returned=[pick.doc_id],
                read=[pick.doc_id],
                started=started,
            )
            return result

        open_symbol.__doc__ = """Read one Python definition's source, with line numbers.

        Parameters
        ----------
        name : str
            A qualified name from find_symbol, or a name only one definition has.

        Returns
        -------
        dict
            ``qualname``, ``doc_id``, ``lines``, and ``text``, or an ``error``
            with ``candidates`` when the name is ambiguous.
        """
        return open_symbol


def _pick_section(sections: Sequence[Section], wanted: str) -> Section | None:
    wanted = str(wanted).strip()
    if wanted.isdigit():
        number = int(wanted)
        return next((s for s in sections if s.number == number), None)
    lowered = wanted.lower()
    return next((s for s in sections if s.title.lower() == lowered), None) or next(
        (s for s in sections if lowered and lowered in s.title.lower()), None
    )


def _match_symbols(symbols: Sequence[Symbol], name: str) -> list[Symbol]:
    if not name:
        return []
    exact = [s for s in symbols if s.qualname == name]
    if exact:
        return exact
    last = name.rsplit(".", 1)[-1]
    tiers = (
        [s for s in symbols if s.name == name],
        [s for s in symbols if s.qualname.endswith("." + name)],
        [
            s
            for s in symbols
            if "." in name
            and s.name == last
            and s.qualname.endswith(name.split(".", 1)[-1])
        ],
        [s for s in symbols if name.lower() in s.qualname.lower()],
    )
    order = {"class": 0, "function": 1, "method": 2, "variable": 3}
    for tier in tiers:
        if tier:
            return sorted(tier, key=lambda s: (order.get(s.kind, 4), s.qualname))
    return []


def build_server(
    profile: DataProfile,
    *,
    search: Any = None,
    grep: Any = None,
    corpus: Corpus | None = None,
) -> tuple[CorpusToolServer, ProfileTools]:
    """Return a tool server over a profile's corpus with the generated tools registered.

    Parameters
    ----------
    profile : DataProfile
        The profile, with a root.
    search : callable or None, optional
        The search backend, such as a
        :class:`~aieng.advanced_rag.agents.corpus_tools.DenseSearch`.
    grep : GrepBackend or None, optional
        The grep backend. A regular-expression grep over the corpus when omitted.
    corpus : Corpus or None, optional
        The documents, when already read. ``profile.corpus()`` otherwise.

    Returns
    -------
    tuple
        The server and the :class:`ProfileTools`, whose ``arm`` adds the
        generated tools to an arm.
    """
    documents = corpus if corpus is not None else profile.corpus()
    tools = ProfileTools(profile, documents)
    server = CorpusToolServer(documents, search=search, grep=grep)
    tools.register(server)
    return server, tools


__all__ = [
    "PROFILE_TOOL_NAMES",
    "SQL_MAX_ROWS",
    "SQL_TIMEOUT",
    "CodeSpec",
    "Column",
    "DataProfile",
    "FileRule",
    "IdentifierKind",
    "ProfileError",
    "ProfileTools",
    "Section",
    "SqlTables",
    "Symbol",
    "TableSpec",
    "build_server",
    "index_python",
    "load_profile",
    "module_name",
    "parse_sections",
    "profile_from_dict",
]
