"""X3: relational questions over the ServerFault graph against retrieval over posts.

Four question families, generated from graph patterns with a fixed seed and no
model call:

* ``count``: "How many posts mention X?"
* ``co_mention``: "Which posts mention both X and Y?" (a complete list of posts)
* ``relation``: "Across all posts, which errors are reported as caused by X?" and
  the ``depends_on`` and ``fixes`` variants (a complete list of entities)
* ``path``: "Which entities link X and Y?", where no single post mentions both,
  so the answer exists only across posts.

Each question carries a graph gold (the answer a query over the graph returns)
and, for the two mention families, a lexical gold (the posts whose text contains
the name, whole word, ignoring case), which does not depend on the graph. The
distance between the two golds is the graph's own coverage of the corpus.

The graph is loaded into SQLite as three tables (``entities``, ``mentions``,
``relations``) so generated queries run against the same nodes and edges the
templates read. SQL stands in for Cypher because the locked environment has no
graph database; the failure under test, a generated query that runs and
returns nothing, is the same in both.
"""

from __future__ import annotations

import json
import random
import re
import sqlite3
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


SEED = 20261023
PER_FAMILY = 15
FAMILIES = ("count", "co_mention", "relation", "path")
_DOC = re.compile(r"doc:(q\d+)")
_SECTION = re.compile(r"section:([^|]+)")
#: Names too generic to ask about: they match most posts and no reader can list them.
GENERIC = frozenset(
    {
        "system",
        "server",
        "service",
        "file",
        "files",
        "user",
        "users",
        "network",
        "operating system",
        "web application",
        "application",
        "database server",
        "monitoring system",
        "config",
        "client",
        "linux",
        "windows",
        "web server",
        "computer",
        "machine",
        "host",
        "port",
        "software",
        "error",
    }
)
RELATION_WORDING = {
    "causes": ("ErrorOrSymptom", "Across all posts, which errors or symptoms are reported as caused by {x}?"),
    "depends_on": (None, "Across all posts, what is {x} reported to depend on?"),
    "fixes": (None, "Across all posts, what is reported to fix {x}?"),
}


@dataclass(frozen=True)
class Graph:
    """The ServerFault graph reduced to what the questions read."""

    types: Mapping[str, str]
    entity_docs: Mapping[str, frozenset[str]]
    relations: Sequence[tuple[str, str, str, frozenset[str]]]

    @classmethod
    def load(cls, path: Path) -> Graph:
        """Read ``graph.json`` as ``build_serverfault_graph.py`` writes it."""
        raw = json.loads(path.read_text(encoding="utf-8"))
        types = {e["name"]: e["type"] for e in raw["entities"]}
        entity_docs = {e["name"]: frozenset(_docs(e["mentions"])) for e in raw["entities"]}
        relations = [(r["source"], r["type"], r["target"], frozenset(_docs(r["mentions"]))) for r in raw["relations"]]
        return cls(types, entity_docs, relations)

    def neighbours(self) -> dict[str, set[str]]:
        """Return each entity's neighbours over any relation, ignoring direction."""
        out: dict[str, set[str]] = defaultdict(set)
        for source, _, target, _ in self.relations:
            if source != target:
                out[source].add(target)
                out[target].add(source)
        return out


def _docs(mentions: Iterable[Mapping[str, Any]]) -> set[str]:
    found = set()
    for mention in mentions:
        match = _DOC.match(mention.get("provenance", ""))
        if match:
            found.add(match.group(1))
    return found


@dataclass
class Question:
    """One relational question with its golds; holds no post text."""

    qid: str
    family: str
    question: str
    entities: tuple[str, ...]
    relation: str | None
    graph_gold: list[str] | int
    lexical_gold: list[str] | int | None = None
    evidence_docs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return asdict(self)


def name_pattern(name: str) -> re.Pattern[str]:
    """Return a case-insensitive whole-word pattern for a name."""
    return re.compile(r"(?<![\w-])" + re.escape(name) + r"(?![\w-])", re.IGNORECASE)


def lexical_docs(name: str, texts: Mapping[str, str]) -> set[str]:
    """Return the posts whose text contains the name as a whole word."""
    pattern = name_pattern(name)
    return {doc for doc, text in texts.items() if pattern.search(text)}


SPECIFIC_TYPES = frozenset({"Software", "Protocol", "Command", "Hardware"})


def generate(graph: Graph, texts: Mapping[str, str], per_family: int = PER_FAMILY, seed: int = SEED) -> list[Question]:
    """Return ``per_family`` questions per family from graph patterns, seeded."""
    rng = random.Random(seed)
    specific = sorted(n for n, t in graph.types.items() if t in SPECIFIC_TYPES and n not in GENERIC and len(n) >= 3)
    return [
        *_count_questions(graph, texts, specific, rng, per_family),
        *_co_mention_questions(graph, texts, specific, rng, per_family),
        *_relation_questions(graph, rng, per_family),
        *_path_questions(graph, specific, rng, per_family),
    ]


def _count_questions(
    graph: Graph, texts: Mapping[str, str], specific: Sequence[str], rng: random.Random, limit: int
) -> list[Question]:
    # The graph cites 6 to 30 posts, and the name is specific enough that grep stays
    # under 100.
    pool = [n for n in specific if 6 <= len(graph.entity_docs[n]) <= 30]
    rng.shuffle(pool)
    out: list[Question] = []
    for name in pool:
        lexical = lexical_docs(name, texts)
        if not 3 <= len(lexical) <= 100:
            continue
        docs = graph.entity_docs[name]
        out.append(
            Question(
                f"count-{len(out):02d}",
                "count",
                f"How many posts in the collection mention {name}?",
                (name,),
                None,
                len(docs),
                len(lexical),
                sorted(docs),
            )
        )
        if len(out) == limit:
            break
    return out


def _co_mention_questions(
    graph: Graph, texts: Mapping[str, str], specific: Sequence[str], rng: random.Random, limit: int
) -> list[Question]:
    # Two specific entities that the graph places together in 2 to 10 posts.
    frequent = [n for n in specific if 4 <= len(graph.entity_docs[n]) <= 80]
    by_doc: dict[str, list[str]] = defaultdict(list)
    for n in frequent:
        for d in graph.entity_docs[n]:
            by_doc[d].append(n)
    seen: set[tuple[str, str]] = set()
    for names in by_doc.values():
        ordered = sorted(names)
        for i, a in enumerate(ordered):
            seen.update((a, b) for b in ordered[i + 1 :])
    pairs = [
        (a, b)
        for a, b in sorted(seen)
        if 2 <= len(graph.entity_docs[a] & graph.entity_docs[b]) <= 10
        and a.lower() not in b.lower()
        and b.lower() not in a.lower()
    ]
    rng.shuffle(pairs)
    used: set[str] = set()
    out: list[Question] = []
    for a, b in pairs:
        if a in used or b in used:
            continue
        lexical = lexical_docs(a, texts) & lexical_docs(b, texts)
        if not 1 <= len(lexical) <= 40:
            continue
        used.update((a, b))
        shared = sorted(graph.entity_docs[a] & graph.entity_docs[b])
        out.append(
            Question(
                f"co_mention-{len(out):02d}",
                "co_mention",
                f"Which posts mention both {a} and {b}?",
                (a, b),
                None,
                shared,
                sorted(lexical),
                shared,
            )
        )
        if len(out) == limit:
            break
    return out


def _relation_questions(graph: Graph, rng: random.Random, limit: int) -> list[Question]:
    # An entity with 3 to 15 distinct targets of one relation type, from at least 2
    # posts.
    by_subject: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for source, rel, target, docs in graph.relations:
        if rel == "fixes":
            by_subject[(target, rel)][source].update(docs)
        elif rel in RELATION_WORDING:
            wanted = RELATION_WORDING[rel][0]
            if wanted is None or graph.types.get(target) == wanted:
                by_subject[(source, rel)][target].update(docs)
    candidates = []
    for (subject, rel), targets in sorted(by_subject.items()):
        subject_docs: set[str] = set().union(*targets.values())
        if subject not in GENERIC and len(subject) >= 3 and 3 <= len(targets) <= 15 and len(subject_docs) >= 2:
            candidates.append((subject, rel, targets, subject_docs))
    rng.shuffle(candidates)
    quota = {"causes": limit // 3 + limit % 3, "depends_on": limit // 3, "fixes": limit // 3}
    taken: dict[str, int] = defaultdict(int)
    out: list[Question] = []
    for subject, rel, targets, cited in candidates:
        if taken[rel] >= quota[rel]:
            continue
        taken[rel] += 1
        out.append(
            Question(
                f"relation-{len(out):02d}",
                "relation",
                RELATION_WORDING[rel][1].format(x=subject),
                (subject,),
                rel,
                sorted(targets),
                None,
                sorted(cited),
            )
        )
        if len(out) == limit:
            break
    return out


def _path_partners(graph: Graph, neighbours: Mapping[str, set[str]], a: str, used: set[str]) -> list[str]:
    """Return entities two hops from ``a`` that never share a post with it."""
    options = []
    for z in sorted(neighbours[a] - GENERIC):
        for b in sorted(neighbours.get(z, ())):
            if b == a or b in used or b in GENERIC or b in neighbours[a]:
                continue
            if graph.types.get(b) in SPECIFIC_TYPES and not graph.entity_docs[a] & graph.entity_docs[b]:
                options.append(b)
    return options


def _path_questions(graph: Graph, specific: Sequence[str], rng: random.Random, limit: int) -> list[Question]:
    # Two specific entities never in the same post, linked through 1 to 6 shared
    # neighbours.
    neighbours = graph.neighbours()
    linked = [n for n in specific if 3 <= len(graph.entity_docs[n]) <= 40 and 2 <= len(neighbours.get(n, ())) <= 40]
    rng.shuffle(linked)
    used: set[str] = set()
    out: list[Question] = []
    for a in linked:
        if len(out) == limit:
            break
        if a in used:
            continue
        options = _path_partners(graph, neighbours, a, used)
        rng.shuffle(options)
        for b in options:
            via = sorted((neighbours[a] & neighbours[b]) - GENERIC)
            if 1 <= len(via) <= 6:
                used.update((a, b))
                out.append(
                    Question(
                        f"path-{len(out):02d}",
                        "path",
                        f"Which concepts, tools, or settings link {a} and {b}, given that no single post discusses both?",
                        (a, b),
                        None,
                        via,
                        None,
                        sorted(graph.entity_docs[a] | graph.entity_docs[b]),
                    )
                )
                break
    return out


# ---------------------------------------------------------------- scoring


def normalise(text: str) -> str:
    """Lower-case a name and collapse punctuation and spaces."""
    return re.sub(r"[^a-z0-9./+#_-]+", " ", text.lower()).strip()


def names_match(a: str, b: str) -> bool:
    """Return whether two entity names refer to the same thing, leniently.

    Equal after normalising, one containing the other as whole words, or a token
    Jaccard of at least 0.5.
    """
    x, y = normalise(a), normalise(b)
    if not x or not y:
        return False
    if x == y:
        return True
    if re.search(r"(?<!\w)" + re.escape(x) + r"(?!\w)", y) or re.search(r"(?<!\w)" + re.escape(y) + r"(?!\w)", x):
        return True
    tx, ty = set(x.split()), set(y.split())
    return len(tx & ty) / len(tx | ty) >= 0.5


def set_scores(predicted: Sequence[str], gold: Sequence[str], match: Callable[[str, str], bool]) -> dict[str, float]:
    """Return precision, recall, and F1 of a predicted list against a gold list."""
    predicted = list(dict.fromkeys(p for p in predicted if p))
    if not gold and not predicted:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    hit_pred = sum(1 for p in predicted if any(match(p, g) for g in gold))
    hit_gold = sum(1 for g in gold if any(match(p, g) for p in predicted))
    precision = hit_pred / len(predicted) if predicted else 0.0
    recall = hit_gold / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def count_score(predicted: int | None, gold: int) -> dict[str, float]:
    """Return whether a count lands within 20% of the gold, and its relative error."""
    if predicted is None:
        return {"within_20pct": 0.0, "relative_error": 1.0}
    tolerance = max(1.0, 0.2 * gold)
    return {
        "within_20pct": float(abs(predicted - gold) <= tolerance),
        "relative_error": round(abs(predicted - gold) / max(gold, 1), 4),
    }


def score(question: Question, answer: Any) -> dict[str, Any]:
    """Score one answer against the graph gold and, where present, the lexical gold."""
    out: dict[str, Any] = {}
    golds = {"graph": question.graph_gold, "lexical": question.lexical_gold}
    for label, gold in golds.items():
        if gold is None:
            continue
        if question.family == "count":
            out[label] = count_score(answer if isinstance(answer, int) else None, int(gold))  # type: ignore[arg-type]
        elif question.family == "co_mention":
            out[label] = set_scores([str(a) for a in answer or []], list(gold), lambda p, g: p == g)  # type: ignore[arg-type]
        else:
            out[label] = set_scores([str(a) for a in answer or []], list(gold), names_match)  # type: ignore[arg-type]
    return out


# ---------------------------------------------------------------- graph access


def sqlite_graph(graph: Graph, path: str = ":memory:") -> sqlite3.Connection:
    """Load the graph into SQLite as entity, mention, and relation tables."""
    db = sqlite3.connect(path, check_same_thread=False)
    db.executescript(
        """
        CREATE TABLE entities (name TEXT PRIMARY KEY, type TEXT);
        CREATE TABLE mentions (entity TEXT, doc_id TEXT);
        CREATE TABLE relations (source TEXT, type TEXT, target TEXT, doc_id TEXT);
        """
    )
    db.executemany("INSERT INTO entities VALUES (?, ?)", sorted(graph.types.items()))
    db.executemany(
        "INSERT INTO mentions VALUES (?, ?)", ((n, d) for n, docs in graph.entity_docs.items() for d in sorted(docs))
    )
    db.executemany(
        "INSERT INTO relations VALUES (?, ?, ?, ?)",
        ((s, r, t, d) for s, r, t, docs in graph.relations for d in sorted(docs or {""})),
    )
    db.executescript(
        "CREATE INDEX m_e ON mentions(entity); CREATE INDEX m_d ON mentions(doc_id);"
        "CREATE INDEX r_s ON relations(source); CREATE INDEX r_t ON relations(target);"
    )
    db.commit()
    return db


def template_answer(question: Question, db: sqlite3.Connection) -> Any:
    """Answer with a fixed query per family, matching entity names exactly."""
    if question.family == "count":
        (n,) = db.execute("SELECT COUNT(DISTINCT doc_id) FROM mentions WHERE entity = ?", question.entities).fetchone()
        return int(n)
    if question.family == "co_mention":
        rows = db.execute(
            "SELECT DISTINCT a.doc_id FROM mentions a JOIN mentions b ON a.doc_id = b.doc_id "
            "WHERE a.entity = ? AND b.entity = ? ORDER BY a.doc_id",
            question.entities,
        )
        return [r[0] for r in rows]
    if question.family == "relation":
        (x,) = question.entities
        if question.relation == "fixes":
            rows = db.execute("SELECT DISTINCT source FROM relations WHERE type = 'fixes' AND target = ?", (x,))
        elif question.relation == "causes":
            rows = db.execute(
                "SELECT DISTINCT r.target FROM relations r JOIN entities e ON e.name = r.target "
                "WHERE r.type = 'causes' AND r.source = ? AND e.type = 'ErrorOrSymptom'",
                (x,),
            )
        else:
            rows = db.execute(
                "SELECT DISTINCT target FROM relations WHERE type = ? AND source = ?", (question.relation, x)
            )
        return sorted(r[0] for r in rows)
    a, b = question.entities
    rows = db.execute(
        """
        WITH n(x, y) AS (SELECT source, target FROM relations UNION SELECT target, source FROM relations)
        SELECT DISTINCT n1.y FROM n n1 JOIN n n2 ON n1.y = n2.y
        WHERE n1.x = ? AND n2.x = ? AND n1.y NOT IN (?, ?)
        """,
        (a, b, a, b),
    )
    return sorted(r[0] for r in rows if r[0] not in GENERIC)


SCHEMA_PROMPT = """You write one SQLite query over a knowledge graph extracted from ServerFault posts.

Tables:
- entities(name TEXT PRIMARY KEY, type TEXT). Names are lower case. Types: {types}.
- mentions(entity TEXT, doc_id TEXT): one row per entity and post that mentions it; doc_id looks like 'q9708'.
- relations(source TEXT, type TEXT, target TEXT, doc_id TEXT): one row per relation and the post it came from.
  Relation types: {relations}. 'A fixes B' means A is reported to fix B; 'A causes B' means A is reported to cause B.

Sample entity names: {samples}

Question: {question}

Return only the SQL, with no explanation. For a count, return one number. For a list of posts, return
doc_id values. For a list of entities, return entity names."""

REPAIR_PROMPT = """The query below {problem}. Write a corrected SQLite query for the same question over the
same tables. Return only the SQL.

Question: {question}
Query:
{sql}"""


def extract_sql(text: str) -> str:
    """Return the SQL inside a reply, stripping code fences and trailing prose."""
    fence = re.search(r"```(?:sql)?\s*(.+?)```", text, re.DOTALL | re.IGNORECASE)
    sql = fence.group(1) if fence else text
    return sql.strip().rstrip(";").strip()


def run_sql(db: sqlite3.Connection, sql: str, limit: int = 500) -> tuple[list[tuple[Any, ...]] | None, str | None]:
    """Run a read-only query; return (rows, None) or (None, error)."""
    if not re.match(r"^\s*(with|select)\b", sql, re.IGNORECASE):
        return None, "is not a SELECT query"
    try:
        return db.execute(sql).fetchmany(limit), None
    except sqlite3.Error as error:
        return None, f"failed with: {error}"


def rows_to_answer(family: str, rows: Sequence[tuple[Any, ...]]) -> Any:
    """Turn query rows into an answer of the family's shape."""
    if family == "count":
        if len(rows) == 1 and rows[0] and isinstance(rows[0][0], (int, float)):
            return int(rows[0][0])
        return len(rows)
    return sorted({str(r[0]) for r in rows if r and r[0] is not None})


def text2sql_answer(
    question: Question, db: sqlite3.Connection, complete: Callable[[str], str], samples: Sequence[str], repair: bool
) -> tuple[Any, dict[str, Any]]:
    """Answer with a generated query, with one optional repair."""
    prompt = SCHEMA_PROMPT.format(
        types=", ".join(
            sorted({"Software", "Protocol", "Command", "ConfigItem", "Concept", "ErrorOrSymptom", "Hardware"})
        ),
        relations="configures, depends_on, part_of, fixes, alternative_to, diagnoses, causes",
        samples=", ".join(samples),
        question=question.question,
    )
    sql = extract_sql(complete(prompt))
    rows, error = run_sql(db, sql)
    trace: dict[str, Any] = {"error": error is not None, "empty": rows is not None and not rows, "repaired": False}
    empty_count = rows is not None and question.family == "count" and len(rows) == 1 and rows[0] and rows[0][0] == 0
    if repair and (error or not rows or empty_count):
        problem = error or "returned no rows"
        sql = extract_sql(complete(REPAIR_PROMPT.format(problem=problem, question=question.question, sql=sql)))
        rows, error = run_sql(db, sql)
        trace["repaired"] = True
        trace["after_repair_error"] = error is not None
        trace["after_repair_empty"] = rows is not None and not rows
    return (
        rows_to_answer(question.family, rows) if rows is not None else ([] if question.family != "count" else None)
    ), trace


# ---------------------------------------------------------------- reading posts


READ_PROMPT = """Answer the question using only the posts below. Each post starts with its id in brackets.
{instruction}

Posts:
{posts}

Question: {question}
Answer:"""

INSTRUCTIONS = {
    "count": "Reply with one integer only.",
    "co_mention": 'Reply with JSON only: {"posts": ["q123", ...]}, listing every post id that qualifies.',
    "relation": 'Reply with JSON only: {"answers": ["...", ...]}, a complete list of short names.',
    "path": 'Reply with JSON only: {"answers": ["...", ...]}, a complete list of short names.',
}

MAP_PROMPT = """Read one ServerFault post and answer about it alone.
{task}

Post [{doc_id}]:
{post}

Reply with JSON only: {schema}"""


def map_task(question: Question) -> tuple[str, str]:
    """Return the per-post task and reply schema for the map step of a question."""
    if question.family == "count":
        return f"Does this post mention {question.entities[0]}?", '{"yes": true or false}'
    if question.family == "co_mention":
        a, b = question.entities
        return f"Does this post mention both {a} and {b}?", '{"yes": true or false}'
    if question.family == "relation":
        x = question.entities[0]
        what = {
            "causes": f"List the errors or symptoms this post says are caused by {x}.",
            "depends_on": f"List what this post says {x} depends on.",
            "fixes": f"List what this post says fixes {x}.",
        }[question.relation or "causes"]
        return what + " An empty list if none.", '{"answers": ["...", ...]}'
    a, b = question.entities
    return (
        f"List the concepts, tools, or settings this post relates directly to {a} or to {b}. "
        "An empty list if the post discusses neither.",
        '{"related_to_a": ["..."], "related_to_b": ["..."]}',
    )


def parse_json(text: str) -> dict[str, Any]:
    """Return the first JSON object in a reply, or an empty dictionary."""
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    body = fence.group(1) if fence else text
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        value = json.loads(body[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def parse_read(family: str, text: str) -> Any:
    """Turn a reader reply into an answer of the family's shape."""
    if family == "count":
        found = re.search(r"-?\d[\d,]*", text)
        return int(found.group(0).replace(",", "")) if found else None
    data = parse_json(text)
    key = "posts" if family == "co_mention" else "answers"
    values = data.get(key, [])
    if not isinstance(values, list):
        return []
    if family == "co_mention":
        return sorted({v for v in (str(x).strip() for x in values) if re.fullmatch(r"q\d+", v)})
    return [str(v).strip() for v in values if str(v).strip()]


def reduce_map(question: Question, replies: Mapping[str, str]) -> Any:
    """Combine per-post replies into one answer."""
    parsed = {doc: parse_json(text) for doc, text in replies.items()}
    if question.family in {"count", "co_mention"}:
        yes = sorted(doc for doc, data in parsed.items() if data.get("yes") is True)
        return len(yes) if question.family == "count" else yes
    if question.family == "relation":
        names: list[str] = []
        for data in parsed.values():
            names.extend(str(v) for v in data.get("answers", []) if isinstance(v, str))
        return _dedupe(names)
    a_side: list[str] = []
    b_side: list[str] = []
    for data in parsed.values():
        a_side.extend(str(v) for v in data.get("related_to_a", []) if isinstance(v, str))
        b_side.extend(str(v) for v in data.get("related_to_b", []) if isinstance(v, str))
    a, b = question.entities
    return _dedupe(
        [
            x
            for x in a_side
            if any(names_match(x, y) for y in b_side) and not names_match(x, a) and not names_match(x, b)
        ]
    )


def _dedupe(names: Iterable[str]) -> list[str]:
    out: list[str] = []
    for name in names:
        if name.strip() and not any(names_match(name, kept) for kept in out):
            out.append(name.strip())
    return out


def mention_coverage(graph: Graph, texts: Mapping[str, str], questions: Sequence[Question]) -> dict[str, float]:
    """Return how much of the lexical mention gold the graph reaches.

    Over the ``count`` questions: the share of (name, post) pairs found by whole-word
    match that the graph's exact entity also cites, and that any entity whose name
    contains the question's name cites.
    """
    lexical_total = exact = substring = 0
    for q in questions:
        if q.family != "count":
            continue
        name = q.entities[0]
        lex = lexical_docs(name, texts)
        variants = set().union(*(docs for n, docs in graph.entity_docs.items() if name.lower() in n.lower()))
        lexical_total += len(lex)
        exact += len(lex & graph.entity_docs.get(name, frozenset()))
        substring += len(lex & variants)
    return {
        "lexical_mentions": lexical_total,
        "exact_name_recall": round(exact / lexical_total, 4) if lexical_total else 0.0,
        "substring_recall": round(substring / lexical_total, 4) if lexical_total else 0.0,
    }


class RelationalArms:
    """The X3 arms over one corpus and graph, each answering a :class:`Question`.

    Parameters
    ----------
    texts : mapping
        Post text by document identifier.
    db : sqlite3.Connection
        The graph tables from :func:`sqlite_graph`.
    search : callable
        ``search(query, k)`` returning ranked post identifiers.
    complete : callable
        ``complete(prompt) -> text`` for the reader; called from worker threads.
    samples : sequence of str
        Entity names shown to the query writer as examples.
    workers : int, optional
        Concurrent reader calls in a map step.
    """

    names = ("dense", "dense-map", "grep", "grep-map", "graph-template", "graph-like", "text2sql", "text2sql-repair")

    def __init__(
        self,
        texts: Mapping[str, str],
        db: sqlite3.Connection,
        search: Callable[[str, int], Sequence[str]],
        complete: Callable[[str], str],
        *,
        samples: Sequence[str],
        workers: int = 16,
    ) -> None:
        from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

        self.texts, self.db, self.search, self.complete, self.samples = texts, db, search, complete, samples
        self._pool = ThreadPoolExecutor(workers)

    def answer(self, arm: str, question: Question) -> tuple[Any, dict[str, Any] | None]:
        """Return one arm's answer to a question, and a query trace for the SQL arms."""
        if arm.startswith("text2sql"):
            return text2sql_answer(question, self.db, self.complete, self.samples, repair=arm == "text2sql-repair")
        method = {
            "dense": self._dense,
            "dense-map": self._dense_map,
            "grep": self._grep,
            "grep-map": self._grep_map,
            "graph-template": lambda q: template_answer(q, self.db),
            "graph-like": self._like,
        }[arm]
        return method(question), None

    def _read(self, question: Question, docs: Sequence[str]) -> Any:
        posts = "\n\n".join(f"[{d}] {self.texts[d][:1500]}" for d in docs)
        prompt = READ_PROMPT.format(instruction=INSTRUCTIONS[question.family], posts=posts, question=question.question)
        return parse_read(question.family, self.complete(prompt))

    def _map(self, question: Question, docs: Sequence[str]) -> Any:
        task, schema = map_task(question)
        prompts = {d: MAP_PROMPT.format(task=task, doc_id=d, post=self.texts[d][:2500], schema=schema) for d in docs}
        return reduce_map(question, dict(zip(prompts, self._pool.map(self.complete, prompts.values()), strict=True)))

    def _grep_docs(self, question: Question) -> list[str]:
        sets = [lexical_docs(name, self.texts) for name in question.entities]
        if question.family == "co_mention":
            return sorted(sets[0] & sets[1])
        return sorted(set().union(*sets))

    def _dense(self, question: Question) -> Any:
        return self._read(question, self.search(question.question, 10))

    def _dense_map(self, question: Question) -> Any:
        return self._map(question, self.search(question.question, 50))

    def _grep(self, question: Question) -> Any:
        hits = self._grep_docs(question)
        if question.family == "count":
            return len(hits)
        if question.family == "co_mention":
            return hits
        return self._read(question, hits[:10])

    def _grep_map(self, question: Question) -> Any:
        if question.family == "path":
            # Up to 25 posts per entity, so both ends of the path are read.
            a, b = (sorted(lexical_docs(n, self.texts))[:25] for n in question.entities)
            return self._map(question, sorted(set(a) | set(b)))
        return self._map(question, self._grep_docs(question)[:50])

    def _like(self, question: Question) -> Any:
        # Substring match over entity names: what a reader of the graph does when exact
        # names miss variants.
        def docs_like(name: str) -> set[str]:
            rows = self.db.execute("SELECT DISTINCT doc_id FROM mentions WHERE entity LIKE ?", (f"%{name}%",))
            return {r[0] for r in rows}

        if question.family == "count":
            return len(docs_like(question.entities[0]))
        if question.family == "co_mention":
            return sorted(docs_like(question.entities[0]) & docs_like(question.entities[1]))
        return template_answer(question, self.db)
