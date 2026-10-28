"""The team-facing settings: who asks, what they ask, how many, and how it is checked.

The shape follows DataMorgana (Filice et al., ACL 2025 Industry): weighted user
categories and question categories, combined per question into the prompt, so
the set's mix is a stated input rather than whatever the generator drifts to.
Few-shot examples are a handful of real questions from the team's users.
Promptagator (Dai et al., 2022) moved its query distribution with eight.

A team fills in ``team-config.example.yaml`` next to this module and loads it
with :func:`load_config`. JSON works too, and needs no YAML parser.
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Category:
    """One weighted option in a category list.

    Parameters
    ----------
    name : str
        Short label, recorded on every question that used it.
    description : str
        What the generator is told about it.
    weight : float, optional
        Relative sampling weight.
    """

    name: str
    description: str
    weight: float = 1.0


#: Who is asking. Domain-neutral defaults; a team replaces them.
DEFAULT_USERS: tuple[Category, ...] = (
    Category(
        "practitioner",
        "Someone who works with these documents daily and uses the field's "
        "vocabulary, asking a short, precise question.",
        2.0,
    ),
    Category(
        "newcomer",
        "Someone outside the field who describes what they want in everyday "
        "words and does not know the document's terms.",
        1.0,
    ),
)

#: What they ask. Each one is a different kind of lookup.
DEFAULT_QUESTIONS: tuple[Category, ...] = (
    Category(
        "factoid",
        "A question with a short answer: a name, number, period, or party.",
        2.0,
    ),
    Category(
        "yes_no",
        "A question answered yes or no, with the condition that decides it.",
        1.0,
    ),
    Category(
        "condition_or_exception",
        "A question about when something applies, or what is excluded from it.",
        1.5,
    ),
    Category(
        "procedure",
        "A question about what someone has to do, and how or when.",
        1.0,
    ),
)

#: Surface form, sampled independently of the category.
DEFAULT_PHRASINGS: tuple[Category, ...] = (
    Category("natural", "A complete, natural question, as typed into a chat box.", 3.0),
    Category("keywords", "A terse search-engine style query of a few words.", 1.0),
)


@dataclass(frozen=True)
class GenerationConfig:
    """Everything a build reads, besides the corpus and the models.

    Parameters
    ----------
    domain : str, optional
        One line naming the documents, such as "non-disclosure agreements".
    users, questions, phrasings : tuple of Category, optional
        The weighted category lists.
    few_shot : tuple of str, optional
        Real questions from the team's users. Shown as examples of style, never
        as content to copy.
    passages_per_document : int, optional
        Passages drawn per indexed document.
    statements_per_passage : int, optional
        Statements asked for per passage.
    passage_chars : int, optional
        Rough passage width.
    null_share : float, optional
        Target share of null questions in the accepted set.
    near_miss_share : float, optional
        Of the null questions, the share built by changing one detail of an
        answerable question. The rest come from held-out documents.
    held_out_share : float, optional
        Fraction of documents kept out of the index for null generation.
    held_out_minimum : int, optional
        Hold out at least this many documents.
    held_out_passages : int, optional
        Passages drawn per held-out document.
    scope : str, optional
        ``"document"`` scopes every question to one document, as ContractNLI and
        CUAD retrieval is scoped. ``"corpus"`` searches everything.
    uniqueness_k : int, optional
        Non-gold passages the uniqueness filter shows the checker.
    null_check_k : int, optional
        Passages per retriever the null check shows the checker.
    dedup_cosine : float, optional
        Two questions above this BGE-small cosine are near-duplicates. Survey
        it per corpus, as the pair band is surveyed.
    boilerplate_cosine : float, optional
        A passage whose nearest passage in another document is above this
        cosine is drawn at most once across the corpus.
    paraphrase : bool, optional
        Write a paraphrase variant of each accepted answerable question.
    probe_chunk_chars, probe_overlap_chars : int, optional
        The chunking the generator's own probe index uses for the uniqueness
        filter, the null check, and the recorded ranks. Scoring may use any.
    seed : int, optional
        Sampling seed.
    workers : int, optional
        Candidates processed at once. The rate limiter, not this, bounds calls.
    """

    domain: str = "documents"
    users: tuple[Category, ...] = DEFAULT_USERS
    questions: tuple[Category, ...] = DEFAULT_QUESTIONS
    phrasings: tuple[Category, ...] = DEFAULT_PHRASINGS
    few_shot: tuple[str, ...] = ()
    passages_per_document: int = 4
    statements_per_passage: int = 2
    passage_chars: int = 1200
    null_share: float = 0.2
    near_miss_share: float = 0.5
    held_out_share: float = 0.1
    held_out_minimum: int = 3
    held_out_passages: int = 6
    scope: str = "corpus"
    uniqueness_k: int = 5
    null_check_k: int = 10
    dedup_cosine: float = 0.92
    boilerplate_cosine: float = 0.95
    paraphrase: bool = True
    probe_chunk_chars: int = 1200
    probe_overlap_chars: int = 200
    seed: int = 0
    workers: int = 12
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject settings that cannot produce a set."""
        if self.scope not in ("document", "corpus"):
            raise ValueError(
                f"scope must be 'document' or 'corpus', got {self.scope!r}"
            )
        if not 0.0 <= self.null_share < 1.0:
            raise ValueError("null_share must be in [0, 1)")
        if not 0.0 <= self.near_miss_share <= 1.0:
            raise ValueError("near_miss_share must be in [0, 1]")
        for name in ("users", "questions", "phrasings"):
            if not getattr(self, name):
                raise ValueError(f"{name} must name at least one category")

    def sample(self, key: str) -> dict[str, Category]:
        """Draw a user, a question category, and a phrasing for one candidate.

        The draw is seeded by the config seed and ``key``, so a candidate gets
        the same categories however many run at once.
        """
        rng = random.Random(f"{self.seed}:{key}")
        return {
            "user": _weighted(rng, self.users),
            "question": _weighted(rng, self.questions),
            "phrasing": _weighted(rng, self.phrasings),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable form, as recorded on results."""
        record = asdict(self)
        record["extra"] = dict(self.extra)
        return record


def _weighted(rng: random.Random, options: Sequence[Category]) -> Category:
    return rng.choices(list(options), weights=[c.weight for c in options], k=1)[0]


def _categories(raw: Any, default: tuple[Category, ...]) -> tuple[Category, ...]:
    if raw is None:
        return default
    return tuple(
        Category(
            name=str(item["name"]),
            description=str(item.get("description", item["name"])),
            weight=float(item.get("weight", 1.0)),
        )
        for item in raw
    )


def config_from_dict(raw: Mapping[str, Any]) -> GenerationConfig:
    """Build a config from a parsed YAML or JSON mapping.

    Parameters
    ----------
    raw : mapping
        Keys are :class:`GenerationConfig` fields. ``users``, ``questions``, and
        ``phrasings`` are lists of ``{name, description, weight}``.

    Returns
    -------
    GenerationConfig
        The config.

    Raises
    ------
    ValueError
        On a key that is not a config field.
    """
    known = {f.name for f in fields(GenerationConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown config key(s) {sorted(unknown)}")
    values: dict[str, Any] = {
        k: v
        for k, v in raw.items()
        if k not in ("users", "questions", "phrasings", "few_shot")
    }
    values["users"] = _categories(raw.get("users"), DEFAULT_USERS)
    values["questions"] = _categories(raw.get("questions"), DEFAULT_QUESTIONS)
    values["phrasings"] = _categories(raw.get("phrasings"), DEFAULT_PHRASINGS)
    values["few_shot"] = tuple(str(q) for q in raw.get("few_shot") or ())
    return GenerationConfig(**values)


def load_config(path: Path) -> GenerationConfig:
    """Read a team config from YAML or JSON.

    Parameters
    ----------
    path : Path
        A ``.yaml``, ``.yml``, or ``.json`` file.

    Returns
    -------
    GenerationConfig
        The config.

    Raises
    ------
    ImportError
        For a YAML file when PyYAML is not installed.
    """
    text = Path(path).read_text(encoding="utf-8")
    if Path(path).suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as error:  # pragma: no cover - depends on the install
            raise ImportError(
                "Reading a YAML config needs PyYAML; install it or write the "
                "config as JSON"
            ) from error
        raw = yaml.safe_load(text) or {}
    else:
        raw = json.loads(text)
    return config_from_dict(raw)


#: The example a team copies.
EXAMPLE_CONFIG = Path(__file__).with_name("team-config.example.yaml")


__all__ = [
    "DEFAULT_PHRASINGS",
    "DEFAULT_QUESTIONS",
    "DEFAULT_USERS",
    "EXAMPLE_CONFIG",
    "Category",
    "GenerationConfig",
    "config_from_dict",
    "load_config",
]
