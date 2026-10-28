"""Assemble the comparison tables and the committed results from cached steps.

Everything here reads finished steps and calls no model, except
:func:`audit_descriptions`, which is a model step of its own. The committed
files under ``outputs/`` hold question identifiers, page identifiers, counts,
and metrics only, because the question text and the descriptions derive from
FinanceBench (CC BY-NC 4.0) and stay in the local cache.
"""

from __future__ import annotations

import json
import math
import random
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from aieng.advanced_rag import spend
from aieng.advanced_rag.evaluation.multihop import mcnemar_exact
from aieng.advanced_rag.models import ModelChoice, choose_models
from aieng.advanced_rag.runs import Step, StepResult

from . import arms, enrich
from .pages import crop_element, data_uri, page_id
from .pipeline import REFERENCE_CHECKER, REFERENCE_JUDGE, REFERENCE_VLM, RI_ID, Pipeline, digest_of
from .scoring import Scenario, break_even_questions, page_metrics, scenario_cost
from .vlm import parse_json


#: Default models of the three paid families, for the cost tables.
PAID_MODELS: tuple[str, ...] = ("claude-haiku-5-5", "gemini-3.5-flash-lite", "gpt-6-luna")
#: Each arm's oracle: the same reader input, with the gold page in place of retrieval.
ORACLE_OF: Mapping[str, str] = {
    "T": "O-text",
    "E1": "O-enriched",
    "E2": "O-enriched",
    "L": "O-enriched",
    "VT": "O-enriched",
    "VV": "O-image",
    "TV": "O-image",
    "H": "O-both",
}
#: The paired comparisons the notebook reports, with the group each is read on.
PAIRS: tuple[tuple[str, str, str], ...] = (
    ("VV", "T", "figure-only"),
    ("TV", "T", "figure-only"),
    ("E2", "T", "figure-only"),
    ("E2", "E1", "figure-only"),
    ("VT", "L", "figure-only"),
    ("O-image", "O-text", "figure-only"),
    ("T", "VV", "table"),
    ("O-text", "O-image", "table"),
    ("E2", "VV", "control"),
)

AUDIT_PROMPT = """The image is a chart or picture from a company filing. Below is a description
another model wrote of it.

Description:
{description}

Check the description against the image. Return one JSON object and nothing else:
  "wrong_value": true if any number in the description differs from the image;
  "invented": true if the description names a label, series, or object the
    image does not show;
  "missing_values": true if the image prints data values that the description
    leaves out."""


def group_of(question: Mapping[str, Any]) -> str:
    """Return the reporting group: figure-only, duplicated, table, control, or null."""
    if question["group"] == "figure":
        return str(question.get("subgroup") or "figure")
    return str(question["group"])


GROUPS: tuple[str, ...] = ("figure-only", "duplicated", "table", "control", "null")


def accuracy_table(pipe: Pipeline, which: str = "all", rows: Sequence[str] = arms.ROWS) -> dict[str, dict[str, Any]]:
    """Return, per row, answer accuracy by group and the reader's cost per question."""
    found = {str(q["query_id"]): q for q in pipe.questions(which)}
    table: dict[str, dict[str, Any]] = {}
    for row in rows:
        verdicts = pipe.verdicts(row, which).value
        answers = pipe.answers(row, which).value
        by_group: dict[str, list[bool]] = {g: [] for g in GROUPS}
        abstained_answerable = 0
        answerable = 0
        for qid, verdict in verdicts.items():
            by_group[group_of(found[qid])].append(bool(verdict["correct"]))
            if found[qid]["group"] != "null":
                answerable += 1
                abstained_answerable += verdict["method"] == "abstained"
        prompt = [int(a["prompt_tokens"]) for a in answers.values()]
        completion = [int(a["completion_tokens"]) for a in answers.values()]
        table[row] = {
            "accuracy": {g: _rate(v) for g, v in by_group.items()},
            "n": {g: len(v) for g, v in by_group.items()},
            "false_abstention": round(abstained_answerable / answerable, 4) if answerable else math.nan,
            "prompt_tokens": round(statistics.mean(prompt), 1) if prompt else 0,
            "completion_tokens": round(statistics.mean(completion), 1) if completion else 0,
            "images": round(statistics.mean(int(a["images"]) for a in answers.values()), 2) if answers else 0,
            "model": pipe.answers(row, which).provenance.model if pipe.answers(row, which).provenance else "",
            "origin": pipe.answers(row, which).origin,
        }
    return table


def retrieval_table(pipe: Pipeline, which: str = "all", k: int = 5) -> dict[str, dict[str, dict[str, float]]]:
    """Return page recall, nDCG, and failure at ``k`` per arm and group."""
    found = pipe.questions(which)
    rankings = pipe.rankings()
    out: dict[str, dict[str, dict[str, float]]] = {}
    for arm in arms.ARMS:
        out[arm] = {}
        for group in GROUPS[:-1]:
            gold = {str(q["query_id"]): q["gold_pages"] for q in found if group_of(q) == group}
            out[arm][group] = page_metrics(rankings[arm], gold, k)
        gold_all = {str(q["query_id"]): q["gold_pages"] for q in found if q["group"] != "null"}
        out[arm]["all"] = page_metrics(rankings[arm], gold_all, k)
    return out


def decomposition(accuracy: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return each arm's retrieval loss: its oracle's accuracy minus its own."""
    out: dict[str, dict[str, Any]] = {}
    for arm, oracle in ORACLE_OF.items():
        if arm not in accuracy or oracle not in accuracy:
            continue
        loss = {
            g: round(accuracy[oracle]["accuracy"][g] - accuracy[arm]["accuracy"][g], 4)
            for g in GROUPS
            if not math.isnan(accuracy[oracle]["accuracy"][g]) and not math.isnan(accuracy[arm]["accuracy"][g])
        }
        out[arm] = {"oracle": oracle, "loss": loss}
    return out


def paired(pipe: Pipeline, which: str = "all") -> list[dict[str, Any]]:
    """Return the exact McNemar test for each pair in :data:`PAIRS`."""
    found = {str(q["query_id"]): q for q in pipe.questions(which)}
    rows: list[dict[str, Any]] = []
    for left, right, group in PAIRS:
        a = pipe.verdicts(left, which).value
        b = pipe.verdicts(right, which).value
        ids = sorted(q for q in a if group_of(found[q]) == group and q in b)
        test = mcnemar_exact([bool(a[q]["correct"]) for q in ids], [bool(b[q]["correct"]) for q in ids])
        rows.append(
            {
                "left": left,
                "right": right,
                "group": group,
                "n": len(ids),
                "left_only": test.only_a,
                "right_only": test.only_b,
                "p": round(test.p_value, 6),
            }
        )
    return rows


def enrichment_costs(pipe: Pipeline) -> dict[str, dict[str, float]]:
    """Return calls, tokens, and seconds per element for each enrichment pass."""
    out: dict[str, dict[str, float]] = {}
    for name in ("e1-pictures", "e2-pictures", "e2-tables"):
        records = list(pipe.enrichment(name).value.values())
        n = len(records)
        prompt = sum(int(r["prompt_tokens"]) for r in records)
        completion = sum(int(r["completion_tokens"]) for r in records)
        seconds = [float(r["seconds"]) for r in records if float(r["seconds"]) > 0.05]
        out[name] = {
            "elements": n,
            "prompt_tokens_per_element": round(prompt / n, 1) if n else 0,
            "completion_tokens_per_element": round(completion / n, 1) if n else 0,
            "seconds_per_element": round(statistics.median(seconds), 2) if seconds else math.nan,
            "empty": sum(1 for r in records if not r.get("text")),
        }
    return out


def paid_estimates(tokens: Mapping[str, Mapping[str, float]], per: int = 1000) -> dict[str, dict[str, float | None]]:
    """Return list-price dollars per ``per`` items on each paid default model.

    The token counts are the on-prem reader's. Image tokens differ by provider,
    so these scale the measured counts and are estimates.
    """
    out: dict[str, dict[str, float | None]] = {}
    for name, row in tokens.items():
        out[name] = {}
        for model in PAID_MODELS:
            cost = spend.estimate_tokens(
                model,
                int(row["prompt_tokens"] * per),
                int(row["completion_tokens"] * per),
                table=spend.OFFLINE_PRICES,
            )
            out[name][model] = round(cost, 3) if cost is not None else None
    return out


def selection_counts(pipe: Pipeline) -> dict[str, int]:
    """Return how many elements the universe holds and how many enrichment selected."""
    found = pipe.elements()
    pages = pipe.universe().pages
    return {
        "pages": len(pages),
        "pictures": sum(1 for e in found if e.kind == "picture"),
        "pictures_described": len(enrich.describable_pictures(found, pages)),
        "tables": sum(1 for e in found if e.kind == "table"),
        "tables_summarised": len(enrich.summarisable_tables(found, pages)),
    }


def scenario_table(
    pipe: Pipeline,
    scenario: Scenario,
    model: str,
    accuracy: Mapping[str, Mapping[str, Any]],
    machine_seconds_per_page: Mapping[str, float],
) -> dict[str, Any]:
    """Return cost per arm under one scenario and model, and E2's break-even with VV."""
    costs = enrichment_costs(pipe)
    weights = [costs["e2-pictures"], costs["e2-tables"]]
    elements = sum(c["elements"] for c in weights)
    per_element = (
        sum(
            spend.estimate_tokens(
                model,
                int(c["prompt_tokens_per_element"] * c["elements"]),
                int(c["completion_tokens_per_element"] * c["elements"]),
                table=spend.OFFLINE_PRICES,
            )
            or 0.0
            for c in weights
        )
        / elements
        if elements
        else 0.0
    )
    reader = {
        arm: (
            spend.estimate_tokens(
                model,
                int(accuracy[arm]["prompt_tokens"]),
                int(accuracy[arm]["completion_tokens"]),
                table=spend.OFFLINE_PRICES,
            )
            or 0.0
        )
        for arm in arms.ARMS
        if arm in accuracy
    }
    table = scenario_cost(
        scenario,
        enrichment_usd_per_element=per_element,
        reader_usd_per_question=reader,
        machine_seconds_per_page=machine_seconds_per_page,
    )
    e2, vv = table.get("E2"), table.get("VV")
    even = (
        break_even_questions(e2["index_usd"], reader["E2"], vv["index_usd"], reader["VV"])
        if e2 is not None and vv is not None
        else math.nan
    )
    return {
        "model": model,
        "usd_per_element": round(per_element, 6),
        "usd_per_question": {k: round(v, 6) for k, v in reader.items()},
        "arms": table,
        "break_even_questions_e2_vs_vv": even,
    }


def audit_step(pipe: Pipeline) -> Step:
    """Return the step that checks 30 descriptions per prompt against their crops."""
    return Step(
        RI_ID,
        "description-audit",
        "crop-check",
        {"prompt": digest_of(AUDIT_PROMPT), "sample": 30, "seed": 0},
        role="judging",
        reference_model=REFERENCE_CHECKER,
        upstream={k: v for k, v in pipe.enrichment_digests().items() if k != "e2-tables"},
    )


def audit_descriptions(pipe: Pipeline) -> StepResult:
    """Ask a second model whether sampled descriptions misstate their pictures."""
    e1 = pipe.enrichment("e1-pictures").value
    e2 = pipe.enrichment("e2-pictures").value
    shared = sorted(set(e1) & set(e2))
    sample = random.Random(0).sample(shared, min(30, len(shared)))
    by_id = {e.element_id: e for e in pipe.elements()}

    def compute(choice: ModelChoice) -> Any:
        def one(item: tuple[str, str]) -> tuple[str, dict[str, Any]]:
            name, eid = item
            text = str((e1 if name == "E1" else e2)[eid]["text"])
            reply = pipe.llm.complete(
                choice.model,
                AUDIT_PROMPT.format(description=text[:4000]),
                [data_uri(crop_element(pipe.corpus, by_id[eid]))],
                max_tokens=200,
            )
            try:
                found = parse_json(reply.text)
            except ValueError:
                found = {}
            return f"{name}|{eid}", {
                k: bool(found.get(k, False)) for k in ("wrong_value", "invented", "missing_values")
            }

        return dict(enrich.parallel_map(one, [(n, e) for n in ("E1", "E2") for e in sample]))

    models = choose_models("onprem", judging=REFERENCE_CHECKER, generation=REFERENCE_VLM, extraction=REFERENCE_VLM)
    return pipe._step(audit_step(pipe), compute, models)


def audit_summary(value: Mapping[str, Mapping[str, bool]]) -> dict[str, dict[str, float]]:
    """Return the share of sampled descriptions flagged, per prompt."""
    out: dict[str, dict[str, float]] = {}
    for name in ("E1", "E2"):
        rows = [v for k, v in value.items() if k.startswith(name + "|")]
        out[name] = {
            "n": len(rows),
            **{flag: _rate([bool(r[flag]) for r in rows]) for flag in ("wrong_value", "invented", "missing_values")},
        }
    return out


def question_summary(pipe: Pipeline) -> dict[str, Any]:
    """Return the question set's counts, funnel, and identifiers, without text."""
    value = pipe.question_set().value
    found = value["questions"]
    counts: dict[str, int] = {}
    for q in found:
        counts[group_of(q)] = counts.get(group_of(q), 0) + 1
    tables = [q for q in found if q["group"] == "table"]
    return {
        "counts": counts,
        "table_split": sum(1 for q in tables if q.get("subgroup") == "split"),
        "funnel": value["funnel"],
        "subset": value["subset"],
        "questions": [
            {
                "query_id": q["query_id"],
                "group": q["group"],
                "subgroup": q.get("subgroup"),
                "gold_pages": q["gold_pages"],
                "rewritten_from": q.get("rewritten_from"),
                "element_ids": q.get("element_ids", []),
                "source": q.get("candidate_id") if q["group"] == "control" else None,
            }
            for q in found
        ],
    }


def _rate(values: Sequence[bool]) -> float:
    return round(sum(values) / len(values), 4) if values else math.nan


def _produced_by(pipe: Pipeline) -> dict[str, Any]:
    return {
        "model": REFERENCE_VLM,
        "reader": REFERENCE_VLM,
        "enrichment": REFERENCE_VLM,
        "question_generator": REFERENCE_VLM,
        "question_checker": REFERENCE_CHECKER,
        "judge": REFERENCE_JUDGE,
        "endpoint": "vector-cluster",
        "embeddings": "BAAI/bge-small-en-v1.5 (local)",
        "page_embeddings": "Qdrant/colmodernvbert (local)",
        "producer": "implementations/special_topics/visual_documents/build_reference.py",
        "mode": "explore: one run, no seeds beyond the fixed sampling seed 0",
    }


def write_outputs(pipe: Pipeline, directory: Path, cpu_probe: Mapping[str, float] | None = None) -> list[Path]:
    """Write the committed results: identifiers, counts, and metrics, no corpus text."""
    directory.mkdir(parents=True, exist_ok=True)
    accuracy = accuracy_table(pipe, "all")
    retrieval = retrieval_table(pipe, "all")
    index = pipe.page_index()
    _, query_meta = pipe.query_embeddings()
    text = pipe.text_rankings()
    pipe.rankings()
    probe = dict(cpu_probe or {})
    enrichment = enrichment_costs(pipe)
    reader_tokens = {
        row: {"prompt_tokens": a["prompt_tokens"], "completion_tokens": a["completion_tokens"], "images": a["images"]}
        for row, a in accuracy.items()
    }
    costs: dict[str, Any] = {
        "selection": selection_counts(pipe),
        "enrichment": enrichment,
        "page_index": {
            "pages": len(index.page_ids),
            "megabytes_float16": index.megabytes,
            "vectors_per_page": round(index.vectors.shape[0] / len(index.page_ids), 1),
            "seconds_per_page_gpu": index.meta.get("seconds_per_page"),
            "hardware": index.meta.get("hardware"),
            "query_seconds_gpu": query_meta.get("seconds_per_query"),
            "maxsim_seconds_per_query_cpu": pipe._memo.get("maxsim_seconds"),
            **{f"cpu_{k}": v for k, v in probe.items()},
        },
        "text_index": {
            "chunks": text["chunks"],
            "seconds": text["seconds"],
            "megabytes_float32": {arm: round(n * 384 * 4 / 1e6, 2) for arm, n in text["chunks"].items()},
        },
        "reader_tokens_per_question": reader_tokens,
    }
    costs["paid_reader_usd_per_1000_questions"] = paid_estimates(reader_tokens)
    costs["paid_enrichment_usd_per_1000_elements"] = paid_estimates(
        {
            name: {
                "prompt_tokens": c["prompt_tokens_per_element"],
                "completion_tokens": c["completion_tokens_per_element"],
            }
            for name, c in enrichment.items()
        },
    )
    results = {
        "produced_by": _produced_by(pipe),
        "k": 5,
        "retrieval": retrieval,
        "answers": accuracy,
        "retrieval_loss": decomposition(accuracy),
        "paired": paired(pipe, "all"),
    }
    written = []
    for name, payload in (
        ("question_set.json", {"produced_by": _produced_by(pipe), **question_summary(pipe)}),
        ("results.json", results),
        ("costs.json", {"produced_by": _produced_by(pipe), **costs}),
        (
            "description_audit.json",
            {"produced_by": _produced_by(pipe), **audit_summary(audit_descriptions(pipe).value)},
        ),
    ):
        path = directory / name
        path.write_text(json.dumps(clean(payload), indent=1, allow_nan=False) + "\n", encoding="utf-8")
        written.append(path)
    return written


def clean(value: Any) -> Any:
    """Return a JSON-ready copy with NaN and infinity written as None."""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, Mapping):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


__all__ = [
    "AUDIT_PROMPT",
    "GROUPS",
    "ORACLE_OF",
    "PAID_MODELS",
    "PAIRS",
    "accuracy_table",
    "audit_descriptions",
    "audit_summary",
    "clean",
    "decomposition",
    "enrichment_costs",
    "group_of",
    "paid_estimates",
    "page_id",
    "paired",
    "question_summary",
    "retrieval_table",
    "scenario_table",
    "selection_counts",
    "write_outputs",
]
