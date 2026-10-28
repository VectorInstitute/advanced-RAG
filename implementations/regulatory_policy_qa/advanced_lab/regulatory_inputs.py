"""Reconstruct evaluation inputs from the shipped public-domain QA artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from aieng.advanced_rag.retrieval.corpus import chunk_contract

from .common import OUTPUT, digest


def regulatory_inputs(output: Path = OUTPUT) -> list[dict[str, Any]]:
    """Recover exactly the text the generator read, without a model or download.

    The saved question/answer report keeps ordered retrieved IDs. The pinned
    government snapshot supplies source text; only its original 1,200-character
    passage previews reach evaluation. References never affect retrieval.
    """
    source = json.loads((output / "regulatory_source.json").read_text())
    report = json.loads((output / "regulatory.json").read_text())
    actual = hashlib.sha256(source["text"].encode()).hexdigest()
    if actual != source["text_sha256"] or actual != report["source"]["text_sha256"]:
        raise ValueError("regulatory input source digest mismatch")
    chunks = chunk_contract(source["document_id"], source["text"])
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    rows = []
    seen = set()
    for outcome in report["outcomes"]:
        query_id = outcome["query_id"]
        if query_id in seen or type(outcome["is_null"]) is not bool:
            raise ValueError("regulatory input IDs must be unique and null labels boolean")
        seen.add(query_id)
        if not outcome["retrieved_ids"] or any(key not in by_id for key in outcome["retrieved_ids"]):
            raise ValueError("regulatory input contains unknown or missing evidence")
        rows.append(
            {
                "query_id": query_id,
                "user_input": outcome["question"],
                "response": outcome["answer"],
                "reference": outcome["reference"],
                "retrieved_contexts": [by_id[key].text[:1200] for key in outcome["retrieved_ids"]],
                "is_null": outcome["is_null"],
            }
        )
    return rows


def input_fingerprints(output: Path = OUTPUT) -> dict[str, str]:
    """Identify the public source and QA report behind an evaluation."""
    return {name: digest(output / name) for name in ("regulatory_source.json", "regulatory.json")}
