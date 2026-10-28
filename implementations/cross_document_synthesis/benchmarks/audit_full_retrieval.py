"""Re-encode every full-run decomposition query and replay its local ranking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from aieng.advanced_rag.corpora.musique import fetch_musique, load_musique
from aieng.advanced_rag.retrieval.index import DenseIndex
from synthesis_lab.dense_baseline import rank_questions
from synthesis_lab.runtime import DEFAULT_CACHE, embeddings, local_embedder, write_json

from benchmarks.judge_full_answers import load_arm


def audit(cache_dir: Path) -> dict[str, int | bool]:
    """Use the pinned encoder, original batch shapes, and no proxy completions."""
    passages, questions = load_musique(fetch_musique(dest=cache_dir / "corpus"))
    embedder = local_embedder(cache_dir)
    docs, queries, _ = embeddings(cache_dir, passages, questions, embedder=embedder)
    dense = json.loads(
        (Path(__file__).resolve().parents[1] / "outputs/full_evaluation/dense_replay/rankings.json").read_text()
    )
    ranks = rank_questions(passages, questions, docs, queries, k=20)
    for q in questions:
        if ranks[q.question_id] != dense[q.question_id]:
            raise ValueError(f"dense ranking changed: {q.question_id}")
    multi = load_arm("decomposition")
    index = DenseIndex([p.passage_id for p in passages], docs)
    query_count = 0
    for number, q in enumerate(questions, start=1):
        row = multi[q.question_id]
        vectors = np.asarray(embedder.encode_queries(row["retrieval_queries"]), dtype=np.float32)
        replay = index.search(vectors, k=20)
        if replay != row["step_rankings"]:
            raise ValueError(f"decomposition ranking changed: {q.question_id}")
        query_count += len(replay)
        if number % 100 == 0:
            print(f"Replayed local retrieval: {number}/{len(questions)}", flush=True)
    report = {
        "dense_questions": len(questions),
        "decomposition_questions": len(questions),
        "decomposition_search_queries": query_count,
        "ranking_mismatches": 0,
        "new_proxy_calls": 0,
        "query_vectors_reencoded": True,
    }
    write_json(Path(__file__).resolve().parents[1] / "outputs/full_evaluation/retrieval_audit.json", report)
    return report


def main() -> None:
    """Run this local-embedding audit only inside a compute allocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    print(audit(parser.parse_args().cache_dir))


if __name__ == "__main__":
    main()
