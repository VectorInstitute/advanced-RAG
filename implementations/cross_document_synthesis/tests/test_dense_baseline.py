"""Dense baseline: corpus-wide ranking, multi-hop scoring, and gold-free answering."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub
import numpy as np
import pytest
from aieng.advanced_rag.corpora.musique import MusiqueQuestion, Passage, SubQuestion
from aieng.advanced_rag.metering import CallUsage, UsageMeter, report_call
from aieng.advanced_rag.retrieval.interface import Hints, ScopeUnsupported
from synthesis_lab import runtime as runner
from synthesis_lab.dense_baseline import DensePassageRetriever, answer_baseline, rank_questions, score_by_hops


PASSAGES = [Passage(f"p{i}", f"Title {i}", f"Corpus text {i}") for i in range(5)]


def question(hops):
    return MusiqueQuestion(
        f"{hops}hop__1_2",
        f"asked {hops}",
        "gold-only answer",
        ("alias",),
        hops,
        tuple(f"p{i}" for i in range(hops)),
        tuple(SubQuestion("SECRET GOLD DECOMPOSITION", "SECRET SUBANSWER", f"p{i}") for i in range(hops)),
    )


def test_rankings_use_whole_corpus_and_break_ties_by_passage_id():
    # Candidate/support annotations must not restrict the corpus searched.
    passages = list(reversed(PASSAGES))
    vectors = np.asarray([[1, 0]] * len(passages), dtype=np.float32)
    runs = rank_questions(passages, [question(2)], vectors, np.asarray([[1, 0]]), 5)
    assert list(runs.values()) == [[p.passage_id for p in PASSAGES]]


@pytest.mark.parametrize("depth", [0, 6])
def test_invalid_ranking_depth_is_rejected(depth):
    with pytest.raises(ValueError, match="corpus size"):
        rank_questions(PASSAGES, [question(2)], np.ones((5, 2)), np.ones((1, 2)), depth)


def test_mismatched_and_nonfinite_vectors_are_rejected():
    with pytest.raises(ValueError, match="shapes"):
        rank_questions(PASSAGES, [question(2)], np.ones((4, 2)), np.ones((1, 2)), 2)
    with pytest.raises(ValueError, match="finite"):
        rank_questions(PASSAGES, [question(2)], np.ones((5, 2)), np.asarray([[np.nan, 0]]), 2)


def test_hop_metrics_distinguish_partial_recall_from_all_supports():
    questions = [question(h) for h in (2, 3, 4)]
    runs = {q.question_id: ["p0", "p4", "p1", "p2", "p3"] for q in questions}
    report = score_by_hops(questions, runs, [2, 5], {q.question_id: "alias" for q in questions})
    assert report["2"]["retrieval"]["2"]["recall_at_2"] == 0.5
    assert report["2"]["retrieval"]["2"]["all_supports_at_2"] == 0
    assert report["3"]["retrieval"]["2"]["recall_at_2"] == 0.3333
    assert report["4"]["retrieval"]["2"]["recall_at_2"] == 0.25
    assert report["overall"]["retrieval"]["5"]["all_supports_at_5"] == 1
    assert report["overall"]["answers"] == {"exact_match": 1, "token_f1": 1}


def test_missing_questions_are_not_silently_dropped_from_scores():
    questions = [question(2), question(3)]
    with pytest.raises(ValueError, match="exactly"):
        score_by_hops(questions, {questions[0].question_id: ["p0", "p1"]}, [2])
    with pytest.raises(ValueError, match="scoring depth"):
        score_by_hops(questions[:1], {questions[0].question_id: ["p0"]}, [2])


def test_retriever_preserves_provenance_and_rejects_scope():
    retriever = DensePassageRetriever(PASSAGES, {"asked": ["p3", "p0"]})
    assert [e.chunk_id for e in retriever("asked", k=2)] == ["p3", "p0"]
    assert retriever.retrieve("asked", 1)[0].text == "Title 3\nCorpus text 3"
    with pytest.raises(ScopeUnsupported):
        retriever.retrieve("asked", scope="p0")
    with pytest.raises(ValueError, match="depth"):
        retriever.retrieve("asked", 3)


@pytest.mark.parametrize("ranking", [["p0", "p0"], ["foreign"]])
def test_invalid_passage_ids_are_rejected(ranking):
    with pytest.raises(ValueError, match="duplicate or unknown"):
        DensePassageRetriever(PASSAGES, {"asked": ranking})


def test_harness_answers_original_question_once_without_gold_leakage():
    q = question(2)
    prompts = []
    meter = UsageMeter()

    def complete(prompt):
        prompts.append(prompt)
        report_call(CallUsage(100, 10, 110, reported=True), 0.1)
        return json.dumps({"answer": "alias", "sources": [1], "not_stated": False})

    retriever = DensePassageRetriever(PASSAGES, {q.question: ["p4", "p0", "p1", "p2", "p3"]})
    result = answer_baseline([q], retriever, complete, meter, evidence_budget=2)
    assert len(prompts) == 1
    assert "asked 2" in prompts[0] and "Corpus text 4" in prompts[0]
    assert "Corpus text 1" not in prompts[0]
    assert "gold-only answer" not in prompts[0] and "SECRET" not in prompts[0]
    assert result.results[0].answer.text == "alias"
    assert len(result.results[0].attempts) == 1
    assert result.usage.calls == 1
    assert meter.per_question()[q.question_id].total_tokens == 110


def test_embedding_cache_checks_integrity_and_configuration(monkeypatch, tmp_path):
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *args, **kwargs: "local-model")
    fake = SimpleNamespace(
        describe=lambda: {"max_seq_length": 512},
        encode=lambda texts: [[1.0, 0.0] for _ in texts],
        encode_queries=lambda texts: [[0.0, 1.0] for _ in texts],
    )
    monkeypatch.setattr(runner, "SentenceTransformerEmbedder", lambda *args, **kwargs: fake)
    args = argparse.Namespace(cache_dir=tmp_path)
    first = runner.embeddings(args.cache_dir, PASSAGES, [question(2)])
    cached = runner.embeddings(args.cache_dir, PASSAGES, [question(2)])
    assert not first[2]["cache_hit"] and cached[2]["cache_hit"]
    np.testing.assert_array_equal(first[0], cached[0])
    manifest = tmp_path / "embeddings.json"
    original = manifest.read_text()
    changed = json.loads(original)
    changed["identity"]["passage_format"] = "different"
    manifest.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="configuration differs"):
        runner.embeddings(args.cache_dir, PASSAGES, [question(2)])
    manifest.write_text(original)
    (tmp_path / "embeddings.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        runner.embeddings(args.cache_dir, PASSAGES, [question(2)])


@pytest.mark.parametrize(
    "path",
    sorted(
        [
            *Path(__file__).resolve().parents[1].glob("*.py"),
            *Path(__file__).resolve().parents[1].glob("synthesis_lab/*.py"),
        ]
    ),
)
def test_embeddings_stay_local_and_generation_uses_vector_proxy(path):
    source = path.read_text()
    assert "embeddings.create" not in source
    assert "text-embedding-" not in source
    assert "api.openai.com" not in source


def test_dense_runner_writes_answers_with_the_supplied_corpus(monkeypatch, tmp_path):
    import run_dense_baseline as baseline_runner  # noqa: PLC0415

    class Completion:
        def __init__(self, *args):
            self.requests = []

        def __call__(self, prompt):
            assert "Corpus text 0" in prompt
            self.requests.append({"cache_key": "test-request", "cache_hit": False})
            report_call(CallUsage(100, 10, 110, reported=True), 0.01)
            return '{"answer":"alias","sources":[1],"not_stated":false}'

    q = question(2)
    monkeypatch.setattr(baseline_runner, "CachedProxyCompletion", Completion)
    baseline_runner._answer_questions(
        PASSAGES, [q], {q.question_id: [p.passage_id for p in PASSAGES]}, tmp_path, tmp_path, answer_model="test-model"
    )
    predictions = json.loads((tmp_path / "predictions.json").read_text())
    assert predictions[0]["answer"] == "alias"
    assert predictions[0]["citation_ids"] == ["p0"]
    assert json.loads((tmp_path / "answers.json").read_text())["metrics"]["overall"]["answers"]["exact_match"] == 1


#: A reply that repeats the prompt back.
ECHO = object()


class FakeLiteLLM:
    """Stands in for ``litellm.acompletion``: answers each prompt with ``reply``.

    ``reply`` of :data:`ECHO` repeats the prompt, and None is a reply with no
    content. A numeric prompt ``n`` costs ``n`` prompt tokens, so concurrent
    calls can be told apart by their usage.
    """

    def __init__(self, reply, barrier=None):
        self.reply = reply
        self.barrier = barrier
        self.requests = []

    async def __call__(self, **kwargs):
        import litellm  # noqa: PLC0415

        self.requests.append(kwargs)
        prompt = kwargs["messages"][-1]["content"]
        if self.barrier is not None:
            await asyncio.get_running_loop().run_in_executor(None, self.barrier.wait, 5)
        tokens = int(prompt) if prompt.isdigit() else 100
        return litellm.ModelResponse(
            id="r1",
            model=kwargs["model"].split("/", 1)[1],
            choices=[
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": prompt if self.reply is ECHO else self.reply},
                }
            ],
            usage={"prompt_tokens": tokens, "completion_tokens": 10, "total_tokens": tokens + 10},
        )


def cached_completion(monkeypatch, tmp_path, reply, barrier=None):
    import litellm  # noqa: PLC0415
    from aieng.advanced_rag import metering  # noqa: PLC0415

    fake = FakeLiteLLM(reply, barrier)
    monkeypatch.setattr(litellm, "acompletion", fake)
    monkeypatch.setattr(metering, "before_call", lambda model: None)
    monkeypatch.setenv("VECTOR_PROXY_API_KEY", "test-key")
    return runner.CachedProxyCompletion(tmp_path, runner.ANSWER_MODEL), fake


def test_proxy_cache_hits_are_separate_from_live_calls(monkeypatch, tmp_path):
    completion, fake = cached_completion(monkeypatch, tmp_path, "answer")
    meter = UsageMeter()
    with meter.question("cold"):
        assert completion("prompt") == "answer"
    with meter.question("warm"):
        assert completion("prompt") == "answer"
    assert len(fake.requests) == 1
    assert meter.per_question()["cold"].calls == 1
    assert meter.per_question()["cold"].total_tokens == 110
    assert meter.per_question()["warm"].calls == 0
    assert meter.per_question()["warm"].cache_hits == 1
    assert completion.requests[0]["cache_key"] == completion.requests[1]["cache_key"]
    assert [row["cache_hit"] for row in completion.requests] == [False, True]


def test_the_cache_key_is_the_one_the_committed_results_record(monkeypatch, tmp_path):
    from aieng.advanced_rag.graphrag.proxy import ResponseCache  # noqa: PLC0415
    from aieng.advanced_rag.models import request_params  # noqa: PLC0415

    completion, _ = cached_completion(monkeypatch, tmp_path, "answer")
    request = {
        "prompt": "prompt",
        "model": runner.ANSWER_MODEL,
        "base_url": completion.base_url,
        "request_params": request_params(runner.ANSWER_MODEL, max_tokens=256, temperature=0, thinking=False),
        "temperature": 0,
        "max_tokens": 256,
        "thinking": False,
    }
    assert completion.key("prompt") == ResponseCache.key(request)


def test_reasoning_configuration_is_part_of_the_response_cache_identity(monkeypatch, tmp_path):
    completion, fake = cached_completion(monkeypatch, tmp_path, "answer")
    completion("prompt")
    completion.reasoning_effort = "low"
    completion("prompt")
    completion("prompt")
    assert completion.requests[0]["cache_key"] != completion.requests[1]["cache_key"]
    assert [row["cache_hit"] for row in completion.requests] == [False, False, True]
    assert fake.requests[1]["reasoning_effort"] == "low"


def test_empty_proxy_responses_fail_without_becoming_scored_answers(monkeypatch, tmp_path):
    completion, fake = cached_completion(monkeypatch, tmp_path, None)
    with pytest.raises(RuntimeError, match="no answer content"):
        completion("prompt")
    assert len(fake.requests) == 1
    assert len(completion.cache) == 0


def test_concurrent_proxy_workers_keep_response_usage_separate(monkeypatch, tmp_path):
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415
    from threading import Barrier  # noqa: PLC0415

    completion, _ = cached_completion(monkeypatch, tmp_path, ECHO, barrier=Barrier(2))
    with ThreadPoolExecutor(max_workers=2) as workers:
        assert list(workers.map(completion, ["17", "93"])) == ["17", "93"]
    for request in completion.requests:
        entry = completion.cache.entry(request["cache_key"])
        assert entry.usage.prompt_tokens == int(entry.payload)


def test_precomputed_retriever_preserves_sync_and_async_depth_and_scope_contract():
    passages = [Passage(f"p{i}", f"Title {i}", f"Text {i}") for i in range(20)]
    ranked = [p.passage_id for p in passages]
    retriever = DensePassageRetriever(passages, {"query": ranked})
    for depth in (3, 15):
        expected = retriever.retrieve("query", depth)
        assert retriever("query", k=depth) == expected
        assert asyncio.run(retriever.aretrieve("query", depth, hints=Hints(keywords=("advisory",)))) == expected
    with pytest.raises(ScopeUnsupported):
        retriever("query", scope="document")
    with pytest.raises(ScopeUnsupported):
        asyncio.run(retriever.aretrieve("query", scope="document"))
