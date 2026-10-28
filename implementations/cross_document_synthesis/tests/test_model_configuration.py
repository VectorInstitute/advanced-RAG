"""Shared model roles, experiment identities, and bounded live spending."""

import json
from types import SimpleNamespace

import pytest
from aieng.advanced_rag.models import REFERENCE_MODELS, choose_models
from aieng.advanced_rag.spend import SpendLimitReached
from run_benchmark import experiment_spec, open_report
from synthesis_lab import runtime
from synthesis_lab.tutorial import synthetic_corpus


@pytest.fixture
def clean_configuration(monkeypatch, tmp_path):
    for name in (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "VECTOR_PROXY_API_KEY",
        "ADVANCED_RAG_MODEL_FAMILY",
        "ADVANCED_RAG_GENERATION_MODEL",
        "ADVANCED_RAG_EXTRACTION_MODEL",
        "ADVANCED_RAG_JUDGE_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()
    return tmp_path


def test_resume_identity_includes_auxiliary_roles(clean_configuration, tmp_path):
    corpus, _, _ = synthetic_corpus()
    first = experiment_spec(
        corpus, "dense", models=choose_models("onprem", judging=choose_models("google").judging.model)
    )
    other = experiment_spec(corpus, "dense", models=REFERENCE_MODELS)
    assert first["model"] == other["model"]
    assert first["model_roles"] != other["model_roles"]
    path = tmp_path / "report.json"
    path.write_text(json.dumps(open_report(path, first)))
    with pytest.raises(ValueError, match="differs"):
        open_report(path, other)


def test_budget_refusal_is_not_retried_or_cached(tmp_path, monkeypatch):
    import litellm  # noqa: PLC0415
    from aieng.advanced_rag import metering  # noqa: PLC0415

    sent = []
    calls = []

    async def provider(**kwargs):
        sent.append(kwargs)
        raise AssertionError("the provider was called")

    def refuse(model):
        calls.append(model)
        raise SpendLimitReached("fixture budget")

    monkeypatch.setattr(litellm, "acompletion", provider)
    monkeypatch.setattr(metering, "before_call", refuse)
    monkeypatch.setenv("VECTOR_PROXY_API_KEY", "test-key")
    complete = runtime.CachedProxyCompletion(tmp_path, "fixture-model")
    with pytest.raises(SpendLimitReached):
        complete("question")
    assert calls == ["fixture-model"]
    assert sent == []
    assert complete.requests == []
    assert complete.completed == 0
    assert len(complete.cache) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["dense", "decomposition", "corrective", "graded-single-pass"])
async def test_role_wiring_and_budget_checkpoint(clean_configuration, tmp_path, monkeypatch, method):
    import run_benchmark as runner  # noqa: PLC0415

    corpus, vectors, encode = synthetic_corpus()
    models = choose_models("google", extraction="gpt-oss-120b")
    captured = {}

    def completion(cache, model):
        return SimpleNamespace(model=model)

    class RefusingAnswerer:
        def __init__(self, *args, **kwargs):
            captured["arguments"] = args
            captured["keywords"] = kwargs

        def answer(self, *args):
            raise SpendLimitReached("fixture budget")

    monkeypatch.setattr(runner.spend, "session", lambda: None)
    monkeypatch.setattr(runner, "corpus_vectors", lambda *args: (vectors, encode, {}))
    monkeypatch.setattr(runner, "CachedProxyCompletion", completion)
    monkeypatch.setattr(runner, "DenseAnswerer", RefusingAnswerer)
    monkeypatch.setattr(runner, "DecompositionAnswerer", RefusingAnswerer)
    output = tmp_path / "interrupted.json"
    with pytest.raises(SpendLimitReached):
        await runner.run(corpus, method, output, tmp_path, models=models)
    report = json.loads(output.read_text())
    assert not report["completed"]
    assert report["records"] == []
    assert report["interrupted"]["reason"] == "spend_limit"
    assert report["spec"]["model_roles"] == models.as_dict()
    if method == "decomposition":
        assert captured["arguments"][3].model == models.extraction.model
        assert captured["arguments"][4].model == models.generation.model
    else:
        assert captured["arguments"][3].model == models.generation.model
        if method in {"corrective", "graded-single-pass"}:
            assert captured["keywords"]["grade"].model == models.judging.model
        if method == "corrective":
            assert captured["keywords"]["rewrite"].model == models.extraction.model
