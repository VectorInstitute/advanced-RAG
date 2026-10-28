"""Tests for grounded generation, citations, and abstention."""

from __future__ import annotations

import json
from types import SimpleNamespace

from aieng.advanced_rag.metering import UsageMeter
from aieng.advanced_rag.pipeline.generate import (
    ABSTENTION_TEXT,
    ChatCompleter,
    GroundedAnswer,
    abstention_flags,
    generate,
    looks_like_abstention,
    parse_answer,
)
from aieng.advanced_rag.reasoning import THINKING_OFF


class Recorder:
    def __init__(self, response: str = "") -> None:
        self.response = response
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


def answer_json(answer: str, sources: list[int], not_stated: bool = False) -> str:
    return json.dumps({"answer": answer, "sources": sources, "not_stated": not_stated})


class FakeCompletions:
    """Stands in for ``client.chat.completions``, recording what it was sent."""

    def __init__(
        self,
        content: str | None,
        with_choices: bool = True,
        usage: tuple[int, int] | None = None,
    ) -> None:
        self.content = content
        self.with_choices = with_choices
        self.usage = usage
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        # The proxy returns both token detail blocks as null, so a meter has only
        # the three totals to read.
        block = (
            SimpleNamespace(
                prompt_tokens=self.usage[0],
                completion_tokens=self.usage[1],
                total_tokens=sum(self.usage),
                prompt_tokens_details=None,
                completion_tokens_details=None,
            )
            if self.usage
            else None
        )
        if not self.with_choices:
            return SimpleNamespace(choices=[], usage=block)
        message = SimpleNamespace(content=self.content, tool_calls=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            usage=block,
        )


def fake_client(
    content: str | None,
    with_choices: bool = True,
    usage: tuple[int, int] | None = None,
):
    completions = FakeCompletions(content, with_choices, usage)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return client, completions


def test_a_well_formed_answer_resolves_its_citations(make_evidence):
    evidence = make_evidence(["Ontario law governs.", "Unrelated clause."])
    answer = parse_answer("q1", answer_json("Ontario law governs.", [1]), evidence)
    assert not answer.abstained
    assert answer.chunk_ids == (evidence[0].chunk_id,)
    assert answer.citations[0] is evidence[0].provenance


def test_the_model_reporting_not_stated_abstains(make_evidence):
    answer = parse_answer("q1", answer_json("", [], True), make_evidence(["a"]))
    assert answer.abstained
    assert answer.text == ABSTENTION_TEXT
    assert "the sources do not answer" in answer.reason


def test_a_refusal_in_prose_abstains_when_the_format_was_ignored(make_evidence):
    answer = parse_answer(
        "q1", "The governing law is not stated in these sources.", make_evidence(["a"])
    )
    assert answer.abstained
    assert answer.reason == "the answer reads as a refusal"


def test_the_models_own_flag_beats_the_phrase_heuristic(make_evidence):
    # An answer can legitimately contain "not specified" while still answering,
    # so the flag is authoritative whenever it is present.
    raw = answer_json("Ontario governs; the venue is not specified.", [1])
    answer = parse_answer("q1", raw, make_evidence(["a"]))
    assert not answer.abstained


def test_the_phrase_fallback_can_be_switched_off(make_evidence):
    raw = "Not stated anywhere [1]."
    answer = parse_answer("q1", raw, make_evidence(["a"]), phrase_fallback=False)
    assert not answer.abstained


def test_an_empty_completion_abstains(make_evidence):
    # ChatCompleter turns content=None into "", which lands here.
    answer = parse_answer("q1", "", make_evidence(["a"]))
    assert answer.abstained
    assert answer.reason == "the model returned no content"


def test_an_uncited_answer_abstains_when_the_citation_gate_is_on(make_evidence):
    raw = answer_json("Ontario law governs.", [])
    assert parse_answer("q1", raw, make_evidence(["a"])).abstained


def test_the_citation_gate_can_be_switched_off(make_evidence):
    raw = answer_json("Ontario law governs.", [])
    answer = parse_answer("q1", raw, make_evidence(["a"]), require_citations=False)
    assert not answer.abstained
    assert answer.citations == ()


def test_a_citation_to_a_source_that_was_never_shown_is_recorded_and_dropped(
    make_evidence,
):
    raw = answer_json("Ontario law governs.", [1, 9])
    answer = parse_answer("q1", raw, make_evidence(["a", "b"]))
    assert answer.dropped_citations == (9,)
    assert len(answer.citations) == 1


def test_an_answer_citing_only_a_source_that_was_never_shown_abstains(make_evidence):
    raw = answer_json("Ontario law governs.", [9])
    answer = parse_answer("q1", raw, make_evidence(["a"]))
    assert answer.abstained
    assert answer.reason == "the answer cited no usable source"


def test_inline_markers_are_read_when_the_json_lists_no_sources(make_evidence):
    raw = json.dumps({"answer": "Ontario law governs [2].", "not_stated": False})
    answer = parse_answer("q1", raw, make_evidence(["a", "b"]))
    assert answer.chunk_ids == ("doc1::001",)


def test_a_rejected_draft_is_kept_in_raw_and_never_shown_as_the_answer(
    make_evidence,
):
    raw = answer_json("Ontario law governs.", [])
    answer = parse_answer("q1", raw, make_evidence(["a"]))
    assert answer.text == ABSTENTION_TEXT
    assert "Ontario law governs." in answer.raw


def test_generate_abstains_with_no_evidence_and_makes_no_call():
    model = Recorder(answer_json("something", [1]))
    answer = generate("q1", "who signs", [], model)
    assert answer.abstained
    assert model.prompts == []
    assert answer.reason == "no evidence to answer from"


def test_generate_places_only_the_budgeted_passages_in_the_prompt(make_evidence):
    model = Recorder(answer_json("a", [1]))
    evidence = make_evidence(["first passage", "second passage", "third passage"])
    generate("q1", "who signs", evidence, model, budget=2)
    prompt = model.prompts[0]
    assert "second passage" in prompt
    assert "third passage" not in prompt


def test_generate_asks_the_question_the_reader_asked(make_evidence):
    model = Recorder(answer_json("a", [1]))
    generate("q1", "who signs the agreement", make_evidence(["a"]), model)
    assert "who signs the agreement" in model.prompts[0]


def test_abstention_flags_are_what_the_null_scorer_reads():
    answers = [
        GroundedAnswer(query_id="q1", text="an answer"),
        GroundedAnswer.abstain("q2", reason="nothing retrieved"),
    ]
    assert abstention_flags(answers) == {"q1": False, "q2": True}


def test_an_answer_serialises_for_a_report(make_evidence):
    answer = parse_answer("q1", answer_json("Ontario.", [1]), make_evidence(["a"]))
    payload = json.loads(json.dumps(answer.to_dict()))
    assert payload["chunk_ids"] == ["doc1::000"]
    assert payload["abstained"] is False


def test_abstention_phrases_are_matched_case_insensitively():
    assert looks_like_abstention("The contract DOES NOT STATE a governing law.")
    assert not looks_like_abstention("The governing law is Ontario.")


def test_the_completer_disables_reasoning_on_every_call():
    client, completions = fake_client("hello")
    complete = ChatCompleter(client, "test-model")
    assert complete("say hello") == "hello"
    assert completions.calls[0]["extra_body"] == THINKING_OFF


def test_the_completer_turns_a_missing_content_into_an_empty_string():
    # The proxy returns content=None when max_tokens runs out during reasoning.
    client, _ = fake_client(None)
    complete = ChatCompleter(client, "test-model")
    assert complete("prompt") == ""
    assert complete.empty_responses == 1
    assert complete.calls == 1


def test_the_completer_survives_a_response_with_no_choices():
    client, _ = fake_client("hello", with_choices=False)
    complete = ChatCompleter(client, "test-model")
    assert complete("prompt") == ""
    assert complete.empty_responses == 1


def test_reasoning_can_be_left_on_for_a_non_reasoning_deployment():
    client, completions = fake_client("hello")
    ChatCompleter(client, "test-model", disable_reasoning=False)("prompt")
    assert "extra_body" not in completions.calls[0]


def test_the_completer_keeps_the_tokens_every_call_reported():
    client, _ = fake_client("hello", usage=(282, 26))
    complete = ChatCompleter(client, "test-model")
    complete("prompt")
    complete("prompt")

    assert complete.prompt_tokens == 564
    assert complete.completion_tokens == 52
    assert complete.last_usage is not None
    assert complete.last_usage.total_tokens == 308


def test_a_response_with_no_usage_leaves_the_token_totals_alone():
    client, _ = fake_client("hello")
    complete = ChatCompleter(client, "test-model")
    complete("prompt")

    assert complete.prompt_tokens == 0
    assert complete.last_usage is not None
    assert not complete.last_usage.reported


def test_the_completer_reports_every_call_to_the_meter():
    meter = UsageMeter()
    client, _ = fake_client("hello", usage=(282, 26))
    complete = ChatCompleter(client, "test-model")
    with meter.question("q1"):
        complete("prompt")

    cost = meter.per_question()["q1"]
    assert cost.calls == 1
    assert cost.total_tokens == 308
    assert cost.service_seconds > 0.0
