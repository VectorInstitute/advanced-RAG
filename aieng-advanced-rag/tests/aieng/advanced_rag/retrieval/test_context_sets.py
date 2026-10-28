"""Context sets, the two prompts, the free header, and the price estimate.

No model is called. A stub client stands in for the proxy and reports usage the
way an OpenAI-compatible response does, so the tests cover what the notebook
relies on: a context set replays exactly from JSON, the shared prefix of a
document is identical for every chunk so a prefix cache can serve it, and the
cached estimate bills one write and then reads per document.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from aieng.advanced_rag.retrieval.chunkers import fixed_chars
from aieng.advanced_rag.retrieval.contextual import (
    FULL_DOCUMENT,
    FULL_DOCUMENT_NAMED,
    HEAD_NEIGHBOURHOOD,
    SYSTEM,
    ContextRecord,
    ContextSet,
    ModelPrice,
    build_prompt,
    clean_name,
    clean_source_name,
    context_messages,
    generate_contexts,
    header_chunks,
    metadata_header,
    name_documents,
    naming_queries,
    price_contexts,
    section_at,
)
from aieng.advanced_rag.retrieval.corpus import AnswerSpan, Query


DOCUMENT = (
    "MUTUAL NDA between Acme Ltd and Globex.\n\n1. DEFINITIONS\n"
    + "The parties agree. " * 120
)
OTHER = "SUPPLY AGREEMENT for Initech.\n\n" + "Goods are supplied. " * 60


class StubCompletions:
    """Answers every call, records the requests, and reports usage."""

    def __init__(
        self, answer: str = "Context for the chunk.", fail_on: str | None = None
    ) -> None:
        self.answer = answer
        self.fail_on = fail_on
        self.requests: list[dict] = []
        self._lock = threading.Lock()

    def create(self, **kwargs):
        with self._lock:
            self.requests.append(kwargs)
        text = str(kwargs["messages"][-1]["content"])
        if self.fail_on and self.fail_on in text:
            raise RuntimeError("the model is unavailable")
        usage = SimpleNamespace(
            prompt_tokens=1000,
            completion_tokens=20,
            prompt_tokens_details=SimpleNamespace(cached_tokens=600),
        )
        message = SimpleNamespace(content=f"  {self.answer}  ")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def stub_client(**kwargs) -> SimpleNamespace:
    return SimpleNamespace(chat=SimpleNamespace(completions=StubCompletions(**kwargs)))


def chunks_of(doc_id: str, text: str):
    return fixed_chars(doc_id, text, chunk_chars=400, overlap_chars=80)


class TestContextMessages:
    def test_the_full_document_prompt_puts_the_document_first(self):
        chunk = chunks_of("acme", DOCUMENT)[2]
        messages, prefix_chars = context_messages(FULL_DOCUMENT, DOCUMENT, chunk)
        user = messages[1]["content"]
        assert user.startswith("<document>\n" + DOCUMENT)
        assert user.index(DOCUMENT) < user.index(chunk.text, len(DOCUMENT))
        assert prefix_chars == len(SYSTEM) + len(f"<document>\n{DOCUMENT}\n</document>")

    def test_every_chunk_of_a_document_shares_the_same_prefix(self):
        prefixes = set()
        for chunk in chunks_of("acme", DOCUMENT)[:4]:
            messages, prefix_chars = context_messages(FULL_DOCUMENT, DOCUMENT, chunk)
            prefixes.add(messages[1]["content"][: prefix_chars - len(SYSTEM)])
        assert len(prefixes) == 1

    def test_the_head_prompt_is_unchanged_from_the_shipped_one(self):
        chunk = chunks_of("acme", DOCUMENT)[3]
        messages, _ = context_messages(HEAD_NEIGHBOURHOOD, DOCUMENT, chunk)
        assert messages[1]["content"] == build_prompt(DOCUMENT, chunk)

    def test_cache_control_marks_only_the_prefix(self):
        chunk = chunks_of("acme", DOCUMENT)[1]
        messages, _ = context_messages(
            FULL_DOCUMENT, DOCUMENT, chunk, cache_control=True
        )
        parts = messages[1]["content"]
        assert parts[0]["cache_control"] == {"type": "ephemeral"}
        assert "cache_control" not in parts[1]
        assert chunk.text in parts[1]["text"]

    def test_an_unknown_prompt_version_is_refused(self):
        with pytest.raises(ValueError, match="unknown prompt version"):
            context_messages("v0", DOCUMENT, chunks_of("acme", DOCUMENT)[0])


class TestGenerateContexts:
    def test_each_chunk_gets_a_record_labelled_with_the_model(self):
        documents = {"acme": DOCUMENT, "initech": OTHER}
        chunks = chunks_of("acme", DOCUMENT) + chunks_of("initech", OTHER)
        client = stub_client()
        contexts = generate_contexts(
            documents, chunks, client, model="m-1", workers=2, progress_every=0
        )
        assert contexts.model == "m-1"
        assert contexts.prompt_version == FULL_DOCUMENT
        assert set(contexts.records) == {c.chunk_id for c in chunks}
        record = contexts.records[chunks[0].chunk_id]
        assert record.context == "Context for the chunk."
        assert (
            record.prompt_tokens,
            record.completion_tokens,
            record.cached_tokens,
        ) == (1000, 20, 600)
        assert {r["model"] for r in client.chat.completions.requests} == {"m-1"}

    def test_a_failed_call_is_listed_and_the_chunk_stays_plain(self):
        chunks = chunks_of("acme", DOCUMENT)
        target = chunks[1]
        client = stub_client(fail_on=f"<chunk>\n{target.text}\n</chunk>")
        contexts = generate_contexts(
            {"acme": DOCUMENT}, chunks, client, model="m", workers=1, progress_every=0
        )
        assert target.chunk_id in contexts.failures
        applied = contexts.apply(chunks)
        assert applied[1].indexed_text == target.text
        assert applied[0].indexed_text.startswith("Context for the chunk.")

    def test_a_set_replays_exactly_from_json(self):
        chunks = chunks_of("acme", DOCUMENT)
        contexts = generate_contexts(
            {"acme": DOCUMENT}, chunks, stub_client(), model="m", progress_every=0
        )
        again = ContextSet.from_json(contexts.to_json())
        assert again == contexts
        assert [c.indexed_text for c in again.apply(chunks)] == [
            c.indexed_text for c in contexts.apply(chunks)
        ]

    def test_applying_keeps_the_span_and_the_text(self):
        chunks = chunks_of("acme", DOCUMENT)
        contexts = generate_contexts(
            {"acme": DOCUMENT}, chunks, stub_client(), model="m", progress_every=0
        )
        for before, after in zip(chunks, contexts.apply(chunks)):
            assert (after.chunk_id, after.text, after.start, after.end) == (
                before.chunk_id,
                before.text,
                before.start,
                before.end,
            )


def record(doc_id: str, prompt: int = 1000, prefix_share: float = 0.8) -> ContextRecord:
    return ContextRecord(
        doc_id=doc_id,
        context="c",
        prompt_tokens=prompt,
        completion_tokens=50,
        prefix_chars=int(1000 * prefix_share),
        prompt_chars=1000,
    )


class TestPriceContexts:
    PRICE = ModelPrice(
        "cheap", 1.0, 10.0, cache_write_multiplier=1.25, cache_read_multiplier=0.1
    )

    def test_uncached_bills_every_prompt_token_at_the_input_rate(self):
        contexts = ContextSet(
            "m", FULL_DOCUMENT, {"a::0": record("a"), "a::1": record("a")}
        )
        estimate = price_contexts(contexts, self.PRICE)
        assert estimate["uncached_prompt_tokens"] == 2000
        assert estimate["usd"] == pytest.approx((2000 * 1.0 + 100 * 10.0) / 1e6)

    def test_cached_writes_once_per_document_and_reads_after(self):
        contexts = ContextSet(
            "m",
            FULL_DOCUMENT,
            {
                "a::0": record("a"),
                "a::1": record("a"),
                "a::2": record("a"),
                "b::0": record("b"),
            },
        )
        estimate = price_contexts(contexts, self.PRICE, cached=True)
        assert estimate["cache_write_tokens"] == 1600
        assert estimate["cache_read_tokens"] == 1600
        assert estimate["uncached_prompt_tokens"] == 800
        expected = (800 * 1.0 + 1600 * 1.25 + 1600 * 0.1 + 200 * 10.0) / 1e6
        assert estimate["usd"] == pytest.approx(expected, abs=1e-4)

    def test_a_prefix_below_the_minimum_is_not_cached(self):
        price = ModelPrice("p", 1.0, 1.0, 1.25, 0.1, min_cacheable_tokens=5000)
        contexts = ContextSet(
            "m", FULL_DOCUMENT, {"a::0": record("a"), "a::1": record("a")}
        )
        estimate = price_contexts(contexts, price, cached=True)
        assert estimate["cache_read_tokens"] == 0
        assert estimate["uncached_prompt_tokens"] == 2000

    def test_a_cached_estimate_needs_cache_rates(self):
        with pytest.raises(ValueError, match="no cache rates"):
            price_contexts(
                ContextSet("m", FULL_DOCUMENT), ModelPrice("p", 1.0, 1.0), cached=True
            )


class TestNaming:
    def test_names_are_cleaned_and_counted(self):
        client = stub_client(answer='"The mutual NDA between Acme Ltd and Globex."')
        names, usage = name_documents({"acme": DOCUMENT}, client, model="m")
        assert names == {"acme": "The mutual NDA between Acme Ltd and Globex"}
        assert usage == {"calls": 1, "prompt_tokens": 1000, "completion_tokens": 20}

    def test_clean_name_puts_the_name_on_one_line(self):
        assert clean_name("  'Acme\n NDA'. ") == "Acme NDA"

    def test_a_named_query_keeps_its_identity_and_judgments(self):
        query = Query(
            "acme::nda-1",
            "acme",
            "Purpose",
            "Use is limited.",
            (AnswerSpan("x", 0, 1),),
        )
        orphan = Query("other::nda-1", "other", "Purpose", "Use is limited.")
        named, unchanged = naming_queries([query, orphan], {"acme": "the Acme NDA"})
        assert named.question == "In the Acme NDA: Use is limited."
        assert (named.query_id, named.doc_id, named.answers) == (
            query.query_id,
            "acme",
            query.answers,
        )
        assert unchanged.question == "Use is limited."


class TestMetadataHeader:
    def test_source_names_become_words(self):
        assert clean_source_name("01_Acme-Mutual%20NDA.pdf") == "01 Acme Mutual NDA"

    def test_the_section_is_the_last_heading_before_the_chunk(self):
        headings = [(0, "PREAMBLE"), (100, "1. DEFINITIONS"), (900, "2. TERM")]
        assert section_at(headings, 500) == "1. DEFINITIONS"
        assert section_at(headings, 50) == "PREAMBLE"
        assert section_at([], 50) == ""

    def test_the_header_names_the_file_the_opening_and_the_section(self):
        chunk = chunks_of("acme", DOCUMENT)[2]
        header = metadata_header(
            DOCUMENT,
            chunk,
            source_name="acme_nda.pdf",
            headings=[(42, "1. DEFINITIONS")],
            head_chars=40,
        )
        assert header.splitlines() == [
            "Document: acme nda",
            "Opening: MUTUAL NDA between Acme Ltd and Globex.",
            "Section: 1. DEFINITIONS",
        ]

    def test_header_chunks_change_only_the_indexed_text(self):
        chunks = chunks_of("acme", DOCUMENT)
        labelled = header_chunks(
            chunks, {"acme": DOCUMENT}, source_names={"acme": "acme.pdf"}
        )
        for before, after in zip(chunks, labelled):
            assert after.text == before.text
            assert after.span == before.span
            assert after.indexed_text.startswith("Document: acme\n")
            assert after.indexed_text.endswith(before.text)


def test_the_named_full_document_prompt_shares_the_prefix_and_asks_for_parties():
    chunk = chunks_of("acme", DOCUMENT)[2]
    plain_messages, plain_prefix = context_messages(FULL_DOCUMENT, DOCUMENT, chunk)
    named_messages, named_prefix = context_messages(
        FULL_DOCUMENT_NAMED, DOCUMENT, chunk
    )
    assert plain_prefix == named_prefix
    assert "Name the document, the parties" in named_messages[1]["content"]
    assert "Name the document" not in plain_messages[1]["content"]


class AnthropicCompletions(StubCompletions):
    """Reports usage the way an Anthropic model does behind the proxy."""

    def create(self, **kwargs):
        response = super().create(**kwargs)
        first = not any(
            r is not kwargs and r["messages"][0] == kwargs["messages"][0]
            for r in self.requests[:-1]
        )
        response.usage = SimpleNamespace(
            prompt_tokens=30,
            completion_tokens=20,
            cache_read_input_tokens=0 if first else 900,
            cache_creation_input_tokens=900 if first else 0,
        )
        return response


class TestPaidModelCaching:
    def test_a_caching_family_gets_the_shared_prefix_in_a_cacheable_system_message(
        self,
    ):
        chunk = chunks_of("acme", DOCUMENT)[2]
        messages, prefix_chars = context_messages(
            FULL_DOCUMENT, DOCUMENT, chunk, cache_control=True, model="claude-haiku-5-5"
        )
        system = messages[0]["content"]
        assert [p["text"] for p in system][0] == SYSTEM
        assert system[-1]["cache_control"] == {"type": "ephemeral"}
        assert system[-1]["text"].startswith("<document>")
        assert messages[1]["content"].startswith("Here is the chunk")
        assert chunk.text in messages[1]["content"]
        assert prefix_chars == len(SYSTEM) + len(system[-1]["text"])

    def test_every_chunk_of_a_document_shares_the_system_message(self):
        chunks = chunks_of("acme", DOCUMENT)
        systems = {
            repr(
                context_messages(
                    FULL_DOCUMENT,
                    DOCUMENT,
                    c,
                    cache_control=True,
                    model="claude-haiku-5-5",
                )[0][0]
            )
            for c in chunks
        }
        assert len(systems) == 1

    def test_generate_contexts_turns_caching_on_for_the_family_by_default(self):
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=AnthropicCompletions())
        )
        generate_contexts(
            {"acme": DOCUMENT},
            chunks_of("acme", DOCUMENT),
            client,
            model="claude-haiku-5-5",
            workers=1,
            progress_every=0,
        )
        for request in client.chat.completions.requests:
            assert request["messages"][0]["content"][-1]["cache_control"]

    def test_cache_reads_and_writes_count_in_the_set_usage(self):
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=AnthropicCompletions())
        )
        chunks = chunks_of("acme", DOCUMENT)
        contexts = generate_contexts(
            {"acme": DOCUMENT},
            chunks,
            client,
            model="claude-haiku-5-5",
            workers=1,
            progress_every=0,
        )
        usage = contexts.usage()
        n = len(chunks)
        assert usage["prompt_tokens"] == n * 30 + 900 * n
        assert usage["cached_tokens_reported"] == 900 * (n - 1)
        assert usage["cache_write_tokens_reported"] == 900
        restored = ContextSet.from_json(contexts.to_json())
        assert restored.usage() == usage

    def test_an_on_prem_request_is_unchanged_by_the_caching_option(self):
        chunk = chunks_of("acme", DOCUMENT)[2]
        plain, plain_chars = context_messages(FULL_DOCUMENT, DOCUMENT, chunk)
        for kwargs in (
            {"cache_control": True, "model": "Qwen3_6-35B-A3B"},
            {"model": "Qwen3_6-35B-A3B"},
        ):
            assert context_messages(FULL_DOCUMENT, DOCUMENT, chunk, **kwargs) == (
                plain,
                plain_chars,
            )
        assert plain[0] == {"role": "system", "content": SYSTEM}
        assert isinstance(plain[1]["content"], str)

    def test_generate_contexts_sends_the_on_prem_request_it_always_sent(self):
        chunks = chunks_of("acme", DOCUMENT)
        client = stub_client()
        generate_contexts(
            {"acme": DOCUMENT},
            chunks,
            client,
            model="Qwen3_6-35B-A3B",
            workers=1,
            progress_every=0,
        )
        sent = client.chat.completions.requests
        assert [r["messages"] for r in sent] == [
            context_messages(FULL_DOCUMENT, DOCUMENT, c)[0]
            for c in sorted(chunks, key=lambda c: (c.start, c.chunk_id))
        ]
        assert set(sent[0]) == {"model", "messages", "max_tokens", "temperature"}

    def test_an_on_prem_set_serialises_without_cache_write_keys(self):
        client = stub_client()
        contexts = generate_contexts(
            {"acme": DOCUMENT},
            chunks_of("acme", DOCUMENT),
            client,
            model="Qwen3_6-35B-A3B",
            workers=1,
            progress_every=0,
        )
        assert "cache_write_tokens_reported" not in contexts.usage()
        record = next(iter(contexts.to_json()["records"].values()))
        assert "cache_write_tokens" not in record


def test_the_ledger_and_collect_usage_carry_cache_reads_and_writes():
    from aieng.advanced_rag import metering
    from aieng.advanced_rag.graphrag.proxy import CallLedger
    from aieng.advanced_rag.retrieval.contextual import (
        ContextualCost,
        collect_usage,
    )

    ledger = CallLedger()
    response = {
        "usage": {
            "prompt_tokens": 30,
            "completion_tokens": 5,
            "cache_read_input_tokens": 800,
            "cache_creation_input_tokens": 100,
        }
    }
    ledger.record(metering.usage_of(response), 0.1)
    cost = collect_usage([SimpleNamespace(usage=ledger)], ContextualCost(chunks=1))
    assert (cost.prompt_tokens, cost.cache_read_tokens, cost.cache_write_tokens) == (
        930,
        800,
        100,
    )
    assert cost.to_dict()["cache_write_tokens"] == 100
    assert "cache_read_tokens" not in ContextualCost(chunks=1).to_dict()
