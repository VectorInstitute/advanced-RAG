"""Provenance encoding, decoding, and construction."""

from __future__ import annotations

import pytest

from aieng.advanced_rag.provenance import Provenance


class TestEncoding:
    """The flat form has to survive a graph property round trip."""

    def test_round_trip_with_every_field(self):
        original = Provenance("q1", "q1::0002", "Accepted answer", 10, 50, 3)
        assert Provenance.decode(original.encode()) == original

    def test_round_trip_with_only_a_document(self):
        original = Provenance("q1")
        assert Provenance.decode(original.encode()) == original

    def test_absent_fields_are_omitted_not_encoded_as_none(self):
        assert "None" not in Provenance("q1").encode()

    @pytest.mark.parametrize(
        "value", ["doc|with|pipes", "doc:with:colons", "both|and:mixed", "back\\slash"]
    )
    def test_separator_characters_survive(self, value):
        # Without escaping, a doc id containing "|" would split its own record.
        assert Provenance.decode(Provenance(value).encode()).doc_id == value

    def test_section_with_separators_survives(self):
        original = Provenance("d", section="Chapter 2: Setup | Advanced")
        assert Provenance.decode(original.encode()).section == original.section

    def test_decode_rejects_a_record_without_a_document(self):
        with pytest.raises(ValueError, match="no doc id"):
            Provenance.decode("chunk:c1|page:2")

    def test_decode_ignores_unknown_segments(self):
        assert Provenance.decode("doc:d1|nonsense|future_field:x").doc_id == "d1"


class TestSpanAndCitation:
    def test_span_returns_offsets(self):
        assert Provenance("d", start=5, end=9).span == (5, 9)

    def test_span_is_none_when_offsets_are_partial(self):
        assert Provenance("d", start=5).span is None

    def test_citation_includes_section_and_page(self):
        assert (
            Provenance("q1", section="Answer", page=3).citation()
            == "q1 > Answer (p. 3)"
        )

    def test_citation_degrades_to_the_document(self):
        assert Provenance("q1").citation() == "q1"


class _Chunk:
    def __init__(self, chunk_id, doc_id=None, contract_id=None, start=0, end=10):
        self.chunk_id = chunk_id
        self.text = "x"
        if doc_id:
            self.doc_id = doc_id
        if contract_id:
            self.contract_id = contract_id
        self.start, self.end = start, end


class TestConstruction:
    def test_from_chunk_with_doc_id(self):
        provenance = Provenance.from_chunk(_Chunk("c1", doc_id="d1"))
        assert (provenance.doc_id, provenance.chunk_id, provenance.span) == (
            "d1",
            "c1",
            (0, 10),
        )

    def test_from_chunk_accepts_the_harness_contract_id(self):
        # The CUAD harness names the document field contract_id.
        assert Provenance.from_chunk(_Chunk("c1", contract_id="k1")).doc_id == "k1"

    def test_from_chunk_without_a_document_raises(self):
        with pytest.raises(ValueError, match="document id"):
            Provenance.from_chunk(_Chunk("c1"))

    def test_from_docling_reads_the_page_number(self):
        item = {
            "prov": [{"page_no": 7}],
            "self_ref": "#/texts/3",
            "label": "section_header",
        }
        provenance = Provenance.from_docling("manual", item)
        assert (provenance.page, provenance.chunk_id, provenance.section) == (
            7,
            "#/texts/3",
            "section_header",
        )

    def test_from_docling_without_prov(self):
        assert Provenance.from_docling("manual", {}).page is None
