# tests/test_chunker.py
# Unit tests for the corpus chunking pipeline.
# Run: pytest tests/test_chunker.py -v

import pytest
from corpus.chunker import count_tokens, recursive_split, DocumentChunker
from corpus.models import SourceDocument, TextChunk
from config import settings


@pytest.fixture
def ppf_doc():
    return SourceDocument(
        doc_id=SourceDocument.make_doc_id("https://test.example.com/ppf"),
        title="PPF Scheme Rules 2019", source_url="https://test.example.com/ppf",
        governing_body="Ministry of Finance", ring=2, ring_label="Govt Schemes",
        effective_date="2019-12-12",
        raw_text=(
            "The Public Provident Fund offers tax benefits under Section 80C.\n\n"
            "Interest rate: 7.1% per annum, compounded annually.\n\n"
            "Minimum deposit: INR 500. Maximum: INR 1.5 lakh per year.\n\n"
            "Lock-in period: 15 years. Partial withdrawal from year 7 onwards."
        ),
    )


@pytest.fixture
def rbi_doc():
    return SourceDocument(
        doc_id=SourceDocument.make_doc_id("https://test.example.com/rbi"),
        title="RBI Master Direction on Deposits",
        source_url="https://test.example.com/rbi",
        governing_body="RBI", ring=3, ring_label="Banking & RBI",
        effective_date="2023-01-01", circular_ref="RBI/2023-24/10",
        raw_text=(
            "RBI/2023-24/10\nJanuary 01, 2023\n"
            "Subject: Master Direction on Interest Rate on Deposits\n\n"
            "Savings bank deposits shall bear interest at not less than 3.5% per annum.\n\n"
            "Fixed deposits with maturity up to 1 year shall bear interest "
            "as decided by the bank Board of Directors."
        ),
    )


class TestTokenCounter:
    def test_empty(self):          assert count_tokens("") == 0
    def test_deterministic(self):  assert count_tokens("hello") == count_tokens("hello")
    def test_reasonable_range(self):
        assert 5 <= count_tokens("The PPF interest rate is 7.1% per annum.") <= 20


class TestRecursiveSplit:
    def test_short_text_single_chunk(self):
        chunks = recursive_split("Short text about PPF.", chunk_size=512, min_chunk_size=5)
        assert len(chunks) == 1

    def test_token_budget_respected(self):
        text = " ".join(["word"] * 300)   # ~300 tokens
        chunks = recursive_split(text, chunk_size=100, chunk_overlap=10, min_chunk_size=10)
        for c in chunks:
            assert count_tokens(c) <= 100

    def test_empty_text(self):
        assert recursive_split("", chunk_size=512, min_chunk_size=10) == []

    def test_min_size_filters_fragments(self):
        text = "Tiny.\n\n" + "word " * 100
        chunks = recursive_split(text, chunk_size=200, chunk_overlap=10, min_chunk_size=50)
        for c in chunks:
            assert count_tokens(c) >= 50


class TestDocumentChunker:
    def setup_method(self):
        self.chunker = DocumentChunker()

    def test_returns_textchunks(self, ppf_doc):
        assert all(isinstance(c, TextChunk) for c in self.chunker.chunk_document(ppf_doc))

    def test_ids_unique_and_deterministic(self, ppf_doc):
        ids1 = [c.chunk_id for c in self.chunker.chunk_document(ppf_doc)]
        ids2 = [c.chunk_id for c in self.chunker.chunk_document(ppf_doc)]
        assert len(ids1) == len(set(ids1))   # unique
        assert ids1 == ids2                  # deterministic

    def test_rbi_prefix_in_all_chunks(self, rbi_doc):
        chunks = self.chunker.chunk_document(rbi_doc)
        assert all("RBI/2023-24/10" in c.chunk_text for c in chunks)

    def test_empty_doc_returns_empty(self, ppf_doc):
        ppf_doc.raw_text = ""
        assert self.chunker.chunk_document(ppf_doc) == []

    def test_sequential_chunk_indices(self, ppf_doc):
        chunks = self.chunker.chunk_document(ppf_doc)
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_metadata_propagated(self, ppf_doc):
        for c in self.chunker.chunk_document(ppf_doc):
            assert c.doc_id == ppf_doc.doc_id
            assert c.ring == ppf_doc.ring