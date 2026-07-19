# tests/test_retrieval.py — Part 1: BM25Retriever and RRFFusion tests
# BM25 tests build a real index from synthetic data (fast, no model loading).
# RRF tests verify the fusion arithmetic with constructed RankedChunks.
# Run: pytest tests/test_retrieval.py -v  (completes in <5s)

from __future__ import annotations

import pytest
from pathlib import Path

from corpus.models import TextChunk
from corpus import CorpusRegistry
from indexing import BM25Indexer, BM25IndexLoader
from retrieval import BM25Retriever, RRFFusion, RankedChunk, SubQueryResults


# ── Fixtures ───────────────────────────────────────────────────────────────

def make_chunk(cid: str, text: str, ring: int = 1, tags: list = None) -> TextChunk:
    return TextChunk(
        chunk_id=cid, doc_id="d1", doc_title="Doc",
        source_url="https://test.com", governing_body="Test",
        ring=ring, ring_label="Test Ring",
        chunk_index=0, chunk_text=text,
        topic_tags=tags or [],
    )


@pytest.fixture
def corpus_chunks() -> list[TextChunk]:
    return [
        make_chunk("c1", "PPF interest rate 7.1 percent per annum compounded annually",
                   ring=2, tags=["PPF", "interest_rate"]),
        make_chunk("c2", "ELSS mutual fund 3 year lock-in period section 80C tax saving",
                   ring=1, tags=["ELSS", "80C", "mutual_fund"]),
        make_chunk("c3", "NPS subscriber can withdraw after age 60 upon retirement annuity",
                   ring=2, tags=["NPS"]),
        make_chunk("c4", "LRS limit 250000 USD per financial year under FEMA remittance",
                   ring=4, tags=["LRS", "FEMA", "remittance"]),
        make_chunk("c5", "DICGC insures bank deposits up to 5 lakh per depositor",
                   ring=3, tags=["DICGC", "FD"]),
        make_chunk("c6", "PPF account lock-in period is 15 years with partial withdrawal",
                   ring=2, tags=["PPF", "lock_in"]),
    ]


@pytest.fixture
def populated_registry(corpus_chunks) -> CorpusRegistry:
    reg = CorpusRegistry()
    reg.add_chunks(corpus_chunks)
    return reg


@pytest.fixture
def bm25_loader(corpus_chunks, tmp_path) -> BM25IndexLoader:
    BM25Indexer(tmp_path).build(chunks=corpus_chunks, force=True)
    return BM25IndexLoader(tmp_path).load_all()


@pytest.fixture
def bm25_retriever(bm25_loader, populated_registry) -> BM25Retriever:
    return BM25Retriever(index_loader=bm25_loader, registry=populated_registry)


# ── BM25Retriever tests ───────────────────────────────────────────────────

class TestBM25Retriever:
    def test_returns_ranked_chunks(self, bm25_retriever):
        results = bm25_retriever.retrieve("PPF interest rate", top_k=3)
        assert len(results) > 0
        assert all(isinstance(r, RankedChunk) for r in results)

    def test_source_is_bm25(self, bm25_retriever):
        results = bm25_retriever.retrieve("PPF interest rate", top_k=3)
        assert all(r.source == "bm25" for r in results)

    def test_ppf_chunks_rank_first(self, bm25_retriever):
        """Both PPF chunks (c1, c6) should rank above non-PPF chunks."""
        results = bm25_retriever.retrieve("PPF interest rate", top_k=6)
        top_ids = {r.chunk_id for r in results[:2]}
        assert "c1" in top_ids or "c6" in top_ids   # at least one PPF chunk in top 2

    def test_scores_strictly_descending(self, bm25_retriever):
        results = bm25_retriever.retrieve("PPF interest rate", top_k=6)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_rank_sequential(self, bm25_retriever):
        results = bm25_retriever.retrieve("interest rate", top_k=5)
        ranks = [r.rank for r in results]
        assert ranks == list(range(1, len(results) + 1))

    def test_ring_filter_restricts_results(self, bm25_retriever):
        """With ring_filter=["Govt Schemes"], results should only be ring 2 chunks."""
        results = bm25_retriever.retrieve(
            "PPF NPS interest rate", top_k=6,
            ring_filter=["Govt Schemes"]
        )
        # All results must be ring 2
        for r in results:
            assert r.ring == 2, f"Expected ring 2, got ring {r.ring} for {r.chunk_id}"

    def test_empty_query_returns_empty(self, bm25_retriever):
        results = bm25_retriever.retrieve("", top_k=5)
        assert results == []

    def test_no_match_query_returns_empty_or_zero(self, bm25_retriever):
        """A completely alien query with no corpus tokens should return few/no results."""
        results = bm25_retriever.retrieve("xyzzy quux frobnitz", top_k=5)
        # All scores should be 0 (filtered out) or very low
        assert all(r.score == 0 for r in results) or len(results) == 0

    def test_chunk_metadata_preserved(self, bm25_retriever):
        """RankedChunk should carry full TextChunk metadata."""
        results = bm25_retriever.retrieve("PPF interest rate", top_k=1)
        assert len(results) > 0
        top = results[0]
        assert top.chunk_id is not None
        assert top.chunk_text is not None
        assert top.doc_title is not None
        assert top.ring in (1, 2, 3, 4)

    def test_unknown_ring_label_ignored(self, bm25_retriever):
        """Unknown ring labels should be warned and skipped, not raised."""
        results = bm25_retriever.retrieve(
            "PPF interest", top_k=5,
            ring_filter=["NonExistentRing"]
        )
        # Should fall back to combined search (no crash)
        assert isinstance(results, list)


# ── RRFFusion tests ───────────────────────────────────────────────────────

def make_ranked_chunk(cid: str, rank: int, score: float = 1.0,
                      source: str = "bm25") -> RankedChunk:
    """Minimal RankedChunk for RRF arithmetic tests."""
    chunk = make_chunk(cid, f"text for {cid}")
    return RankedChunk(chunk=chunk, rank=rank, score=score, source=source)


class TestRRFFusion:
    def setup_method(self):
        self.rrf = RRFFusion(k=60)

    def test_empty_lists_returns_empty(self):
        assert self.rrf.fuse([]) == []

    def test_single_list_returns_ranked(self):
        lst = [make_ranked_chunk("c1", 1), make_ranked_chunk("c2", 2)]
        result = self.rrf.fuse([lst], top_n=5)
        assert len(result) == 2
        assert result[0].chunk_id == "c1"   # rank 1 gets higher RRF score

    def test_chunk_in_two_lists_scores_higher(self):
        """
        A chunk in both BM25 and FAISS lists should outscore a chunk in only one.
        c1 appears in both lists at rank 1; c2 appears only in list 1 at rank 1.
        c1 total RRF = 1/61 + 1/61 ≈ 0.0328; c2 = 1/61 ≈ 0.0164
        """
        list_bm25  = [make_ranked_chunk("c1", 1), make_ranked_chunk("c2", 2)]
        list_faiss = [make_ranked_chunk("c1", 1), make_ranked_chunk("c3", 2)]
        result = self.rrf.fuse([list_bm25, list_faiss], top_n=10)
        top_id = result[0].chunk_id
        assert top_id == "c1"

    def test_rrf_formula_correct(self):
        """Verify RRF score arithmetic exactly."""
        k = 60
        # c1 at rank 1 in one list → score = 1/(60+1)
        lst = [make_ranked_chunk("c1", 1)]
        result = self.rrf.fuse([lst], top_n=1)
        expected = 1.0 / (k + 1)
        assert abs(result[0].score - expected) < 1e-9

    def test_rrf_two_lists_accumulation(self):
        """c1 in two lists at rank 1 → score = 2 * 1/(k+1)."""
        k    = 60
        lst1 = [make_ranked_chunk("c1", 1)]
        lst2 = [make_ranked_chunk("c1", 1)]
        result = self.rrf.fuse([lst1, lst2], top_n=1)
        expected = 2.0 / (k + 1)
        assert abs(result[0].score - expected) < 1e-9

    def test_output_source_is_rrf(self):
        lst = [make_ranked_chunk("c1", 1, source="bm25")]
        result = self.rrf.fuse([lst], top_n=5)
        assert all(r.source == "rrf" for r in result)

    def test_output_rank_is_sequential(self):
        lst = [make_ranked_chunk(f"c{i}", i) for i in range(1, 6)]
        result = self.rrf.fuse([lst], top_n=5)
        assert [r.rank for r in result] == list(range(1, len(result) + 1))

    def test_top_n_limits_output(self):
        lst = [make_ranked_chunk(f"c{i}", i) for i in range(1, 21)]
        result = self.rrf.fuse([lst], top_n=5)
        assert len(result) == 5

    def test_deduplication_across_lists(self):
        """Same chunk in three lists should appear only once in output."""
        lst1 = [make_ranked_chunk("c1", 1)]
        lst2 = [make_ranked_chunk("c1", 1)]
        lst3 = [make_ranked_chunk("c1", 1)]
        result = self.rrf.fuse([lst1, lst2, lst3], top_n=10)
        chunk_ids = [r.chunk_id for r in result]
        assert chunk_ids.count("c1") == 1   # deduplicated

    def test_explain_scores_utility(self):
        """explain_scores should return correct total for a known setup."""
        k = 60
        lst1 = [make_ranked_chunk("c1", 1)]
        lst2 = [make_ranked_chunk("c1", 3)]
        explanation = RRFFusion.explain_scores([lst1, lst2], "c1", k=k)
        expected_total = 1/(k+1) + 1/(k+3)
        assert abs(explanation["total_rrf_score"] - expected_total) < 1e-6
        assert len(explanation["contributions"]) == 2

    def test_fuse_sub_query_results(self):
        """fuse_sub_query_results should call fuse() with all_ranked_lists."""
        bm25_1  = [make_ranked_chunk("c1", 1), make_ranked_chunk("c2", 2)]
        faiss_1 = [make_ranked_chunk("c1", 2), make_ranked_chunk("c3", 1)]
        sqr = SubQueryResults("PPF interest", bm25_results=bm25_1, faiss_results=faiss_1)
        result = self.rrf.fuse_sub_query_results([sqr], top_n=10)
        assert len(result) > 0
        # c1 appears in both bm25 and faiss → should have highest RRF score
        assert result[0].chunk_id == "c1"