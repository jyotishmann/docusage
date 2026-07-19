# tests/test_indexing.py
# Fast unit tests: tokeniser, BM25, FAISS (no model loading).
# Run: pytest tests/test_indexing.py -v  (completes in <3s)

from __future__ import annotations
import json, pickle
from pathlib import Path
import numpy as np
import pytest
from unittest.mock import patch, PropertyMock

from corpus.models import TextChunk
from indexing.bm25_indexer import BM25Indexer, BM25IndexLoader, tokenise
from indexing.faiss_indexer import FAISSIndexer


def make_chunk(cid: str, text: str, ring: int = 1) -> TextChunk:
    return TextChunk(
        chunk_id=cid, doc_id="d1", doc_title="T",
        source_url="https://t.com", governing_body="Test",
        ring=ring, ring_label="Test", chunk_index=0,
        chunk_text=text, topic_tags=[],
    )


@pytest.fixture
def chunks():
    return [
        make_chunk("c1", "PPF interest rate 7.1 percent per annum compounded annually", ring=2),
        make_chunk("c2", "ELSS mutual fund 3 year lock-in period section 80C", ring=1),
        make_chunk("c3", "NPS subscriber withdraw at age 60 upon retirement", ring=2),
        make_chunk("c4", "LRS limit 250000 USD per financial year FEMA", ring=4),
        make_chunk("c5", "DICGC insures deposits 5 lakh per depositor per bank", ring=3),
    ]


# ── Tokeniser ─────────────────────────────────────────────────────────────
class TestTokenise:
    def test_lowercases(self):
        assert "ppf" in tokenise("PPF Interest Rate")

    def test_removes_single_chars(self):
        assert "a" not in tokenise("a b PPF")
        assert "ppf" in tokenise("a b PPF")

    def test_preserves_acronyms(self):
        for term in ["ELSS", "SCSS", "80CCD", "FCNR"]:
            assert term.lower() in tokenise(term)

    def test_empty(self):
        assert tokenise("") == []

    def test_no_stemming(self):
        # "interest" and "interests" should be different tokens
        t1 = tokenise("interest rate")
        t2 = tokenise("interests")
        assert "interest" in t1 and "interests" in t2


# ── BM25 Indexer ──────────────────────────────────────────────────────────
class TestBM25Indexer:
    def test_creates_combined_pickle(self, chunks, tmp_path):
        BM25Indexer(tmp_path).build(chunks=chunks, force=True)
        assert (tmp_path / "bm25_combined.pkl").exists()

    def test_correct_size(self, chunks, tmp_path):
        BM25Indexer(tmp_path).build(chunks=chunks, force=True)
        with open(tmp_path / "bm25_combined.pkl", "rb") as f:
            p = pickle.load(f)
        assert p["size"] == len(chunks)

    def test_chunk_ids_in_order(self, chunks, tmp_path):
        BM25Indexer(tmp_path).build(chunks=chunks, force=True)
        with open(tmp_path / "bm25_combined.pkl", "rb") as f:
            p = pickle.load(f)
        assert p["chunk_ids"] == [c.chunk_id for c in chunks]

    def test_ring2_sub_index_correct_size(self, chunks, tmp_path):
        paths = BM25Indexer(tmp_path).build(chunks=chunks, force=True)
        with open(paths["ring_2"], "rb") as f:
            p = pickle.load(f)
        assert p["size"] == 2   # c1 and c3 are ring 2

    def test_idempotent_build(self, chunks, tmp_path):
        idx = BM25Indexer(tmp_path)
        idx.build(chunks=chunks, force=True)
        mtime1 = (tmp_path / "bm25_combined.pkl").stat().st_mtime
        idx.build(chunks=chunks, force=False)   # should skip
        assert (tmp_path / "bm25_combined.pkl").stat().st_mtime == mtime1


# ── BM25 Loader ───────────────────────────────────────────────────────────
class TestBM25Loader:
    def get_loader(self, chunks, tmp_path) -> BM25IndexLoader:
        BM25Indexer(tmp_path).build(chunks=chunks, force=True)
        return BM25IndexLoader(tmp_path).load_all()

    def test_ppf_query_top_result(self, chunks, tmp_path):
        loader  = self.get_loader(chunks, tmp_path)
        results = loader.search("PPF interest rate", top_k=5)
        assert results[0][0] == "c1"

    def test_scores_descending(self, chunks, tmp_path):
        loader = self.get_loader(chunks, tmp_path)
        scores = [r[1] for r in loader.search("interest rate", top_k=5)]
        assert scores == sorted(scores, reverse=True)

    def test_ring_filter_restricts(self, chunks, tmp_path):
        loader = self.get_loader(chunks, tmp_path)
        ids = [r[0] for r in loader.search("NPS PPF", top_k=5, ring_filter=[2])]
        assert all(i in ["c1", "c3"] for i in ids)

    def test_no_zero_scores(self, chunks, tmp_path):
        loader = self.get_loader(chunks, tmp_path)
        scores = [r[1] for r in loader.search("PPF interest", top_k=5)]
        assert all(s > 0 for s in scores)


# ── FAISS Indexer ─────────────────────────────────────────────────────────
def rand_embs(n: int, dim: int = 8) -> np.ndarray:
    """Tiny L2-normalised random embeddings (dim=8 for fast tests)."""
    e = np.random.randn(n, dim).astype(np.float32)
    return e / np.linalg.norm(e, axis=1, keepdims=True)


class TestFAISSIndexer:
    def build_index(self, chunks, tmp_path, dim=8):
        embs = rand_embs(len(chunks), dim)
        idx  = FAISSIndexer(tmp_path)
        ip   = tmp_path / "faiss.index"
        mp   = tmp_path / "id_map.json"
        with patch.object(type(idx), "index_path",  new_callable=PropertyMock, return_value=ip), \
             patch.object(type(idx), "id_map_path", new_callable=PropertyMock, return_value=mp):
            idx.build(embs, chunks, force=True)
        return ip, mp, embs

    def test_files_created(self, chunks, tmp_path):
        ip, mp, _ = self.build_index(chunks, tmp_path)
        assert ip.exists() and mp.exists()

    def test_id_map_correct_ids(self, chunks, tmp_path):
        _, mp, _ = self.build_index(chunks, tmp_path)
        with open(mp) as f:
            id_map = json.load(f)
        assert id_map["int_to_chunk_id"] == [c.chunk_id for c in chunks]
        assert id_map["total"] == len(chunks)