# retrieval/models.py
# RankedChunk: the fundamental retrieval output unit.
# All downstream components (reranker, generator, auditor) consume RankedChunks.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from corpus.models import TextChunk


@dataclass
class RankedChunk:
    """
    A TextChunk with retrieval ranking metadata attached.
    Produced by BM25Retriever or DenseRetriever; consumed by all downstream stages.
    """

    chunk: TextChunk                              # full chunk with text + all metadata
    rank: int                                     # 1-based rank in its result list
    score: float                                  # raw retrieval score
    source: Literal["bm25", "faiss", "rrf", "reranker"] = "bm25"

    # ── Convenience pass-throughs to avoid .chunk.xxx everywhere ──────────
    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def chunk_text(self) -> str:
        return self.chunk.chunk_text

    @property
    def doc_title(self) -> str:
        return self.chunk.doc_title

    @property
    def source_url(self) -> str:
        return self.chunk.source_url

    @property
    def governing_body(self) -> str:
        return self.chunk.governing_body

    @property
    def ring(self) -> int:
        return self.chunk.ring

    @property
    def ring_label(self) -> str:
        return self.chunk.ring_label

    @property
    def effective_date(self) -> str:
        return self.chunk.effective_date

    @property
    def circular_ref(self) -> str | None:
        return self.chunk.circular_ref

    def __repr__(self) -> str:
        return (
            f"RankedChunk(rank={self.rank}, score={self.score:.4f}, "
            f"source={self.source!r}, id={self.chunk_id[:8]}...)"
        )


@dataclass
class SubQueryResults:
    """
    Groups BM25 and FAISS results for a single sub-query.
    Passed from HybridRetriever to RRFFusion.
    """

    sub_query: str
    bm25_results:  list[RankedChunk] = field(default_factory=list)
    faiss_results: list[RankedChunk] = field(default_factory=list)

    @property
    def all_ranked_lists(self) -> list[list[RankedChunk]]:
        """Both result lists as a flat list — consumed by RRFFusion."""
        lists = []
        if self.bm25_results:
            lists.append(self.bm25_results)
        if self.faiss_results:
            lists.append(self.faiss_results)
        return lists