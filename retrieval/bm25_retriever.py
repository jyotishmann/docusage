# retrieval/bm25_retriever.py
# Wraps BM25IndexLoader to produce RankedChunk outputs with full TextChunk metadata.

from __future__ import annotations

from config import get_logger, settings
from corpus import CorpusRegistry
from indexing import BM25IndexLoader
from retrieval.models import RankedChunk


logger = get_logger(__name__)


class BM25Retriever:
    """
    Query-time BM25 retrieval.
    Converts ring label strings → ring IDs, searches BM25, hydrates chunk metadata.
    """

    def __init__(
        self,
        index_loader: BM25IndexLoader,
        registry: CorpusRegistry,
    ):
        self.loader   = index_loader   # pre-loaded BM25IndexLoader (startup-loaded)
        self.registry = registry       # CorpusRegistry for chunk_id → TextChunk lookup

    def retrieve(
        self,
        query: str,
        top_k: int = settings.BM25_TOP_K,
        ring_filter: list[str] | None = None,
    ) -> list[RankedChunk]:
        """
        Run BM25 retrieval for a single query string.

        Args:
            query: Raw query string (tokenised internally by BM25IndexLoader).
            top_k: Number of results to return.
            ring_filter: List of ring label strings (e.g. ["Govt Schemes"]).
                         None or empty → search all rings (combined index).

        Returns:
            List of RankedChunk, sorted by BM25 score descending.
        """
        # ── Convert ring labels to ring IDs ───────────────────────────────
        ring_ids = self._labels_to_ids(ring_filter)

        # ── BM25 search → [(chunk_id, score)] ────────────────────────────
        raw_results = self.loader.search(
            query=query,
            top_k=top_k,
            ring_filter=ring_ids if ring_ids else None,
        )

        if not raw_results:
            logger.debug("BM25 search returned no results", query=query[:50])
            return []

        # ── Hydrate chunk_ids to RankedChunks ─────────────────────────────
        ranked_chunks: list[RankedChunk] = []
        for rank, (chunk_id, score) in enumerate(raw_results, start=1):
            text_chunk = self.registry.get_by_id(chunk_id)
            if text_chunk is None:
                # Defensive: chunk in index but not in registry — stale index
                logger.warning(
                    "chunk_id from BM25 not found in registry (stale index?)",
                    chunk_id=chunk_id[:8],
                )
                continue
            ranked_chunks.append(
                RankedChunk(
                    chunk=text_chunk,
                    rank=rank,
                    score=score,
                    source="bm25",
                )
            )

        logger.debug(
            "BM25 retrieve complete",
            query=query[:50],
            results=len(ranked_chunks),
            top_score=ranked_chunks[0].score if ranked_chunks else 0,
        )
        return ranked_chunks

    def _labels_to_ids(self, ring_filter: list[str] | None) -> list[int]:
        """
        Convert ring label strings to integer ring IDs.
        Unknown labels are warned and skipped (not raised — graceful degradation).
        Returns empty list if ring_filter is None or empty.
        """
        if not ring_filter:
            return []
        ids: list[int] = []
        for label in ring_filter:
            ring_id = settings.ring_label_to_id(label)
            if ring_id is not None:
                ids.append(ring_id)
            else:
                logger.warning("Unknown ring label — skipping", label=label)
        return ids