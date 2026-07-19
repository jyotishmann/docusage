# retrieval/dense_retriever.py
# Encodes query with BGE-M3 and searches FAISS for semantic retrieval.

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from config import get_logger, settings
from corpus import CorpusRegistry
from indexing import FAISSIndexLoader
from retrieval.models import RankedChunk

logger = get_logger(__name__)


class DenseRetriever:
    """
    Query-time dense retrieval via BGE-M3 + FAISS.
    Encodes query → L2-normalised vector → cosine search in FAISS index.
    """

    def __init__(
        self,
        faiss_loader: FAISSIndexLoader,
        registry: CorpusRegistry,
        model: SentenceTransformer,
    ):
        self.faiss   = faiss_loader   # pre-loaded FAISSIndexLoader
        self.registry = registry      # CorpusRegistry for chunk_id → TextChunk
        self.model    = model         # injected BGE-M3 model (shared with pipeline)

    def retrieve(
        self,
        query: str,
        top_k: int = settings.FAISS_TOP_K,
        ring_filter: list[str] | None = None,
    ) -> list[RankedChunk]:
        """
        Run dense retrieval for a single query string.

        Args:
            query: Raw query string.
            top_k: Number of results to return.
            ring_filter: List of ring label strings. None/empty → all rings.

        Returns:
            List of RankedChunk sorted by cosine similarity descending.
        """
        # ── Encode query ──────────────────────────────────────────────────
        query_vector = self._encode_query(query)   # (1, 1024) float32

        # ── Build ring filter int IDs ─────────────────────────────────────
        ring_filter_ints = self._build_ring_filter_ints(ring_filter)

        # ── FAISS search → [(chunk_id, score)] ────────────────────────────
        raw_results = self.faiss.search(
            query_vector=query_vector,
            top_k=top_k,
            ring_filter_ints=ring_filter_ints if ring_filter_ints else None,
        )

        if not raw_results:
            logger.debug("Dense search returned no results", query=query[:50])
            return []

        # ── Hydrate chunk_ids to RankedChunks ─────────────────────────────
        ranked_chunks: list[RankedChunk] = []
        for rank, (chunk_id, score) in enumerate(raw_results, start=1):
            text_chunk = self.registry.get_by_id(chunk_id)
            if text_chunk is None:
                logger.warning(
                    "chunk_id from FAISS not found in registry (stale index?)",
                    chunk_id=chunk_id[:8],
                )
                continue
            ranked_chunks.append(
                RankedChunk(
                    chunk=text_chunk,
                    rank=rank,
                    score=float(score),
                    source="faiss",
                )
            )

        logger.debug(
            "Dense retrieve complete",
            query=query[:50],
            results=len(ranked_chunks),
            top_score=ranked_chunks[0].score if ranked_chunks else 0,
        )
        return ranked_chunks

    def _encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query string to a (1, dim) float32 L2-normalised vector.
        Uses the same normalize_embeddings=True as corpus encoding at index time.
        """
        embedding = self.model.encode(
            [query],                      # list input even for single query
            normalize_embeddings=True,    # must match index-build normalisation
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embedding.astype(np.float32)   # (1, 1024) float32

    def _build_ring_filter_ints(
        self, ring_filter: list[str] | None
    ) -> list[int]:
        """
        Convert ring label strings to FAISS integer IDs for the ring's chunks.
        Unions integer ID lists when multiple rings are requested.
        """
        if not ring_filter:
            return []

        all_ints: list[int] = []
        for label in ring_filter:
            ring_id = settings.ring_label_to_id(label)
            if ring_id is None:
                logger.warning("Unknown ring label in dense filter", label=label)
                continue
            ring_ints = self.faiss.get_int_ids_for_ring(ring_id, self.registry)
            all_ints.extend(ring_ints)

        # Deduplicate (a chunk can't belong to two rings, but be safe)
        return list(set(all_ints))