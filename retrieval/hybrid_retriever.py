# retrieval/hybrid_retriever.py
# Orchestrates BM25 + Dense retrieval + RRF fusion for a list of sub-queries.

from __future__ import annotations

from config import get_logger, settings
from retrieval.bm25_retriever import BM25Retriever
from retrieval.dense_retriever import DenseRetriever
from retrieval.models import RankedChunk, SubQueryResults
from retrieval.rrf_fusion import RRFFusion

logger = get_logger(__name__)


class HybridRetriever:
    """
    Top-level retrieval orchestrator.
    Input: list of sub-queries + optional ring filter.
    Output: top-N RankedChunks fused via RRF, ready for the cross-encoder reranker.
    """

    def __init__(
        self,
        bm25_retriever:  BM25Retriever,
        dense_retriever: DenseRetriever,
        rrf_fusion:      RRFFusion | None = None,
        bm25_top_k:      int = settings.BM25_TOP_K,
        faiss_top_k:     int = settings.FAISS_TOP_K,
        reranker_input_k: int = settings.RERANKER_INPUT_K,
    ):
        self.bm25     = bm25_retriever
        self.dense    = dense_retriever
        self.rrf      = rrf_fusion or RRFFusion()
        self.bm25_k   = bm25_top_k
        self.faiss_k  = faiss_top_k
        self.rerank_k = reranker_input_k

    def retrieve(
        self,
        sub_queries: list[str],
        ring_filter: list[str] | None = None,
    ) -> list[RankedChunk]:
        """
        Run hybrid retrieval for a list of sub-queries.

        For each sub-query:
          1. BM25 retrieval (exact-match sparse)
          2. Dense retrieval (semantic FAISS)
          → SubQueryResults container

        Then: RRF fusion across all sub-query result lists.

        Args:
            sub_queries: List of 1-4 focused sub-query strings.
            ring_filter: Optional ring label filter (passed through to both retrievers).

        Returns:
            Top RERANKER_INPUT_K (default 20) RankedChunks sorted by RRF score.
        """
        if not sub_queries:
            logger.warning("HybridRetriever called with empty sub_queries list")
            return []

        logger.info(
            "Hybrid retrieval starting",
            sub_queries=len(sub_queries),
            ring_filter=ring_filter,
        )

        # ── Per-sub-query retrieval ────────────────────────────────────────
        all_sub_results: list[SubQueryResults] = []

        for i, sub_query in enumerate(sub_queries, start=1):
            logger.debug(f"Sub-query {i}/{len(sub_queries)}", query=sub_query[:60])

            # BM25 retrieval
            bm25_results = self.bm25.retrieve(
                query=sub_query,
                top_k=self.bm25_k,
                ring_filter=ring_filter,
            )

            # Dense (FAISS) retrieval
            faiss_results = self.dense.retrieve(
                query=sub_query,
                top_k=self.faiss_k,
                ring_filter=ring_filter,
            )

            sub_result = SubQueryResults(
                sub_query=sub_query,
                bm25_results=bm25_results,
                faiss_results=faiss_results,
            )
            all_sub_results.append(sub_result)

            logger.debug(
                f"Sub-query {i} results",
                bm25=len(bm25_results),
                faiss=len(faiss_results),
            )

        # ── RRF fusion across all sub-query result lists ──────────────────
        fused = self.rrf.fuse_sub_query_results(
            sub_query_results=all_sub_results,
            top_n=self.rerank_k,
        )

        logger.info(
            "Hybrid retrieval complete",
            sub_queries=len(sub_queries),
            fused_candidates=len(fused),
            top_rrf_score=fused[0].score if fused else 0,
        )
        return fused

    def retrieve_single(
        self,
        query: str,
        ring_filter: list[str] | None = None,
    ) -> list[RankedChunk]:
        """
        Convenience: retrieve for a single query (no decomposition).
        Wraps the query in a one-element list and calls retrieve().
        """
        return self.retrieve(sub_queries=[query], ring_filter=ring_filter)