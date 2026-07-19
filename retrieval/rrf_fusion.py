# retrieval/rrf_fusion.py
# Reciprocal Rank Fusion: merges multiple ranked lists into one.
# Implementation of Cormack, Clarke & Buettcher (2009).

from __future__ import annotations

from collections import defaultdict

from config import get_logger, settings
from retrieval.models import RankedChunk, SubQueryResults

logger = get_logger(__name__)


class RRFFusion:
    """
    Merges multiple ranked lists via Reciprocal Rank Fusion.
    Input: list of list[RankedChunk] (one per retrieval method per sub-query).
    Output: single list[RankedChunk] sorted by RRF score descending.
    """

    def __init__(self, k: int = settings.RRF_K):
        self.k = k   # smoothing constant — default 60 (Cormack et al. 2009)

    def fuse(
        self,
        ranked_lists: list[list[RankedChunk]],
        top_n: int = settings.RERANKER_INPUT_K,
    ) -> list[RankedChunk]:
        """
        Merge multiple ranked lists into a single RRF-ranked list.

        Args:
            ranked_lists: Each element is one ranked list (e.g. BM25 results
                          for sub_query_1, FAISS results for sub_query_1, ...).
            top_n: Number of top chunks to return after fusion.

        Returns:
            List of RankedChunk with source="rrf", sorted by RRF score descending.
            Length <= top_n.
        """
        if not ranked_lists:
            return []

        # Filter out empty lists
        non_empty = [rl for rl in ranked_lists if rl]
        if not non_empty:
            return []

        # ── Accumulate RRF scores per chunk_id ───────────────────────────
        # {chunk_id: total_rrf_score}
        rrf_scores: dict[str, float] = defaultdict(float)
        # {chunk_id: RankedChunk} — keep the last seen instance for metadata
        chunk_registry: dict[str, RankedChunk] = {}

        for ranked_list in non_empty:
            for ranked_chunk in ranked_list:
                cid = ranked_chunk.chunk_id
                rank = ranked_chunk.rank  # 1-based

                # RRF contribution from this list
                contribution = 1.0 / (self.k + rank)
                rrf_scores[cid] += contribution

                # Store chunk (overwrite is fine — all copies have same metadata)
                chunk_registry[cid] = ranked_chunk

        # ── Sort by total RRF score descending ────────────────────────────
        sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)
        top_ids    = sorted_ids[:top_n]

        # ── Build output RankedChunks ─────────────────────────────────────
        fused: list[RankedChunk] = []
        for final_rank, cid in enumerate(top_ids, start=1):
            original = chunk_registry[cid]
            fused.append(
                RankedChunk(
                    chunk=original.chunk,     # preserve original TextChunk
                    rank=final_rank,          # rank in the fused output
                    score=rrf_scores[cid],    # total RRF score
                    source="rrf",
                )
            )

        logger.debug(
            "RRF fusion complete",
            input_lists=len(non_empty),
            unique_chunks=len(rrf_scores),
            returned=len(fused),
            top_score=fused[0].score if fused else 0,
        )
        return fused

    def fuse_sub_query_results(
        self,
        sub_query_results: list[SubQueryResults],
        top_n: int = settings.RERANKER_INPUT_K,
    ) -> list[RankedChunk]:
        """
        Convenience method: fuse results from multiple SubQueryResults objects.
        Unpacks all BM25 and FAISS result lists into a flat list and calls fuse().
        """
        all_lists: list[list[RankedChunk]] = []
        for sqr in sub_query_results:
            all_lists.extend(sqr.all_ranked_lists)   # adds bm25 + faiss lists

        logger.debug(
            "RRF fusing sub-query results",
            sub_queries=len(sub_query_results),
            total_lists=len(all_lists),
        )
        return self.fuse(all_lists, top_n=top_n)

    @staticmethod
    def explain_scores(
        ranked_lists: list[list[RankedChunk]],
        chunk_id: str,
        k: int = 60,
    ) -> dict:
        """
        Debugging utility: show RRF score breakdown for a specific chunk.
        Returns {list_index: {rank, contribution, source}} for each list where
        the chunk appears.
        """
        explanation: dict = {"chunk_id": chunk_id, "k": k, "contributions": []}
        total = 0.0
        for i, ranked_list in enumerate(ranked_lists):
            for rc in ranked_list:
                if rc.chunk_id == chunk_id:
                    contrib = 1.0 / (k + rc.rank)
                    total  += contrib
                    explanation["contributions"].append({
                        "list_index": i,
                        "source":     rc.source,
                        "rank":       rc.rank,
                        "contribution": round(contrib, 6),
                    })
        explanation["total_rrf_score"] = round(total, 6)
        return explanation