# generation/reranker.py
# Cross-encoder reranker: BAAI/bge-reranker-v2-m3, top-20 -> top-5.
from __future__ import annotations
from config import get_logger, settings
from retrieval import RankedChunk

logger = get_logger(__name__)


class Reranker:
    '''Cross-encoder reranker. Reduces RERANKER_INPUT_K to GENERATOR_CONTEXT_K.'''

    def __init__(self, model_name: str | None = None,
                 top_k: int = settings.GENERATOR_CONTEXT_K):
        self.model_name = model_name or settings.RERANKER_MODEL
        self.top_k      = top_k
        self._model     = None

    def load(self) -> 'Reranker':
        '''Load BAAI/bge-reranker-v2-m3. ~570MB, cached after first run.'''
        from sentence_transformers import CrossEncoder
        logger.info('Loading reranker', model=self.model_name)
        self._model = CrossEncoder(
            self.model_name,
            max_length=512,
            device=settings.DEVICE,
        )
        logger.info('Reranker ready')
        return self

    def rerank(
        self,
        query: str,
        candidates: list[RankedChunk],
        top_k: int | None = None,
    ) -> list[RankedChunk]:
        '''
        Score each (query, chunk_text) pair; return top_k sorted by score.

        Args:
            query: User query or primary sub-query.
            candidates: Up to RERANKER_INPUT_K RankedChunks from HybridRetriever.
            top_k: Override top_k (defaults to self.top_k).

        Returns:
            List[RankedChunk] source='reranker', sorted by cross-encoder score.
        '''
        if self._model is None:
            raise RuntimeError('Call .load() before .rerank()')
        if not candidates:
            return []
        k = top_k or self.top_k

        pairs  = [(query, chunk.chunk_text) for chunk in candidates]
        scores = self._model.predict(pairs, show_progress_bar=False,
                                     convert_to_numpy=True)
        scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

        reranked = []
        for final_rank, (chunk, score) in enumerate(scored[:k], start=1):
            reranked.append(RankedChunk(
                chunk=chunk.chunk, rank=final_rank,
                score=float(score), source='reranker',
            ))

        logger.debug('Reranking complete', input=len(candidates), output=len(reranked))
        return reranked