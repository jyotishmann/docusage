# retrieval/__init__.py
# Public API for the retrieval package.
# Primary consumer: pipeline/rag_pipeline.py

from retrieval.models         import RankedChunk, SubQueryResults
from retrieval.bm25_retriever import BM25Retriever
from retrieval.dense_retriever import DenseRetriever
from retrieval.rrf_fusion     import RRFFusion
from retrieval.hybrid_retriever import HybridRetriever

__all__ = [
    "RankedChunk",        # primary data model — used across ALL downstream stages
    "SubQueryResults",    # intermediate container for HybridRetriever → RRFFusion
    "BM25Retriever",
    "DenseRetriever",
    "RRFFusion",
    "HybridRetriever",    # primary entry point called by RAGPipeline
]