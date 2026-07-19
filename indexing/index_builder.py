# indexing/index_builder.py
# Orchestrates BM25 + FAISS index construction from CorpusRegistry.

from __future__ import annotations
import pickle
from pathlib import Path
import faiss as _faiss
from config import get_logger, settings
from corpus import CorpusRegistry
from indexing.bm25_indexer import BM25Indexer
from indexing.faiss_indexer import EmbeddingEncoder, FAISSIndexer

logger = get_logger(__name__)


class IndexBuilder:
    """Orchestrates BM25 + FAISS build with post-build validation."""

    def __init__(
        self,
        registry: CorpusRegistry | None = None,
        index_dir: Path | None = None,
    ):
        self.registry  = registry
        self.index_dir = Path(index_dir or settings.INDEX_DIR)

    def build(self, force: bool = False) -> dict:
        """
        Full pipeline: load registry → BM25 → encode → FAISS → validate.
        Returns summary dict with paths and sizes.
        """
        if self.registry is None:
            self.registry = CorpusRegistry.load()

        chunks = self.registry.get_all()
        n      = len(chunks)
        logger.info("Index build starting", total_chunks=n, force=force)

        if n == 0:
            raise ValueError("Empty registry — run download_corpus.py first.")

        summary: dict = {
            "total_chunks":    n,
            "ring_breakdown":  self.registry.get_ring_counts(),
            "bm25_paths":      {},
            "faiss_index_path": None,
            "faiss_id_map":     None,
        }

        # Step 1: BM25 (~5s for 1500 chunks)
        logger.info("Step 1/2: Building BM25 indices...")
        bm25_paths = BM25Indexer(self.index_dir).build(chunks=chunks, force=force)
        summary["bm25_paths"] = {k: str(v) for k, v in bm25_paths.items()}

        # Step 2: FAISS (~7s T4 / ~30s CPU after model cached)
        logger.info("Step 2/2: Building FAISS index...")
        encoder = EmbeddingEncoder()
        encoder.load()
        embeddings = encoder.encode_chunks(chunks)
        idx_path, map_path = FAISSIndexer(self.index_dir).build(
            embeddings=embeddings, chunks=chunks, force=force,
        )
        summary["faiss_index_path"] = str(idx_path)
        summary["faiss_id_map"]     = str(map_path)

        # Validate both index sizes == registry chunk count
        self._validate(summary, n)
        logger.info("Index build complete")
        return summary

    def _validate(self, summary: dict, expected: int) -> None:
        """Warn if index sizes don't match registry chunk count."""
        combined = summary["bm25_paths"].get("combined")
        if combined and Path(combined).exists():
            with open(combined, "rb") as f:
                size = pickle.load(f).get("size", -1)
            if size != expected:
                logger.warning("BM25 size mismatch", expected=expected, actual=size)
            else:
                logger.info("BM25 validated", size=size)

        faiss_path = summary.get("faiss_index_path")
        if faiss_path and Path(faiss_path).exists():
            ntotal = _faiss.read_index(faiss_path).ntotal
            if ntotal != expected:
                logger.warning("FAISS ntotal mismatch", expected=expected, actual=ntotal)
            else:
                logger.info("FAISS validated", ntotal=ntotal)