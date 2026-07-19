# indexing/bm25_indexer.py — Part 1: Tokeniser and BM25 index construction
# Builds BM25Okapi indices: one combined + four per-ring sub-indices.

from __future__ import annotations

import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from config import get_logger, settings
from corpus.models import TextChunk

logger = get_logger(__name__)


def tokenise(text: str) -> list[str]:
    """
    Lowercase, split on non-alphanumeric, keep tokens with len >= 2.
    NO stemming: Indian financial acronyms (ELSS, SCSS, 80CCD) must match exactly.
    """
    text = text.lower()
    raw  = re.split(r"[^a-z0-9%\.]+", text)
    return [t for t in raw if len(t) >= 2]


def augment_with_tags(chunk: TextChunk) -> str:
    """Append topic_tags to chunk text before tokenisation (BM25 relevance boost)."""
    if not chunk.topic_tags:
        return chunk.chunk_text
    return chunk.chunk_text + " " + " ".join(chunk.topic_tags)


class BM25Indexer:
    """
    Builds and persists BM25Okapi indices for the DocuSage corpus.
    One combined index + four per-ring sub-indices.
    """

    def __init__(self, index_dir: Path | None = None):
        self.index_dir = Path(index_dir or settings.INDEX_DIR)
        self.index_dir.mkdir(parents=True, exist_ok=True)

    @property
    def combined_index_path(self) -> Path:
        return self.index_dir / "bm25_combined.pkl"

    def ring_index_path(self, ring_id: int) -> Path:
        return self.index_dir / f"bm25_ring_{ring_id}.pkl"

    def build(self, chunks: list[TextChunk], force: bool = False) -> dict[str, Path]:
        """
        Build combined + per-ring BM25 indices. Skips if exist (unless force=True).
        Returns dict of {label: saved_path}.
        """
        if not force and self.combined_index_path.exists():
            logger.info("BM25 indices exist — skipping (use force=True to rebuild)")
            return self._existing_paths()

        logger.info("Building BM25 indices", total_chunks=len(chunks))

        # Combined index over all chunks
        combined_path = self._build_single_index(
            chunks, self.combined_index_path, "combined"
        )

        # Per-ring sub-indices
        paths: dict[str, Path] = {"combined": combined_path}
        for ring_id in settings.CORPUS_RINGS:
            ring_chunks = [c for c in chunks if c.ring == ring_id]
            if not ring_chunks:
                continue
            p = self._build_single_index(
                ring_chunks, self.ring_index_path(ring_id), f"ring_{ring_id}"
            )
            paths[f"ring_{ring_id}"] = p

        logger.info("BM25 indexing complete", indices=list(paths.keys()))
        return paths

    def _build_single_index(
        self, chunks: list[TextChunk], path: Path, label: str
    ) -> Path:
        """Tokenise chunks, build BM25Okapi, pickle with chunk_ids mapping."""
        tokenised_corpus = [tokenise(augment_with_tags(c)) for c in chunks]
        chunk_ids        = [c.chunk_id for c in chunks]
        bm25 = BM25Okapi(tokenised_corpus)  # k1=1.5, b=0.75 (standard params)
        payload = {
            "bm25":      bm25,
            "chunk_ids": chunk_ids,  # parallel list: chunk_ids[i] <-> bm25 position i
            "label":     label,
            "size":      len(chunks),
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("BM25 index saved", label=label, chunks=len(chunks), path=str(path))
        return path

    def _existing_paths(self) -> dict[str, Path]:
        """Return paths of already-built indices (no rebuild needed)."""
        paths = {"combined": self.combined_index_path}
        for ring_id in settings.CORPUS_RINGS:
            p = self.ring_index_path(ring_id)
            if p.exists():
                paths[f"ring_{ring_id}"] = p
        return paths