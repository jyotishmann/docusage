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
    
# indexing/bm25_indexer.py — Part 2: BM25IndexLoader (append after Part 1)
import numpy as np


class BM25IndexLoader:
    """
    Loads pre-built BM25 pickle indices at startup; provides search().
    All indices kept in memory after load_all() — no per-query disk I/O.
    """

    def __init__(self, index_dir: Path | None = None):
        self.index_dir = Path(index_dir or settings.INDEX_DIR)
        self._indices: dict[str, dict] = {}

    def load_all(self) -> "BM25IndexLoader":
        """Load combined + all ring sub-indices. Call once at startup. Returns self."""
        combined_path = self.index_dir / "bm25_combined.pkl"
        if not combined_path.exists():
            raise FileNotFoundError(
                f"BM25 combined index missing: {combined_path}. Run build_index.py."
            )
        self._indices["combined"] = self._load_pkl(combined_path)
        logger.info("BM25 combined loaded", size=self._indices["combined"]["size"])

        for ring_id in settings.CORPUS_RINGS:
            path = self.index_dir / f"bm25_ring_{ring_id}.pkl"
            if path.exists():
                self._indices[f"ring_{ring_id}"] = self._load_pkl(path)
                logger.info("BM25 ring loaded", ring=ring_id,
                            size=self._indices[f"ring_{ring_id}"]["size"])
        return self

    def search(
        self,
        query: str,
        top_k: int = settings.BM25_TOP_K,
        ring_filter: list[int] | None = None,
    ) -> list[tuple[str, float]]:
        """
        Search BM25. Returns [(chunk_id, score)] sorted descending.
        ring_filter: list of ring IDs — single ring uses per-ring sub-index.
        """
        payload   = self._select_index(ring_filter)
        bm25      = payload["bm25"]
        chunk_ids = payload["chunk_ids"]

        tokens = tokenise(query)
        if not tokens:
            return []

        scores: np.ndarray = bm25.get_scores(tokens)  # shape: (n_docs,)

        # O(n) top-k via argpartition, then sort only the partition
        n = len(scores)
        if top_k >= n:
            top_idx = np.argsort(scores)[::-1]
        else:
            part    = np.argpartition(scores, -top_k)[-top_k:]
            top_idx = part[np.argsort(scores[part])[::-1]]

        results = [
            (chunk_ids[int(i)], float(scores[i]))
            for i in top_idx
            if scores[i] > 0  # BM25 score=0 → no query term matched
        ]
        logger.debug("BM25 search", query=query[:50], results=len(results))
        return results

    def _select_index(self, ring_filter: list[int] | None) -> dict:
        """Single ring → ring sub-index (correct IDF). Multi/no filter → combined."""
        if ring_filter and len(ring_filter) == 1:
            key = f"ring_{ring_filter[0]}"
            if key in self._indices:
                return self._indices[key]
            logger.warning("Ring sub-index unavailable, falling back", ring=ring_filter[0])
        return self._indices["combined"]

    @staticmethod
    def _load_pkl(path: Path) -> dict:
        with open(path, "rb") as f:
            return pickle.load(f)

    @property
    def combined_size(self) -> int:
        return self._indices.get("combined", {}).get("size", 0)