# corpus/registry.py
# JSON manifest of all ingested TextChunks.
# Authoritative source for chunk metadata at index-build and query time.

from __future__ import annotations
import ujson
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from config import get_logger, settings
from corpus.models import TextChunk

logger = get_logger(__name__)


class CorpusRegistry:
    """
    Persists all TextChunks as a JSON registry file.
    Provides O(1) chunk lookup and ring-filtered access.
    """

    def __init__(self, registry_path: Path | None = None):
        self.registry_path = Path(registry_path or settings.CORPUS_REGISTRY_PATH)
        self._chunks:     list[TextChunk] = []
        self._id_index:   dict[str, TextChunk] = {}
        self._ring_index: dict[int, list[TextChunk]] = {1: [], 2: [], 3: [], 4: []}

    def add_chunks(self, chunks: list[TextChunk]) -> None:
        """Add chunks to the in-memory registry."""
        for chunk in chunks:
            self._chunks.append(chunk)
            self._id_index[chunk.chunk_id] = chunk
            self._ring_index.setdefault(chunk.ring, []).append(chunk)

    def save(self) -> None:
        """Write all chunks to the JSON registry file."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": {
                "built_at": datetime.now(timezone.utc).isoformat(),
                "total_chunks": len(self._chunks),
                "ring_breakdown": {r: len(c) for r, c in self._ring_index.items()},
                "registry_version": "1.0",
            },
            "chunks": [c.model_dump() for c in self._chunks],
        }
        with open(self.registry_path, "w", encoding="utf-8") as f:
            ujson.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info("Registry saved", chunks=len(self._chunks),
                    path=str(self.registry_path))
    
    def get_ring_counts(self) -> dict[int, int]:
        """Return {ring_id: chunk_count} for all rings in the registry."""
        return {ring: len(chunks) for ring, chunks in self._ring_index.items()}
    
    def __len__(self) -> int:
        return len(self._chunks)

    @classmethod
    def load(cls, registry_path: Path | None = None) -> "CorpusRegistry":
        """Load registry from JSON. Raises FileNotFoundError if missing."""
        path = Path(registry_path or settings.CORPUS_REGISTRY_PATH)
        if not path.exists():
            raise FileNotFoundError(
                f"Registry not found: {path}. Run scripts/download_corpus.py first."
            )
        with open(path, "r", encoding="utf-8") as f:
            payload = ujson.load(f)
        reg = cls(registry_path=path)
        reg.add_chunks([TextChunk.model_validate(c) for c in payload["chunks"]])
        logger.info("Registry loaded", chunks=reg.total_chunks,
                    built_at=payload["metadata"].get("built_at"))
        return reg

    # ── Query methods ──────────────────────────────────────────────────────
    def get_by_id(self, chunk_id: str) -> Optional[TextChunk]:
        return self._id_index.get(chunk_id)   # O(1) dict lookup

    def get_by_ring(self, ring_id: int) -> list[TextChunk]:
        return self._ring_index.get(ring_id, [])

    def get_all(self) -> list[TextChunk]:
        return list(self._chunks)

    @property
    def total_chunks(self) -> int:
        return len(self._chunks)

    def validate_against_index_size(self, bm25_size: int, faiss_size: int) -> bool:
        """Verify that index sizes match registry. Called at startup."""
        n = self.total_chunks
        ok = bm25_size == n and faiss_size == n
        if not ok:
            logger.warning("Index size mismatch", registry=n,
                           bm25=bm25_size, faiss=faiss_size)
        return ok