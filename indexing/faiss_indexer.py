# indexing/faiss_indexer.py — Part 1: EmbeddingEncoder (batched BGE-M3 encoding)

from __future__ import annotations

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import get_logger, settings
from corpus.models import TextChunk

logger = get_logger(__name__)


class EmbeddingEncoder:
    """
    Wraps BGE-M3 for corpus encoding at index-build time.
    Separate from query-time DenseRetriever for independent lifecycle management.
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
    ):
        self.model_name = model_name or settings.EMBEDDING_MODEL   # "BAAI/bge-m3"
        self.device     = device     or settings.DEVICE
        self.batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE
        self._model: SentenceTransformer | None = None   # lazy-loaded

    def load(self) -> "EmbeddingEncoder":
        """Load BGE-M3 from HF Hub (cached after first ~570MB download). Returns self."""
        logger.info("Loading embedding model", model=self.model_name, device=self.device)
        self._model = SentenceTransformer(self.model_name, device=self.device)

        # Validate output dimension against settings
        test_vec  = self._model.encode(["dimension check"], normalize_embeddings=True)
        actual_dim = test_vec.shape[1]
        if actual_dim != settings.EMBEDDING_DIM:
            raise ValueError(
                f"Embedding dim mismatch: expected {settings.EMBEDDING_DIM}, "
                f"got {actual_dim} from model '{self.model_name}'"
            )
        logger.info("Embedding model ready", dim=actual_dim)
        return self

    def encode_chunks(self, chunks: list[TextChunk]) -> np.ndarray:
        """
        Encode all chunk texts into L2-normalised float32 vectors.

        Returns:
            np.ndarray shape (n_chunks, 1024), dtype float32, L2-normalised.
        """
        if self._model is None:
            raise RuntimeError("Call .load() before .encode_chunks()")

        texts = [c.chunk_text for c in chunks]
        n     = len(texts)
        logger.info("Encoding corpus chunks", count=n, batch_size=self.batch_size)

        all_embs: list[np.ndarray] = []
        n_batches = (n + self.batch_size - 1) // self.batch_size

        for start in tqdm(range(0, n, self.batch_size), desc="Encoding", unit="batch",
                          total=n_batches):
            batch = texts[start : start + self.batch_size]
            embs  = self._model.encode(
                batch,
                normalize_embeddings=True,   # L2-normalise: dot product = cosine sim
                show_progress_bar=False,
                convert_to_numpy=True,
                batch_size=len(batch),
            )
            all_embs.append(embs.astype(np.float32))   # float32 in batch loop

        embeddings = np.vstack(all_embs)   # (n_chunks, 1024) float32

        # Sanity check: unit norms should all be ~1.0
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            logger.warning("Embeddings not unit-normalised",
                           min_norm=float(norms.min()), max_norm=float(norms.max()))

        logger.info("Encoding complete", shape=embeddings.shape, dtype=str(embeddings.dtype))
        return embeddings
    
# indexing/faiss_indexer.py — Part 2: FAISSIndexer + FAISSIndexLoader (append)
import json
import faiss


class FAISSIndexer:
    """Builds and saves faiss.IndexFlatIP + id_map JSON from embeddings."""

    def __init__(self, index_dir: Path | None = None):
        self.index_dir = Path(index_dir or settings.INDEX_DIR)
        self.index_dir.mkdir(parents=True, exist_ok=True)

    @property
    def index_path(self) -> Path:
        return Path(settings.FAISS_INDEX_PATH)

    @property
    def id_map_path(self) -> Path:
        return Path(settings.FAISS_ID_MAP_PATH)

    def build(
        self,
        embeddings: np.ndarray,
        chunks: list[TextChunk],
        force: bool = False,
    ) -> tuple[Path, Path]:
        """Build IndexFlatIP; save binary + id_map. Skips if both exist."""
        if not force and self.index_path.exists() and self.id_map_path.exists():
            logger.info("FAISS index exists — skipping (use force=True to rebuild)")
            return self.index_path, self.id_map_path

        assert len(embeddings) == len(chunks), "Embedding/chunk count mismatch"
        assert embeddings.dtype == np.float32,  "FAISS requires float32 embeddings"

        n, dim = embeddings.shape
        logger.info("Building FAISS IndexFlatIP", n=n, dim=dim)

        index = faiss.IndexFlatIP(dim)   # brute-force inner product (cosine with unit vecs)
        index.add(embeddings)             # O(n) add
        assert index.ntotal == n

        # Build id_map: FAISS integer IDs <-> chunk_id strings
        int_to_chunk_id = [c.chunk_id for c in chunks]
        id_map = {
            "int_to_chunk_id": int_to_chunk_id,
            "chunk_id_to_int": {cid: i for i, cid in enumerate(int_to_chunk_id)},
            "total": n, "dim": dim,
        }

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.index_path))     # binary format
        with open(self.id_map_path, "w", encoding="utf-8") as f:
            json.dump(id_map, f, indent=2)

        logger.info("FAISS index saved", ntotal=n,
                    index=str(self.index_path), id_map=str(self.id_map_path))
        return self.index_path, self.id_map_path


class FAISSIndexLoader:
    """Loads FAISS index + id_map at startup; exposes search()."""

    def __init__(
        self,
        index_path: Path | None = None,
        id_map_path: Path | None = None,
    ):
        self.index_path  = Path(index_path  or settings.FAISS_INDEX_PATH)
        self.id_map_path = Path(id_map_path or settings.FAISS_ID_MAP_PATH)
        self._index:           faiss.IndexFlatIP | None = None
        self._int_to_chunk_id: list[str]                = []
        self._chunk_id_to_int: dict[str, int]           = {}

    def load(self) -> "FAISSIndexLoader":
        """Load binary index + id_map. Returns self for chaining."""
        for p in (self.index_path, self.id_map_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"FAISS artefact missing: {p}. Run build_index.py."
                )
        self._index = faiss.read_index(str(self.index_path))
        with open(self.id_map_path, encoding="utf-8") as f:
            id_map = json.load(f)
        self._int_to_chunk_id = id_map["int_to_chunk_id"]
        self._chunk_id_to_int = id_map["chunk_id_to_int"]
        logger.info("FAISS index loaded", ntotal=self._index.ntotal, dim=self._index.d)
        return self

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = settings.FAISS_TOP_K,
        ring_filter_ints: list[int] | None = None,
    ) -> list[tuple[str, float]]:
        """
        Search FAISS. Returns [(chunk_id, cosine_score)] sorted descending.
        ring_filter_ints: FAISS integer IDs to restrict search to.
        """
        if self._index is None:
            raise RuntimeError("Call .load() before .search()")

        if ring_filter_ints:
            scores, indices = self._filtered_search(query_vector, top_k, ring_filter_ints)
        else:
            scores, indices = self._index.search(query_vector, top_k)

        results: list[tuple[str, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:   # FAISS pads with -1 for unfilled slots
                continue
            results.append((self._int_to_chunk_id[int(idx)], float(score)))
        return results

    def _filtered_search(
        self, qv: np.ndarray, top_k: int, allowed: list[int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fetch top-(k*10) then post-filter to allowed FAISS integer IDs."""
        allowed_set = set(allowed)
        fetch_k = min(top_k * 10, self._index.ntotal)
        sc_all, ix_all = self._index.search(qv, fetch_k)

        f_sc, f_ix = [], []
        for idx, sc in zip(ix_all[0], sc_all[0]):
            if idx in allowed_set:
                f_ix.append(idx)
                f_sc.append(sc)
            if len(f_ix) == top_k:
                break

        # Pad to top_k with -1/0.0 (FAISS convention for missing results)
        while len(f_ix) < top_k:
            f_ix.append(-1)
            f_sc.append(0.0)

        return (np.array([f_sc], dtype=np.float32),
                np.array([f_ix], dtype=np.int64))

    def get_int_ids_for_ring(self, ring_id: int, registry) -> list[int]:
        """FAISS integer IDs for all chunks in ring_id (used for ring filter)."""
        return [
            i for c in registry.get_by_ring(ring_id)
            if (i := self._chunk_id_to_int.get(c.chunk_id)) is not None
        ]

    @property
    def total_vectors(self) -> int:
        return self._index.ntotal if self._index else 0