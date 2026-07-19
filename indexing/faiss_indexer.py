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