# config/settings.py  — Part 1: Paths & Environment Loading
# Central configuration. Every constant in the project lives here.
# Pydantic BaseSettings auto-loads from .env file.

from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings  # pydantic v2 split package


class Settings(BaseSettings):
    """
    DocuSage application settings.
    All values can be overridden via environment variables or .env file.
    Precedence: env var > .env file > default value defined here.
    """

    # ── HuggingFace ────────────────────────────────────────────────────────
    HF_TOKEN: str = Field(default="", description="HuggingFace access token")

    # ── Root data directory ────────────────────────────────────────────────
    # All derived paths are computed relative to DATA_DIR so that changing
    # DATA_DIR (e.g. to /data on HF Spaces) automatically updates everything.
    DATA_DIR: Path = Field(
        default=Path("./data"),
        description="Root directory for all local data artefacts",
    )

    # ── Derived data paths (computed as properties, not fields) ────────────
    # These are @property methods below, not Pydantic fields, because they
    # are derived from DATA_DIR. Making them fields would require keeping them
    # in sync manually — error-prone.

    # ── Index file paths ───────────────────────────────────────────────────
    # Can be overridden if pre-built indices live elsewhere (e.g. HF Hub).
    BM25_INDEX_PATH: Path = Field(
        default=Path("./data/indices/bm25_index.pkl"),
        description="Path to serialised BM25 index (pickle)",
    )
    FAISS_INDEX_PATH: Path = Field(
        default=Path("./data/indices/faiss_index.index"),
        description="Path to FAISS flat index binary",
    )
    FAISS_ID_MAP_PATH: Path = Field(
        default=Path("./data/indices/faiss_id_map.json"),
        description="JSON mapping FAISS integer IDs to chunk metadata",
    )
    CORPUS_REGISTRY_PATH: Path = Field(
        default=Path("./data/chunks/corpus_registry.json"),
        description="JSON manifest of all ingested documents",
    )

    # ── Hardware ───────────────────────────────────────────────────────────
    DEVICE: str = Field(
        default="auto",
        description="Compute device: 'cuda' | 'cpu' | 'auto'",
    )

    # ── Logging ────────────────────────────────────────────────────────────
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Loguru log level: DEBUG | INFO | WARNING | ERROR",
    )

    # ── Pydantic settings config ───────────────────────────────────────────
    model_config = {
        "env_file": ".env",          # Read from .env if present
        "env_file_encoding": "utf-8",
        "case_sensitive": True,      # Env vars are uppercase by convention
        "extra": "ignore",           # Silently ignore unrecognised env vars
    }

    # ── Derived path properties ────────────────────────────────────────────
    @property
    def RAW_DOCS_DIR(self) -> Path:
        """Directory for downloaded raw source documents."""
        return self.DATA_DIR / "raw"  # e.g. ./data/raw

    @property
    def CHUNKS_DIR(self) -> Path:
        """Directory for processed chunk JSON files."""
        return self.CHUNKS_DIR_override or (self.DATA_DIR / "chunks")

    @property
    def INDEX_DIR(self) -> Path:
        """Directory for index artefacts (BM25 pickle, FAISS binary)."""
        return self.DATA_DIR / "indices"  # e.g. ./data/indices

    # ── Validator: resolve device ──────────────────────────────────────────
    @field_validator("DEVICE", mode="after")
    @classmethod
    def resolve_device(cls, v: str) -> str:
        """Replace 'auto' with the actual available device at runtime."""
        if v == "auto":
            try:
                import torch  # noqa: PLC0415
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return v

# config/settings.py — Part 2: Model Configuration
# Append these fields to the Settings class defined in Part 1.
# Each model name is the exact HuggingFace Hub identifier.

    # ── Embedding model (BGE-M3) ───────────────────────────────────────────
    EMBEDDING_MODEL: str = Field(
        default="BAAI/bge-m3",
        description="HF Hub ID for the bi-encoder embedding model",
    )
    EMBEDDING_DIM: int = Field(
        default=1024,
        description="Output embedding dimension of the embedding model",
    )
    EMBEDDING_MAX_LENGTH: int = Field(
        default=512,
        description="Max tokens per chunk for embedding (matched to chunk size)",
    )
    EMBEDDING_BATCH_SIZE: int = Field(
        default=32,
        description="Batch size for encoding chunks at index time",
    )

    # ── Reranker model (BGE-reranker-v2-m3) ───────────────────────────────
    RERANKER_MODEL: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        description="HF Hub ID for the cross-encoder reranker",
    )

    # ── Query decomposer (Qwen2.5-1.5B-Instruct) ──────────────────────────
    DECOMPOSER_MODEL: str = Field(
        default="Qwen/Qwen2.5-1.5B-Instruct",
        description="HF Hub ID for the query decomposition model",
    )
    DECOMPOSER_MAX_NEW_TOKENS: int = Field(
        default=256,
        description="Max new tokens for decomposer output (JSON array of sub-queries)",
    )
    DECOMPOSER_TEMPERATURE: float = Field(
        default=0.2,
        description="Sampling temperature for decomposer (low = more deterministic)",
    )

    # ── Generator (Qwen2.5-3B-Instruct) ───────────────────────────────────
    GENERATOR_MODEL: str = Field(
        default="Qwen/Qwen2.5-3B-Instruct",
        description="HF Hub ID for the answer generation model",
    )
    GENERATOR_MAX_NEW_TOKENS: int = Field(
        default=512,
        description="Max new tokens for generated answer",
    )
    GENERATOR_TEMPERATURE: float = Field(
        default=0.1,
        description="Near-deterministic: financial answers must be factual",
    )
    GENERATOR_TOP_P: float = Field(
        default=0.9,
        description="Nucleus sampling top-p for generator",
    )
    GENERATOR_REPETITION_PENALTY: float = Field(
        default=1.1,
        description="Penalise token repetition in generated output",
    )
    GENERATOR_LOAD_IN_4BIT: bool = Field(
        default=True,
        description="Load generator in 4-bit NF4 quantisation (saves ~4GB VRAM)",
    )

    # ── Hallucination auditor (NLI DeBERTa) ───────────────────────────────
    AUDITOR_MODEL: str = Field(
        default="cross-encoder/nli-deberta-v3-small",
        description="HF Hub ID for the NLI-based hallucination auditor",
    )
    AUDITOR_ENTAILMENT_THRESHOLD: float = Field(
        default=0.70,
        description="Min entailment probability for a sentence to be 'grounded'",
    )
    AUDITOR_FLAG_THRESHOLD: float = Field(
        default=0.40,
        description="Max entailment probability below which sentence is flagged ⚠️",
    )

# config/settings.py — Part 3: Retrieval Hyperparameters + Utilities
# Append these fields and methods to the Settings class.

    # ── Chunking configuration ─────────────────────────────────────────────
    CHUNK_SIZE: int = Field(
        default=512,
        description="Target chunk size in tokens (matched to EMBEDDING_MAX_LENGTH)",
    )
    CHUNK_OVERLAP: int = Field(
        default=64,
        description="Token overlap between consecutive chunks",
    )
    CHUNK_MIN_SIZE: int = Field(
        default=128,
        description="Minimum chunk size — smaller chunks are merged with next",
    )

    # ── Retrieval hyperparameters ──────────────────────────────────────────
    BM25_TOP_K: int = Field(
        default=20,
        description="Number of BM25 candidates per sub-query",
    )
    FAISS_TOP_K: int = Field(
        default=20,
        description="Number of FAISS candidates per sub-query",
    )
    RRF_K: int = Field(
        default=60,
        description="RRF smoothing constant (Cormack et al. 2009)",
    )
    RERANKER_INPUT_K: int = Field(
        default=20,
        description="Top-N from RRF passed to cross-encoder reranker",
    )
    GENERATOR_CONTEXT_K: int = Field(
        default=5,
        description="Top-N from reranker passed to generator as context",
    )

    # ── Query routing thresholds ───────────────────────────────────────────
    ROUTER_MAX_WORDS_SIMPLE: int = Field(
        default=15,
        description="Queries shorter than this are never decomposed",
    )
    ROUTER_MIN_WORDS_COMPLEX: int = Field(
        default=60,
        description="Queries longer than this are always decomposed",
    )
    DECOMPOSER_MAX_SUB_QUERIES: int = Field(
        default=4,
        description="Maximum number of sub-queries the decomposer can produce",
    )

    # ── Corpus ring taxonomy ───────────────────────────────────────────────
    # This is the canonical definition of the four corpus rings.
    # Stored as a dict: ring_id (int) → ring label (str)
    CORPUS_RINGS: dict[int, str] = Field(
        default={
            1: "Market Investments",
            2: "Govt Schemes",
            3: "Banking & RBI",
            4: "Foreign Investments",
        },
        description="Canonical ring taxonomy for the corpus",
    )

    # ── Gradio UI configuration ────────────────────────────────────────────
    GRADIO_SERVER_PORT: int = Field(
        default=7860,
        description="Port for Gradio server (7860 is HF Spaces default)",
    )
    GRADIO_MAX_QUERY_LENGTH: int = Field(
        default=500,
        description="Maximum character length for user query input",
    )
    GRADIO_SHARE: bool = False   # overridden to True in Colab via env var

    # ── Utility methods ────────────────────────────────────────────────────
    def ensure_dirs(self) -> None:
        """
        Create all required data directories if they do not exist.
        Call once at application startup before loading any indices.
        Safe to call multiple times (exist_ok=True).
        """
        dirs_to_create = [
            self.DATA_DIR,
            self.RAW_DOCS_DIR,
            self.DATA_DIR / "chunks",
            self.INDEX_DIR,
        ]
        for d in dirs_to_create:
            Path(d).mkdir(parents=True, exist_ok=True)  # create if missing

    def ring_label_to_id(self, label: str) -> int | None:
        """Reverse lookup: ring label string → ring integer ID."""
        reverse = {v: k for k, v in self.CORPUS_RINGS.items()}
        return reverse.get(label)  # returns None if label not found

    def get_ring_labels(self) -> list[str]:
        """Return ring labels in ring-ID order (for Gradio CheckboxGroup)."""
        return [self.CORPUS_RINGS[i] for i in sorted(self.CORPUS_RINGS)]
