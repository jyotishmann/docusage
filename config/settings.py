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
