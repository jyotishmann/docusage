# corpus/models.py
# Pydantic v2 data models for DocuSage corpus objects.
# SourceDocument: one downloaded file.
# TextChunk: one retrievable unit (many per SourceDocument).

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class SourceDocument(BaseModel):
    """
    A single downloaded source document before chunking.
    One SourceDocument → N TextChunks after chunking.
    """

    # ── Identity ───────────────────────────────────────────────────────────
    doc_id: str = Field(description="Unique document identifier (UUID4 string)")
    title: str = Field(description="Human-readable document title for citation display")
    source_url: str = Field(description="Canonical URL of the source document")

    # ── Corpus classification ──────────────────────────────────────────────
    governing_body: str = Field(
        description="Issuing authority: 'RBI' | 'SEBI' | 'PFRDA' | 'Zerodha' | ..."
    )
    ring: int = Field(
        ge=1, le=4,
        description="Corpus ring: 1=Market 2=GovtSchemes 3=Banking 4=Foreign"
    )
    ring_label: str = Field(
        description="Human-readable ring label matching settings.CORPUS_RINGS"
    )

    # ── Temporal metadata ──────────────────────────────────────────────────
    effective_date: str = Field(
        default="unknown",
        description="ISO date string when document became effective (YYYY-MM-DD)"
    )
    superseded_by: Optional[str] = Field(
        default=None,
        description="doc_id of the document that supersedes this one, if any"
    )

    # ── Reference metadata ─────────────────────────────────────────────────
    circular_ref: Optional[str] = Field(
        default=None,
        description="RBI/SEBI circular reference number, e.g. RBI/2023-24/73"
    )

    # ── Content & storage ─────────────────────────────────────────────────
    raw_text: str = Field(
        default="",
        description="Full extracted text of the document (populated after parsing)"
    )
    file_path: Optional[str] = Field(
        default=None,
        description="Absolute path to the downloaded raw file on disk"
    )
    file_format: str = Field(
        default="pdf",
        description="Source format: 'pdf' | 'html' | 'html_module' | 'txt'"
    )

    @classmethod
    def make_doc_id(cls, source_url: str) -> str:
        """Deterministic UUID5 from URL — same URL always gives same doc_id."""
        return str(uuid.uuid5(uuid.NAMESPACE_URL, source_url))


class TextChunk(BaseModel):
    """
    A single retrievable text chunk, output of the chunking pipeline.
    This is the fundamental unit stored in BM25 and FAISS indices.
    """

    # ── Identity ───────────────────────────────────────────────────────────
    chunk_id: str = Field(description="Deterministic UUID5 from (doc_id + chunk_index)")
    doc_id: str = Field(description="Parent document's doc_id")

    # ── Citation metadata (displayed to user) ─────────────────────────────
    doc_title: str = Field(description="Parent document title")
    source_url: str = Field(description="Parent document canonical URL")
    governing_body: str = Field(description="Issuing authority")

    # ── Retrieval filter metadata ──────────────────────────────────────────
    ring: int = Field(ge=1, le=4)
    ring_label: str = Field()

    # ── Temporal metadata ──────────────────────────────────────────────────
    effective_date: str = Field(default="unknown")
    superseded_by: Optional[str] = Field(default=None)
    circular_ref: Optional[str] = Field(default=None)

    # ── Semantic metadata ─────────────────────────────────────────────────
    topic_tags: list[str] = Field(
        default_factory=list,
        description="Keyword tags e.g. ['PPF', 'interest_rate', '80C']"
    )

    # ── Chunk content & position ───────────────────────────────────────────
    chunk_index: int = Field(
        description="Zero-based position of this chunk within its source document"
    )
    chunk_text: str = Field(description="The actual text content of the chunk")
    token_count: int = Field(
        default=0,
        description="Number of tokens in chunk_text (set by chunker)"
    )

    @classmethod
    def make_chunk_id(cls, doc_id: str, chunk_index: int) -> str:
        """Deterministic UUID5 from (doc_id, chunk_index) for reproducibility."""
        seed = f"{doc_id}::{chunk_index}"
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, seed))  # stable across runs

    @classmethod
    def from_document(
        cls,
        doc: SourceDocument,
        chunk_text: str,
        chunk_index: int,
        token_count: int,
        topic_tags: Optional[list[str]] = None,
    ) -> "TextChunk":
        """Factory: build a TextChunk from its parent SourceDocument."""
        return cls(
            chunk_id=cls.make_chunk_id(doc.doc_id, chunk_index),
            doc_id=doc.doc_id,
            doc_title=doc.title,
            source_url=doc.source_url,
            governing_body=doc.governing_body,
            ring=doc.ring,
            ring_label=doc.ring_label,
            effective_date=doc.effective_date,
            superseded_by=doc.superseded_by,
            circular_ref=doc.circular_ref,
            topic_tags=topic_tags or [],
            chunk_index=chunk_index,
            chunk_text=chunk_text,
            token_count=token_count,
        )