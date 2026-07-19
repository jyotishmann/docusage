# corpus/chunker.py — Part 1: Token counter and recursive splitter core

from __future__ import annotations
from typing import Callable
import tiktoken
from config import get_logger, settings

logger = get_logger(__name__)

# Load cl100k_base tokeniser once at module level (fast, <50ms)
_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens using cl100k_base — fast proxy for BGE-M3 SentencePiece."""
    return len(_ENCODING.encode(text, disallowed_special=()))


# Separator hierarchy: prefer natural discourse boundaries first
DEFAULT_SEPARATORS: list[str] = [
    "\n\n",   # Paragraph boundary (highest priority)
    "\n",     # Line boundary
    ". ",     # Sentence end (space avoids splitting "3.14")
    "? ",     # Question sentence boundary
    "! ",     # Exclamation boundary
    "; ",     # Clause boundary
    ", ",     # Phrase boundary
    " ",      # Word boundary (almost never needed for 512-token chunks)
    "",       # Character (last resort)
]


def _split_on_separator(text: str, separator: str) -> list[str]:
    """Split text on separator, re-attaching separator to each piece."""
    if separator == "":
        return list(text)
    parts = text.split(separator)
    result = []
    for i, part in enumerate(parts):
        if i < len(parts) - 1:
            result.append(part + separator)   # re-attach separator
        elif part:
            result.append(part)
    return [p for p in result if p.strip()]


def recursive_split(
    text: str,
    chunk_size: int = settings.CHUNK_SIZE,
    chunk_overlap: int = settings.CHUNK_OVERLAP,
    min_chunk_size: int = settings.CHUNK_MIN_SIZE,
    separators: list[str] | None = None,
    length_fn: Callable[[str], int] = count_tokens,
) -> list[str]:
    """
    Recursively split text into chunks of at most chunk_size tokens.
    Tries separators in order; recurses on over-size pieces.
    """
    if separators is None:
        separators = DEFAULT_SEPARATORS

    # Base case: text already fits
    if length_fn(text) <= chunk_size:
        return [text] if length_fn(text) >= min_chunk_size else []

    current_sep  = separators[0]
    remaining    = separators[1:]
    splits       = _split_on_separator(text, current_sep)
    good_splits: list[str] = []

    for piece in splits:
        if length_fn(piece) > chunk_size:
            if remaining:
                good_splits.extend(
                    recursive_split(
                        piece, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                        min_chunk_size=min_chunk_size, separators=remaining,
                        length_fn=length_fn,
                    )
                )
            else:
                logger.warning("Hard cut applied (no separators left)", chars=len(piece))
                good_splits.append(piece[: chunk_size * 4])   # char-based fallback
        else:
            good_splits.append(piece)

    return _merge_with_overlap(
        good_splits, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        min_chunk_size=min_chunk_size, length_fn=length_fn,
    )


def _merge_with_overlap(
    pieces: list[str],
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_size: int,
    length_fn: Callable[[str], int],
) -> list[str]:
    """Merge pieces into chunks with trailing overlap between consecutive chunks."""
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for piece in pieces:
        piece_len = length_fn(piece)
        if current_len + piece_len > chunk_size and current:
            text = "".join(current)
            if length_fn(text) >= min_chunk_size:
                chunks.append(text)
            # Build overlap from trailing pieces
            overlap: list[str] = []
            overlap_len = 0
            for p in reversed(current):
                pl = length_fn(p)
                if overlap_len + pl > chunk_overlap:
                    break
                overlap.insert(0, p)
                overlap_len += pl
            current = overlap + [piece]
            current_len = overlap_len + piece_len
        else:
            current.append(piece)
            current_len += piece_len

    if current:
        last = "".join(current)
        if length_fn(last) >= min_chunk_size:
            chunks.append(last)

    return chunks

# corpus/chunker.py — Part 2: Ring-specific processing + DocumentChunker
# Append to corpus/chunker.py after Part 1.

from corpus.models import SourceDocument, TextChunk


def _extract_rbi_header(text: str) -> str:
    """Extract first 3 non-empty lines as a compact header string."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return " | ".join(lines[:3]) if lines else ""


class DocumentChunker:
    """
    Orchestrates chunking for a SourceDocument.
    Applies ring-specific pre-processing, then calls recursive_split().
    Returns a list of TextChunk objects.
    """

    def __init__(
        self,
        chunk_size: int    = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
        min_chunk_size: int = settings.CHUNK_MIN_SIZE,
    ):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk_document(self, doc: SourceDocument) -> list[TextChunk]:
        """Convert a SourceDocument into a list of TextChunks."""
        if not doc.raw_text.strip():
            logger.warning("Empty raw_text — skipping", title=doc.title)
            return []

        # Ring-specific pre-processing
        processed_text, chunk_prefix = self._preprocess(doc)

        # Adjust min_size for short scheme brochures (Ring 2)
        min_size = 200 if doc.ring == 2 else self.min_chunk_size

        raw_chunks = recursive_split(
            text=processed_text,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            min_chunk_size=min_size,
        )

        text_chunks: list[TextChunk] = []
        for idx, chunk_text in enumerate(raw_chunks):
            # Prepend ring-specific prefix to every chunk
            full_text = f"[{chunk_prefix}]\n\n{chunk_text}" if chunk_prefix else chunk_text
            text_chunks.append(
                TextChunk.from_document(
                    doc=doc,
                    chunk_text=full_text,
                    chunk_index=idx,
                    token_count=count_tokens(full_text),
                )
            )

        logger.info("Chunked", title=doc.title, ring=doc.ring, chunks=len(text_chunks))
        return text_chunks

    def _preprocess(self, doc: SourceDocument) -> tuple[str, str]:
        """Return (text_to_split, prefix_for_all_chunks)."""
        text   = doc.raw_text
        prefix = ""

        if doc.ring == 3 and doc.circular_ref:
            header = _extract_rbi_header(text)
            prefix = f"{doc.circular_ref} | {header}" if header else doc.circular_ref

        elif doc.ring == 1 and "Zerodha" in doc.governing_body:
            prefix = doc.title.replace("Zerodha Varsity — ", "")

        return text, prefix

    def chunk_corpus(self, documents: list[SourceDocument]) -> list[TextChunk]:
        """Chunk an entire list of documents. Returns all chunks flat."""
        all_chunks: list[TextChunk] = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        logger.info("Corpus chunked", docs=len(documents), chunks=len(all_chunks))
        return all_chunks