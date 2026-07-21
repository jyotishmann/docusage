# generation/models.py
# Citation and GenerationResult -- the output contract for the generation layer.
from __future__ import annotations
from dataclasses import dataclass, field
from retrieval import RankedChunk


@dataclass
class Citation:
    marker: int          # the N in [N] as it appears in the answer
    chunk:  RankedChunk  # the corresponding context chunk

    @property
    def doc_title(self) -> str:      return self.chunk.doc_title
    @property
    def source_url(self) -> str:     return self.chunk.source_url
    @property
    def governing_body(self) -> str: return self.chunk.governing_body
    @property
    def ring_label(self) -> str:     return self.chunk.ring_label
    @property
    def effective_date(self) -> str: return self.chunk.effective_date
    @property
    def circular_ref(self):          return self.chunk.circular_ref


@dataclass
class GenerationResult:
    answer:           str
    citations:        list[Citation]
    query:            str
    sub_queries:      list[str]          = field(default_factory=list)
    context_chunks:   list[RankedChunk]  = field(default_factory=list)
    tokens_generated: int                = 0
    latency_ms:       float              = 0.0

    @property
    def has_citations(self) -> bool:
        return len(self.citations) > 0

    @property
    def cited_chunk_ids(self) -> list[str]:
        return [c.chunk.chunk_id for c in self.citations]