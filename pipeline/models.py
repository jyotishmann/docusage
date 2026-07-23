# pipeline/models.py
# PipelineResult: single Gradio-facing output object.
from __future__ import annotations
from dataclasses import dataclass, field

from generation.models import GenerationResult, Citation
from audit.models import AuditResult, SentenceAudit
from query.router import RouterDecision


@dataclass
class PipelineResult:
    generation:                GenerationResult
    audit:                     AuditResult
    router_decision:           RouterDecision
    sub_queries:               list[str]
    ring_filter:               list[str]   = field(default_factory=list)
    retrieval_candidate_count: int         = 0
    total_latency_ms:          float       = 0.0
    error:                     str | None  = None

    @property
    def answer(self) -> str:              return self.generation.answer
    @property
    def citations(self) -> list[Citation]: return self.generation.citations
    @property
    def flagged(self) -> bool:            return self.audit.flagged
    @property
    def flag_reason(self) -> str:         return self.audit.flag_reason
    @property
    def sentence_audits(self) -> list[SentenceAudit]:
        return self.audit.sentence_audits
    @property
    def was_decomposed(self) -> bool:     return self.router_decision.decompose
    @property
    def decomposition_reason(self) -> str: return self.router_decision.reason
    @property
    def tokens_generated(self) -> int:   return self.generation.tokens_generated
    @property
    def support_rate(self) -> float:      return self.audit.support_rate
    @property
    def is_error(self) -> bool:           return self.error is not None

    def __repr__(self) -> str:
        return (f"PipelineResult(flagged={self.flagged}, "
                f"latency={self.total_latency_ms:.0f}ms, "
                f"citations={len(self.citations)})")