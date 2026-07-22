# audit/models.py
# SentenceAudit and AuditResult -- the auditor output contract.
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal

SentenceStatus = Literal["SUPPORTED", "UNCERTAIN", "CONTRADICTED", "UNSUPPORTED"]

@dataclass
class SentenceAudit:
    sentence:         str
    status:           SentenceStatus
    entailment_score: float
    cited_markers:    list[int]
    premise_chunk_id: str | None = None

    @property
    def is_supported(self) -> bool:
        return self.status == "SUPPORTED"

    @property
    def is_flagged(self) -> bool:
        return self.status in ("CONTRADICTED", "UNSUPPORTED")


@dataclass
class AuditResult:
    sentence_audits: list[SentenceAudit]
    flagged:         bool
    support_rate:    float
    answer:          str
    latency_ms:      float = 0.0

    @property
    def supported_count(self) -> int:
        return sum(1 for s in self.sentence_audits if s.is_supported)

    @property
    def contradicted_count(self) -> int:
        return sum(1 for s in self.sentence_audits if s.status == "CONTRADICTED")

    @property
    def unsupported_count(self) -> int:
        return sum(1 for s in self.sentence_audits if s.status == "UNSUPPORTED")

    @property
    def flag_reason(self) -> str:
        if self.contradicted_count > 0:
            return f"{self.contradicted_count} sentence(s) contradicted by cited sources"
        if self.support_rate < 0.40:
            return f"Low support rate ({self.support_rate:.0%} of sentences verified)"
        return ""