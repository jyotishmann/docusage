# audit/auditor.py -- Part 1: Model loading and sentence splitting
from __future__ import annotations
import re, time
import numpy as np

from config import get_logger, settings
from retrieval import RankedChunk
from generation.models import GenerationResult
from audit.models import SentenceAudit, AuditResult

logger = get_logger(__name__)

# NLI label indices
_NLI_CONTRADICTION = 0
_NLI_NEUTRAL       = 1
_NLI_ENTAILMENT    = 2

_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z\[])')
_CITATION_RE   = re.compile(r'\[(\d+)\]')


class HallucinationAuditor:
    # Cross-encoder NLI auditor for per-sentence hallucination detection.

    def __init__(
        self,
        model_name: str | None = None,
        entailment_threshold: float = settings.AUDITOR_ENTAILMENT_THRESHOLD,
        flag_threshold:       float = settings.AUDITOR_FLAG_THRESHOLD,
        support_rate_min:     float = 0.40,
    ):
        self.model_name           = model_name or settings.AUDITOR_MODEL
        self.entailment_threshold = entailment_threshold
        self.flag_threshold       = flag_threshold
        self.support_rate_min     = support_rate_min
        self._model               = None

    def load(self) -> "HallucinationAuditor":
        # Load cross-encoder/nli-deberta-v3-small (~400MB, ~0.7GB VRAM).
        from sentence_transformers import CrossEncoder
        logger.info("Loading auditor", model=self.model_name)
        self._model = CrossEncoder(
            self.model_name, num_labels=3, device=settings.DEVICE)
        logger.info("Auditor model ready")
        return self

    @staticmethod
    def split_sentences(text: str, min_len: int = 15) -> list[str]:
        # Split on sentence-ending punctuation then whitespace + capital or '['.
        raw = _SENT_SPLIT_RE.split(text.strip())
        return [s.strip() for s in raw if len(s.strip()) >= min_len]

    @staticmethod
    def extract_cited_markers(sentence: str) -> list[int]:
        # Return all [N] integer values found in the sentence.
        return [int(m) for m in _CITATION_RE.findall(sentence)]