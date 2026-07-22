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

# audit/auditor.py -- Part 2: NLI scoring (append after Part 1)
from scipy.special import softmax


# append to HallucinationAuditor class:

def audit(self, generation_result: GenerationResult) -> AuditResult:
    # Full pipeline: split -> extract markers -> batch NLI -> assign status.
    if self._model is None:
        raise RuntimeError("Call .load() before .audit()")

    t0     = time.perf_counter()
    answer = generation_result.answer
    chunks = generation_result.context_chunks

    sentences    = self.split_sentences(answer)
    sent_markers = [self.extract_cited_markers(s) for s in sentences]

    # Build (premise, hypothesis) pairs for cited sentences
    cited_pairs: list[tuple[str, str]] = []
    pair_map:    dict[int, tuple[int, int]] = {}
    for si, (sentence, markers) in enumerate(zip(sentences, sent_markers)):
        for n in markers:
            if 1 <= n <= len(chunks):
                pair_map[len(cited_pairs)] = (si, n - 1)
                cited_pairs.append((chunks[n - 1].chunk_text, sentence))

    # Batched NLI inference
    if cited_pairs:
        logits            = self._model.predict(cited_pairs,
                            show_progress_bar=False, convert_to_numpy=True)
        probs             = softmax(logits, axis=1)     # (n_pairs, 3)
        entailment_probs  = probs[:, _NLI_ENTAILMENT]   # (n_pairs,)
    else:
        entailment_probs = np.array([])

    # Aggregate: max entailment per sentence across cited chunks
    sent_best: dict[int, tuple[float, int]] = {}
    for pair_idx, (si, ci) in pair_map.items():
        score = float(entailment_probs[pair_idx])
        if si not in sent_best or score > sent_best[si][0]:
            sent_best[si] = (score, ci)

    # Build SentenceAudit for each sentence
    sentence_audits: list[SentenceAudit] = []
    for si, (sentence, markers) in enumerate(zip(sentences, sent_markers)):
        if not markers:
            sa = SentenceAudit(sentence, "UNSUPPORTED", 0.0, [], None)
        elif si in sent_best:
            best_score, best_ci = sent_best[si]
            if best_score >= self.entailment_threshold:
                status = "SUPPORTED"
            elif best_score < self.flag_threshold:
                status = "CONTRADICTED"
            else:
                status = "UNCERTAIN"
            sa = SentenceAudit(sentence, status, best_score, markers,
                                chunks[best_ci].chunk_id)
        else:
            sa = SentenceAudit(sentence, "UNSUPPORTED", 0.0, markers, None)
        sentence_audits.append(sa)

    # Compute overall metrics
    n_total      = max(1, len(sentence_audits))
    supported    = sum(1 for s in sentence_audits if s.status == "SUPPORTED")
    contradicted = sum(1 for s in sentence_audits if s.status == "CONTRADICTED")
    support_rate = supported / n_total
    flagged      = (contradicted > 0) or (support_rate < self.support_rate_min)
    latency_ms   = (time.perf_counter() - t0) * 1000

    logger.info("Audit complete", sentences=n_total, supported=supported,
                contradicted=contradicted, flagged=flagged,
                latency_ms=round(latency_ms))

    return AuditResult(sentence_audits=sentence_audits, flagged=flagged,
                        support_rate=support_rate, answer=answer,
                        latency_ms=latency_ms)