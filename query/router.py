# query/router.py -- Part 1: Vocabulary constants and rule functions
from __future__ import annotations
import re
from config import get_logger, settings
logger = get_logger(__name__)

CONJUNCTION_TRIGGERS = [
    "and also", "as well as", "in addition to", "along with",
]
COMPARISON_TRIGGERS = [
    r"\bvs\b", r"\bversus\b", "compared to", "comparison between",
    "difference between", "better option", "which is better",
    "should i choose between", "pros and cons",
]
MULTI_PART_TRIGGERS = [
    r"\bfirstly\b", r"\bsecondly\b", r"\bthirdly\b", r"\bfinally\b",
]
PRODUCT_VOCABULARY = [
    r"\bPPF\b", r"\bNPS\b", r"\bEPF\b", r"\bSGB\b", r"\bAPY\b",
    r"\bSCSS\b", r"\bNSC\b", r"\bSSY\b", r"\bELSS\b", r"\bSIP\b",
    r"\bETF\b", r"\bNAV\b", r"\b80C\b", r"\b80CCD\b",
    r"\bLTCG\b", r"\bSTCG\b", r"\bFD\b", r"\bRD\b", r"\bDICGC\b",
    r"\bLRS\b", r"\bFEMA\b", r"\bDTAA\b", r"\bFCNR\b",
    r"\bNRE\b", r"\bNRO\b",
]

_CP = [re.compile(t, re.IGNORECASE) for t in CONJUNCTION_TRIGGERS]
_KP = [re.compile(t, re.IGNORECASE) for t in COMPARISON_TRIGGERS]
_MP = [re.compile(t, re.IGNORECASE | re.MULTILINE) for t in MULTI_PART_TRIGGERS]
_PP = [re.compile(t, re.IGNORECASE) for t in PRODUCT_VOCABULARY]

def _word_count(q):   return len(q.strip().split())
def _qmark_count(q):  return q.count("?")
def _match_any(patterns, q):
    for p in patterns:
        m = p.search(q)
        if m: return True, m.group(0)
    return False, ""
def _has_conjunction(q): return _match_any(_CP, q)
def _has_comparison(q):  return _match_any(_KP, q)
def _has_multi_part(q):  return _match_any(_MP, q)
def _product_count(q):   return sum(1 for p in _PP if p.search(q))

# query/router.py -- Part 2: QueryRouter class (append after Part 1)
from dataclasses import dataclass

@dataclass
class RouterDecision:
    decompose:  bool
    reason:     str
    confidence: float = 1.0  # always 1.0 -- rule-based is deterministic

class QueryRouter:
    """Rule-based query complexity classifier. Returns RouterDecision."""
    def __init__(
        self,
        max_words_simple:  int = settings.ROUTER_MAX_WORDS_SIMPLE,
        min_words_complex: int = settings.ROUTER_MIN_WORDS_COMPLEX,
        min_products:      int = 2,
    ):
        self.max_simple   = max_words_simple
        self.min_complex  = min_words_complex
        self.min_products = min_products

    def route(self, query: str) -> RouterDecision:
        """Evaluate 7 rules in priority order. First match wins."""
        query = query.strip()
        if not query:
            return RouterDecision(False, "Empty query")
        wc = _word_count(query)
        if wc < self.max_simple:
            return RouterDecision(False,
                f"Short query ({wc} words < {self.max_simple})")
        qm = _qmark_count(query)
        if qm >= 2:
            return RouterDecision(True, f"Multiple question marks ({qm} found)")
        found, phrase = _has_comparison(query)
        if found:
            return RouterDecision(True, f"Comparison language: {phrase!r}")
        found, phrase = _has_conjunction(query)
        if found:
            return RouterDecision(True, f"Conjunction language: {phrase!r}")
        pc = _product_count(query)
        if pc >= self.min_products:
            return RouterDecision(True,
                f"{pc} product names found (>= {self.min_products})")
        if wc > self.min_complex:
            return RouterDecision(True,
                f"Long query ({wc} words > {self.min_complex})")
        found, phrase = _has_multi_part(query)
        if found:
            return RouterDecision(True, f"Explicit multi-part: {phrase!r}")
        
        return RouterDecision(False,
            f"No complexity triggers ({wc} words, {pc} products)")

    def should_decompose(self, query: str) -> bool:
        return self.route(query).decompose