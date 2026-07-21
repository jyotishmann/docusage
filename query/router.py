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