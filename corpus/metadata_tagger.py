# corpus/metadata_tagger.py
# Vocabulary-based topic tag assignment via compiled regex patterns.
# Deterministic, fast, domain-precise.

from __future__ import annotations
import re
from corpus.models import TextChunk
from config import get_logger

logger = get_logger(__name__)

# (canonical_tag, [alias_patterns]) — case-insensitive matching
TAG_VOCABULARY: list[tuple[str, list[str]]] = [
    # Ring 1: Market investments
    ("mutual_fund",   ["mutual fund", r"\bMF\b", "AMFI"]),
    ("SIP",           ["systematic investment plan", r"\bSIP\b"]),
    ("ETF",           [r"\bETF\b", "exchange traded fund"]),
    ("ELSS",          [r"\bELSS\b", "equity linked saving", "tax saver fund"]),
    ("stock_market",  ["equity market", "stock exchange", r"\bNSE\b", r"\bBSE\b"]),
    ("index_fund",    ["index fund", "Nifty 50 fund", "passive fund"]),
    ("NAV",           [r"\bNAV\b", "net asset value"]),
    ("expense_ratio", ["expense ratio", r"\bTER\b"]),
    ("dividend",      ["dividend", r"\bIDCW\b"]),
    ("derivatives",   ["futures", "options", r"\bF&O\b", "derivative"]),
    # Ring 2: Government schemes
    ("PPF",           ["public provident fund", r"\bPPF\b"]),
    ("NPS",           ["national pension system", r"\bNPS\b"]),
    ("EPFO",          [r"\bEPF\b", r"\bEPFO\b", "employee provident fund"]),
    ("SGB",           ["sovereign gold bond", r"\bSGB\b"]),
    ("APY",           ["atal pension yojana", r"\bAPY\b"]),
    ("NSC",           ["national savings certificate", r"\bNSC\b"]),
    ("SCSS",          ["senior citizen savings scheme", r"\bSCSS\b"]),
    ("Sukanya",       ["sukanya samriddhi", r"\bSSY\b"]),
    ("interest_rate", ["interest rate", "rate of interest", r"\bp\.a\.", "per annum"]),
    ("lock_in",       ["lock-in", "lock in period", "maturity period"]),
    # Ring 3: Banking & RBI
    ("FD",            ["fixed deposit", r"\bFD\b", "term deposit"]),
    ("RD",            ["recurring deposit", r"\bRD\b"]),
    ("DICGC",         [r"\bDICGC\b", "deposit insurance", "deposit guarantee"]),
    ("KYC",           [r"\bKYC\b", "know your customer"]),
    ("G-Sec",         [r"\bG-Sec\b", "government securities", "gilt fund",
                       "RBI Retail Direct"]),
    ("UPI",           [r"\bUPI\b", "unified payment", r"\bNPCI\b"]),
    # Ring 4: Foreign investments
    ("LRS",           [r"\bLRS\b", "liberalized remittance", "remittance scheme"]),
    ("FEMA",          [r"\bFEMA\b", "foreign exchange management"]),
    ("DTAA",          [r"\bDTAA\b", "double taxation", "tax treaty"]),
    ("NRE_account",   [r"\bNRE\b", "non-resident external"]),
    ("NRO_account",   [r"\bNRO\b", "non-resident ordinary"]),
    ("FCNR",          [r"\bFCNR\b", "foreign currency non-resident"]),
    ("remittance",    ["remittance", "wire transfer", "foreign transfer"]),
    # Cross-cutting tax
    ("80C",           [r"\b80C\b", "section 80C"]),
    ("80CCD",         [r"\b80CCD\b", "section 80CCD"]),
    ("LTCG",          [r"\bLTCG\b", "long term capital gain"]),
    ("STCG",          [r"\bSTCG\b", "short term capital gain"]),
    ("TDS",           [r"\bTDS\b", "tax deducted at source"]),
    ("income_tax",    ["income tax", r"\bCBDT\b"]),
]

# Compile patterns once at module load time
_COMPILED: list[tuple[str, list[re.Pattern]]] = [
    (tag, [re.compile(alias, re.IGNORECASE) for alias in aliases])
    for tag, aliases in TAG_VOCABULARY
]


class MetadataTagger:
    """Assigns topic_tags to TextChunks via vocabulary regex matching."""

    def tag_chunk(self, chunk: TextChunk) -> TextChunk:
        """Assign topic_tags in-place. Returns chunk."""
        found: set[str] = set()
        text = chunk.chunk_text
        for canonical, patterns in _COMPILED:
            for pat in patterns:
                if pat.search(text):
                    found.add(canonical)
                    break  # one alias match per tag is sufficient
        chunk.topic_tags = sorted(found)  # sorted for determinism
        return chunk

    def tag_corpus(self, chunks: list[TextChunk]) -> list[TextChunk]:
        """Tag all chunks in-place. Returns the same list."""
        tagged = sum(1 for c in chunks if self.tag_chunk(c).topic_tags)
        logger.info("Tagging complete", total=len(chunks), tagged=tagged)
        return chunks