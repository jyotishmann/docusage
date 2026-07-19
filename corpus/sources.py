# corpus/sources.py
# Master catalogue of Phase 1 corpus sources.
# Organised by ring. Each entry is a SourceDocument template.
# raw_text is populated by the downloader at runtime.

from __future__ import annotations
from corpus.models import SourceDocument


def _doc(
    title: str, url: str, governing_body: str, ring: int, ring_label: str,
    effective_date: str = "unknown", circular_ref: str | None = None,
    file_format: str = "pdf",
) -> SourceDocument:
    """Helper: builds SourceDocument with deterministic doc_id from URL."""
    return SourceDocument(
        doc_id=SourceDocument.make_doc_id(url),
        title=title, source_url=url, governing_body=governing_body,
        ring=ring, ring_label=ring_label, effective_date=effective_date,
        circular_ref=circular_ref, file_format=file_format,
    )


# ── Ring 1: Market Investments ─────────────────────────────────────────────
RING1_SOURCES = [
    _doc("Zerodha Varsity — Introduction to Stock Markets",
         "https://zerodha.com/varsity/module/innerworld-of-trading/",
         "Zerodha", 1, "Market Investments", file_format="html_module"),
    _doc("Zerodha Varsity — Technical Analysis",
         "https://zerodha.com/varsity/module/technical-analysis/",
         "Zerodha", 1, "Market Investments", file_format="html_module"),
    _doc("Zerodha Varsity — Fundamental Analysis",
         "https://zerodha.com/varsity/module/fundamental-analysis/",
         "Zerodha", 1, "Market Investments", file_format="html_module"),
    _doc("Zerodha Varsity — Mutual Funds and Personal Finance",
         "https://zerodha.com/varsity/module/personalfinance/",
         "Zerodha", 1, "Market Investments", file_format="html_module"),
    _doc("Zerodha Varsity — Futures Trading",
         "https://zerodha.com/varsity/module/futures-trading/",
         "Zerodha", 1, "Market Investments", file_format="html_module"),
    _doc("SEBI Investor Education — Basics of Investing",
         "https://investor.sebi.gov.in/pdf/education/basics-of-investing.pdf",
         "SEBI", 1, "Market Investments", file_format="pdf"),
]

# ── Ring 2: Government Schemes ─────────────────────────────────────────────
RING2_SOURCES = [
    _doc("PPF Scheme Rules 2019",
         "https://www.nsiindia.gov.in/PPF-Scheme-Rules.pdf",
         "Ministry of Finance", 2, "Govt Schemes",
         effective_date="2019-12-12"),
    _doc("NPS Subscriber Guide — PFRDA",
         "https://www.pfrda.org.in/nps-subscriber-booklet.pdf",
         "PFRDA", 2, "Govt Schemes", effective_date="2023-01-01"),
    _doc("EPFO Member Handbook",
         "https://www.epfindia.gov.in/MemberHandbook.pdf",
         "EPFO", 2, "Govt Schemes", effective_date="2022-04-01"),
    _doc("Sovereign Gold Bond Scheme Guidelines",
         "https://rbidocs.rbi.org.in/rdocs/notification/SGB_Guidelines.PDF",
         "RBI", 2, "Govt Schemes", effective_date="2023-09-18"),
    _doc("Atal Pension Yojana Subscriber Information",
         "https://www.pfrda.org.in/apy-brochure-english.pdf",
         "PFRDA", 2, "Govt Schemes", effective_date="2023-01-01"),
    _doc("PM Jan Dhan Yojana Scheme Details",
         "https://pmjdy.gov.in/scheme-details.pdf",
         "Ministry of Finance", 2, "Govt Schemes", effective_date="2014-08-28"),
]

# ── Ring 3: Banking & RBI ──────────────────────────────────────────────────
RING3_SOURCES = [
    _doc("DICGC Deposit Insurance Coverage Guidelines",
         "https://www.dicgc.org.in/FD_DepInsGuidelines.html",
         "DICGC", 3, "Banking & RBI",
         effective_date="2021-02-04", file_format="html"),
    _doc("RBI Retail Direct Investor Guide",
         "https://rbiretaildirect.org.in/assets/pdf/Investor_Guide.pdf",
         "RBI", 3, "Banking & RBI", effective_date="2021-11-12"),
    _doc("RBI Master Direction on Interest Rate on Deposits",
         "https://rbidocs.rbi.org.in/rdocs/notification/MD_IRD.PDF",
         "RBI", 3, "Banking & RBI",
         effective_date="2016-03-03", circular_ref="RBI/2015-16/77"),
    _doc("RBI Know Your Customer (KYC) Directions",
         "https://rbidocs.rbi.org.in/rdocs/notification/KYC_Directions.PDF",
         "RBI", 3, "Banking & RBI",
         effective_date="2023-05-10", circular_ref="RBI/2023-24/38"),
]

# ── Ring 4: Foreign Investments & FEMA ────────────────────────────────────
RING4_SOURCES = [
    _doc("LRS Master Direction — RBI Liberalized Remittance Scheme",
         "https://rbidocs.rbi.org.in/rdocs/notification/LRS_Master_Direction.PDF",
         "RBI", 4, "Foreign Investments",
         effective_date="2023-09-01", circular_ref="RBI/2023-24/56"),
    _doc("FEMA 1999 — Full Act with Amendments",
         "https://rbidocs.rbi.org.in/rdocs/Publications/FEMA1999.PDF",
         "RBI", 4, "Foreign Investments", effective_date="1999-06-01"),
    _doc("RBI — NRI Accounts (NRE, NRO, FCNR-B) Master Direction",
         "https://rbidocs.rbi.org.in/rdocs/notification/MD_NRI_Accounts.PDF",
         "RBI", 4, "Foreign Investments", effective_date="2015-01-22"),
    _doc("DTAA India-USA Double Taxation Avoidance Agreement",
         "https://incometaxindia.gov.in/Documents/dtaa/DTAA-USA.pdf",
         "CBDT", 4, "Foreign Investments", effective_date="1990-12-18"),
]

# ── Combined catalogue ─────────────────────────────────────────────────────
ALL_SOURCES: list[SourceDocument] = (
    RING1_SOURCES + RING2_SOURCES + RING3_SOURCES + RING4_SOURCES
)

SOURCES_BY_RING: dict[int, list[SourceDocument]] = {
    1: RING1_SOURCES, 2: RING2_SOURCES, 3: RING3_SOURCES, 4: RING4_SOURCES,
}