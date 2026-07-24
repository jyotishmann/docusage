# tests/conftest.py
# Shared pytest fixtures. Auto-discovered by pytest -- no imports needed.
from __future__ import annotations
import pytest
from pathlib import Path

from corpus.models import TextChunk
from corpus import CorpusRegistry
from retrieval import RankedChunk
from indexing import BM25Indexer, BM25IndexLoader


# ---- Helper functions (accept args, not fixtures) -----------------------

def make_text_chunk(
    cid: str = "c1",
    text: str = "PPF interest rate is 7.1 percent per annum.",
    ring: int = 2,
    ring_label: str = "Govt Schemes",
    doc_title: str = "PPF Rules 2019",
    governing_body: str = "Ministry of Finance",
    source_url: str = "https://example.com/ppf",
    tags: list = None,
) -> TextChunk:
    return TextChunk(
        chunk_id=cid, doc_id=f"doc_{cid}", doc_title=doc_title,
        source_url=source_url, governing_body=governing_body,
        ring=ring, ring_label=ring_label,
        chunk_index=0, chunk_text=text,
        topic_tags=tags or [],
    )


def make_ranked_chunk(
    cid: str = "c1",
    text: str = "PPF interest rate is 7.1 percent per annum.",
    ring: int = 2,
    rank: int = 1,
    score: float = 0.8,
    source: str = "rrf",
) -> RankedChunk:
    tc = make_text_chunk(cid=cid, text=text, ring=ring)
    return RankedChunk(chunk=tc, rank=rank, score=score, source=source)


# ---- Synthetic corpus (12 chunks across 4 rings) -----------------------

@pytest.fixture(scope="module")
def synthetic_corpus() -> list[TextChunk]:
    return [
        # Ring 1: Market Investments
        make_text_chunk("m1", "ELSS mutual funds have a 3 year lock-in under section 80C.", 1,
                        "Market Investments", "Zerodha Varsity", "SEBI", "https://z.co/elss"),
        make_text_chunk("m2", "Nifty 50 index tracks the top 50 companies on NSE by market cap.", 1,
                        "Market Investments", "Zerodha Varsity", "SEBI", "https://z.co/nifty"),
        make_text_chunk("m3", "SIP allows systematic investment in mutual funds monthly.", 1,
                        "Market Investments", "Zerodha Varsity", "SEBI", "https://z.co/sip"),
        # Ring 2: Govt Schemes
        make_text_chunk("g1", "PPF interest rate is 7.1 percent per annum compounded annually.", 2,
                        "Govt Schemes", "PPF Rules 2019", "Ministry of Finance", "https://nssf.gov.in"),
        make_text_chunk("g2", "NPS subscriber must purchase annuity upon retirement at age 60.", 2,
                        "Govt Schemes", "NPS Circular", "PFRDA", "https://pfrda.org.in"),
        make_text_chunk("g3", "SGB offers 2.5 percent annual interest plus capital appreciation.", 2,
                        "Govt Schemes", "SGB Scheme", "RBI", "https://rbi.org.in/sgb"),
        # Ring 3: Banking and RBI
        make_text_chunk("b1", "DICGC insures bank deposits up to 5 lakh rupees per depositor.", 3,
                        "Banking & RBI", "DICGC Act", "RBI", "https://dicgc.org.in"),
        make_text_chunk("b2", "Fixed deposits offer guaranteed returns with tenures from 7 days.", 3,
                        "Banking & RBI", "RBI Master Direction", "RBI", "https://rbi.org.in/fd"),
        make_text_chunk("b3", "KYC is mandatory for all bank account openings in India.", 3,
                        "Banking & RBI", "RBI KYC Master", "RBI", "https://rbi.org.in/kyc"),
        # Ring 4: Foreign Investments
        make_text_chunk("f1", "LRS limit is 250000 USD per financial year under FEMA.", 4,
                        "Foreign Investments", "FEMA LRS Rules", "RBI", "https://rbi.org.in/lrs"),
        make_text_chunk("f2", "DTAA prevents double taxation for NRIs earning in India and abroad.", 4,
                        "Foreign Investments", "DTAA Guidelines", "CBDT", "https://incometax.gov.in"),
        make_text_chunk("f3", "NRE accounts are fully repatriable and interest is tax-free in India.", 4,
                        "Foreign Investments", "FEMA NRE Rules", "RBI", "https://rbi.org.in/nre"),
    ]


@pytest.fixture(scope="module")
def populated_registry(synthetic_corpus) -> CorpusRegistry:
    reg = CorpusRegistry()
    reg.add_chunks(synthetic_corpus)
    return reg


@pytest.fixture
def bm25_index_loader(synthetic_corpus, tmp_path) -> BM25IndexLoader:
    BM25Indexer(tmp_path).build(chunks=synthetic_corpus, force=True)
    return BM25IndexLoader(tmp_path).load_all()


# ---- pytest markers -----------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: marks tests that require model weights or long index builds",
    )