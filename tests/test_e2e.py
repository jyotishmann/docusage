# tests/test_e2e.py -- End-to-end smoke tests (no model weights)
# Run: pytest tests/test_e2e.py -v
# Run including slow: pytest tests/test_e2e.py -v -m slow
from __future__ import annotations
import json
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path


class TestImports:
    def test_config_imports(self):
        from config import settings, get_logger
        assert settings is not None

    def test_corpus_imports(self):
        from corpus import CorpusRegistry
        from corpus.models import TextChunk, SourceDocument

    def test_indexing_imports(self):
        from indexing import BM25Indexer, FAISSIndexer

    def test_retrieval_imports(self):
        from retrieval import HybridRetriever, RankedChunk, RRFFusion

    def test_query_imports(self):
        from query import QueryRouter, QueryDecomposer

    def test_generation_imports(self):
        from generation import Generator, Reranker, PromptBuilder

    def test_audit_imports(self):
        from audit import HallucinationAuditor, AuditResult

    def test_pipeline_imports(self):
        from pipeline import RAGPipeline, PipelineResult

    def test_app_imports(self):
        import app  # should not launch server


class TestConfigSmoke:
    def test_settings_has_required_fields(self):
        from config import settings
        required = [
            "EMBED_MODEL", "DECOMPOSER_MODEL", "RERANKER_MODEL",
            "GENERATOR_MODEL", "AUDITOR_MODEL",
            "BM25_TOP_K", "FAISS_TOP_K", "RRF_K",
            "RERANKER_INPUT_K", "GENERATOR_CONTEXT_K",
            "AUDITOR_ENTAILMENT_THRESHOLD", "AUDITOR_FLAG_THRESHOLD",
            "CHUNK_SIZE", "CHUNK_OVERLAP", "CHUNK_MIN_SIZE",
        ]
        for field in required:
            assert hasattr(settings, field), f"settings missing: {field}"

    def test_ring_label_to_id_all_labels(self):
        from config import settings
        for label in ["Market Investments", "Govt Schemes",
                      "Banking & RBI", "Foreign Investments"]:
            ring_id = settings.ring_label_to_id(label)
            assert ring_id in (1, 2, 3, 4), f"Bad ring_id for {label!r}"

    def test_ensure_dirs_creates_dirs(self, tmp_path, monkeypatch):
        from config import Settings
        monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
        cfg = Settings()
        cfg.ensure_dirs()
        assert Path(cfg.INDEX_DIR).exists()
        assert Path(cfg.CORPUS_DIR).exists()


class TestRouterPromptChain:
    def test_router_to_prompt_builder_chain(self, synthetic_corpus, populated_registry):
        from query import QueryRouter
        from generation import PromptBuilder
        from retrieval import RankedChunk

        router = QueryRouter()

        # Simple query: no decomposition
        simple_dec = router.route("What is PPF interest rate?")
        assert simple_dec.decompose is False

        # Complex query: comparison triggers decomposition
        complex_dec = router.route(
            "PPF vs NPS for long term retirement planning in India comparison")
        assert complex_dec.decompose is True

        # Build a prompt from synthetic corpus chunks
        chunks = [
            RankedChunk(chunk=synthetic_corpus[3], rank=1, score=0.9, source="reranker"),
            RankedChunk(chunk=synthetic_corpus[4], rank=2, score=0.8, source="reranker"),
        ]
        prompt = PromptBuilder.build("What is PPF interest rate?", chunks)
        assert "[1] Title:" in prompt
        assert "[2] Title:" in prompt
        assert "What is PPF interest rate?" in prompt
        assert "<|im_start|>" in prompt

    def test_citation_formatter_chain(self, synthetic_corpus):
        from retrieval import RankedChunk
        from generation import CitationFormatter
        from generation.models import Citation

        chunks = [
            RankedChunk(chunk=synthetic_corpus[3], rank=1, score=0.9, source="reranker"),
            RankedChunk(chunk=synthetic_corpus[4], rank=2, score=0.8, source="reranker"),
        ]
        answer = "PPF rate is 7.1 percent [1]. NPS mandates annuity at 60 [2]."
        citations = CitationFormatter.format(answer, chunks)

        assert len(citations) == 2
        assert citations[0].marker == 1
        assert citations[1].marker == 2
        assert "PPF Rules" in citations[0].doc_title


class TestFullPipelineSmoke:
    """Full pipeline run with mocked model inference."""

    @pytest.fixture
    def smoke_pipeline(self, synthetic_corpus, bm25_index_loader, populated_registry):
        from generation.models import GenerationResult, Citation
        from audit.models import AuditResult, SentenceAudit
        from retrieval import RankedChunk

        reranked_chunk = RankedChunk(
            chunk=synthetic_corpus[3], rank=1, score=0.9, source="reranker")

        with patch("pipeline.rag_pipeline.CorpusRegistry") as mr, \
             patch("pipeline.rag_pipeline.BM25IndexLoader") as mb, \
             patch("pipeline.rag_pipeline.FAISSIndexLoader") as mf:
            mr.load.return_value = populated_registry
            mb.return_value.load_all.return_value = bm25_index_loader
            mf.return_value.load_all.return_value = MagicMock()
            from pipeline import RAGPipeline
            p = RAGPipeline()

        p.decomposer = MagicMock()
        p.decomposer.decompose.return_value = [
            "What is PPF interest rate?", "What is PPF lock-in period?"]

        mock_faiss_ret = MagicMock()
        mock_faiss_ret.retrieve.return_value = []

        from retrieval import BM25Retriever, HybridRetriever, RRFFusion
        real_bm25_ret = BM25Retriever(bm25_index_loader, populated_registry)

        from unittest.mock import MagicMock as MM
        mock_dense = MM()
        mock_dense.retrieve.return_value = [reranked_chunk]

        p.retriever = HybridRetriever(
            bm25_retriever=real_bm25_ret, dense_retriever=mock_dense)

        p.reranker = MagicMock()
        p.reranker.rerank.return_value = [reranked_chunk]

        p.generator = MagicMock()
        p.generator.generate.return_value = GenerationResult(
            answer="PPF interest rate is 7.1 percent per annum [1].",
            citations=[Citation(marker=1, chunk=reranked_chunk)],
            query="What is PPF interest rate?",
            sub_queries=["What is PPF interest rate?", "What is PPF lock-in?"],
            context_chunks=[reranked_chunk],
            tokens_generated=40, latency_ms=2500.0,
        )

        p.auditor = MagicMock()
        sa = SentenceAudit(
            "PPF interest rate is 7.1 percent per annum [1].",
            "SUPPORTED", 0.88, [1], reranked_chunk.chunk_id,
        )
        p.auditor.audit.return_value = AuditResult([sa], False, 1.0,
            "PPF interest rate is 7.1 percent per annum [1].")

        p._loaded = True
        return p

    def test_pipeline_run_returns_result(self, smoke_pipeline):
        from pipeline import PipelineResult
        result = smoke_pipeline.run_safe(
            "PPF vs NPS for retirement planning in India comparison", [])
        assert isinstance(result, PipelineResult)

    def test_pipeline_result_has_answer(self, smoke_pipeline):
        result = smoke_pipeline.run_safe("What is PPF rate?", [])
        assert len(result.answer) > 0

    def test_pipeline_result_has_citations(self, smoke_pipeline):
        result = smoke_pipeline.run_safe("What is PPF rate?", [])
        assert len(result.citations) == 1
        assert result.citations[0].marker == 1

    def test_pipeline_audit_clean(self, smoke_pipeline):
        result = smoke_pipeline.run_safe("What is PPF rate?", [])
        assert result.flagged is False
        assert result.support_rate == 1.0

    def test_pipeline_latency_recorded(self, smoke_pipeline):
        result = smoke_pipeline.run_safe("What is PPF rate?", [])
        assert result.total_latency_ms > 0

    def test_bm25_retriever_real_search(self, bm25_index_loader, populated_registry):
        from retrieval import BM25Retriever
        retriever = BM25Retriever(bm25_index_loader, populated_registry)
        results = retriever.retrieve("PPF interest rate", top_k=3)
        assert len(results) > 0
        top_text = results[0].chunk_text.lower()
        assert "ppf" in top_text or "interest" in top_text


class TestFrontendRenderers:
    """Verify HTML renderers produce valid strings without Gradio running."""

    @pytest.fixture
    def mock_result(self, synthetic_corpus):
        from generation.models import GenerationResult, Citation
        from audit.models import AuditResult, SentenceAudit
        from query.router import RouterDecision
        from retrieval import RankedChunk
        from pipeline import PipelineResult

        chunk = RankedChunk(
            chunk=synthetic_corpus[3], rank=1, score=0.9, source="reranker")
        cit = Citation(marker=1, chunk=chunk)
        sa  = SentenceAudit(
            "PPF rate is 7.1 percent [1].", "SUPPORTED", 0.88, [1])
        gen = GenerationResult(
            answer="PPF rate is 7.1 percent [1].",
            citations=[cit], query="What is PPF?",
            sub_queries=["What is PPF rate?"],
            context_chunks=[chunk],
            tokens_generated=30, latency_ms=1500.0,
        )
        aud = AuditResult([sa], False, 1.0, gen.answer)
        rd  = RouterDecision(decompose=False, reason="Short query (3 words)")
        return PipelineResult(
            generation=gen, audit=aud, router_decision=rd,
            sub_queries=["What is PPF rate?"],
            ring_filter=["Govt Schemes"],
            retrieval_candidate_count=12,
            total_latency_ms=3200.0,
        )

    def test_render_answer_returns_string(self, mock_result):
        from app import render_answer
        html = render_answer(mock_result)
        assert isinstance(html, str) and len(html) > 0

    def test_render_answer_contains_cite_badge(self, mock_result):
        from app import render_answer
        assert "cite-badge" in render_answer(mock_result)

    def test_render_answer_colour_codes_sentence(self, mock_result):
        from app import render_answer
        html = render_answer(mock_result)
        assert "sent-audit" in html or "SUPPORTED" in html

    def test_render_citations_returns_string(self, mock_result):
        from app import render_citations
        html = render_citations(mock_result)
        assert isinstance(html, str) and len(html) > 0

    def test_render_citations_contains_link(self, mock_result):
        from app import render_citations
        assert "href=" in render_citations(mock_result)

    def test_render_banner_empty_when_not_flagged(self, mock_result):
        from app import render_banner
        assert render_banner(mock_result) == ""

    def test_render_banner_non_empty_when_flagged(self, mock_result, synthetic_corpus):
        from app import render_banner
        mock_result.audit.flagged = True
        mock_result.audit.sentence_audits[0].status = "CONTRADICTED"
        html = render_banner(mock_result)
        assert len(html) > 0 and "warning" in html.lower()

    def test_render_metrics_contains_latency(self, mock_result):
        from app import render_metrics
        html = render_metrics(mock_result)
        assert "ms" in html

    def test_render_subqueries_contains_sub_query(self, mock_result):
        from app import render_subqueries
        html = render_subqueries(mock_result)
        assert "PPF rate" in html