# tests/test_pipeline.py -- 18 pipeline integration tests
import pytest
from unittest.mock import MagicMock, patch
from corpus.models import TextChunk
from retrieval import RankedChunk
from generation.models import GenerationResult, Citation
from audit.models import AuditResult, SentenceAudit
from query.router import RouterDecision
from pipeline import RAGPipeline, PipelineResult


def make_chunk(cid="c1"):
    tc = TextChunk(chunk_id=cid, doc_id="d1", doc_title="Doc",
        source_url="https://t.com", governing_body="NSSF",
        ring=2, ring_label="Govt Schemes", chunk_index=0,
        chunk_text="PPF interest rate is 7.1 percent per annum.")
    return RankedChunk(chunk=tc, rank=1, score=0.8, source="reranker")


def make_gen_result():
    ch = make_chunk()
    return GenerationResult(
        answer="PPF rate is 7.1% [1].",
        citations=[Citation(marker=1, chunk=ch)],
        query="test", sub_queries=[], context_chunks=[ch],
        tokens_generated=50, latency_ms=3000.0)


def make_audit_result(flagged=False):
    sa = SentenceAudit("PPF rate is 7.1% [1].", "SUPPORTED", 0.85, [1])
    return AuditResult([sa], flagged, 1.0, "PPF rate is 7.1% [1].")


@pytest.fixture
def pipeline():
    with patch("pipeline.rag_pipeline.CorpusRegistry") as mr,          patch("pipeline.rag_pipeline.BM25IndexLoader") as mb,          patch("pipeline.rag_pipeline.FAISSIndexLoader") as mf:
        mr.load.return_value = MagicMock()
        mb.return_value.load_all.return_value = MagicMock()
        mf.return_value.load_all.return_value = MagicMock()
        p = RAGPipeline()

    p.decomposer = MagicMock()
    p.decomposer.decompose.return_value = [
        "What are PPF tax benefits?", "What is the PPF lock-in period?"]

    p.retriever = MagicMock()
    p.retriever.retrieve.return_value = [make_chunk("c1"), make_chunk("c2")]

    p.reranker = MagicMock()
    p.reranker.rerank.return_value = [make_chunk("c1")]

    p.generator = MagicMock()
    p.generator.generate.return_value = make_gen_result()

    p.auditor = MagicMock()
    p.auditor.audit.return_value = make_audit_result()

    p._loaded = True
    return p


class TestPipelineDataFlow:
    def test_returns_pipeline_result(self, pipeline):
        assert isinstance(pipeline.run("What is PPF?", []), PipelineResult)

    def test_answer_in_result(self, pipeline):
        assert len(pipeline.run("What is PPF?", []).answer) > 0

    def test_citations_in_result(self, pipeline):
        assert isinstance(pipeline.run("What is PPF?", []).citations, list)

    def test_retriever_gets_sub_queries(self, pipeline):
        # Force decomposition by using a comparison query
        pipeline.run(
            "PPF vs NPS for long term retirement planning in India", [])
        call = pipeline.retriever.retrieve.call_args
        sub_q = call[1].get("sub_queries") or call[0][0]
        assert isinstance(sub_q, list)

    def test_reranker_gets_original_query(self, pipeline):
        original = "What is the PPF interest rate in India today?"
        pipeline.run(original, [])
        call = pipeline.reranker.rerank.call_args
        q = call[1].get("query") or call[0][0]
        assert q == original

    def test_generator_gets_original_query(self, pipeline):
        original = "What is the PPF interest rate in India today?"
        pipeline.run(original, [])
        call = pipeline.generator.generate.call_args
        q = call[1].get("query") or call[0][0]
        assert q == original

    def test_auditor_receives_gen_result(self, pipeline):
        pipeline.run("What is PPF?", [])
        pipeline.auditor.audit.assert_called_once()
        assert isinstance(
            pipeline.auditor.audit.call_args[0][0], GenerationResult)

    def test_total_latency_recorded(self, pipeline):
        assert pipeline.run("What is PPF?", []).total_latency_ms > 0


class TestPipelineRouting:
    def test_simple_not_decomposed(self, pipeline):
        # "What is PPF?" -> < 15 words -> decompose=False
        result = pipeline.run("What is PPF?", [])
        assert result.was_decomposed is False

    def test_simple_decomposer_not_called(self, pipeline):
        pipeline.run("What is PPF?", [])
        pipeline.decomposer.decompose.assert_not_called()

    def test_complex_decomposed(self, pipeline):
        result = pipeline.run(
            "PPF vs NPS for long term retirement planning in India comparison", [])
        assert result.was_decomposed is True

    def test_complex_decomposer_called(self, pipeline):
        pipeline.run(
            "PPF vs NPS for long term retirement planning in India comparison", [])
        pipeline.decomposer.decompose.assert_called_once()


class TestPipelineErrorHandling:
    def test_empty_query_is_error(self, pipeline):
        assert pipeline.run("", []).is_error is True

    def test_whitespace_query_is_error(self, pipeline):
        assert pipeline.run("   ", []).is_error is True

    def test_no_candidates_is_error(self, pipeline):
        pipeline.retriever.retrieve.return_value = []
        assert pipeline.run("What is PPF?", []).is_error is True

    def test_run_safe_catches_exception(self, pipeline):
        pipeline.generator.generate.side_effect = RuntimeError("CUDA OOM")
        result = pipeline.run_safe("What is PPF?", [])
        assert result.is_error is True

    def test_error_result_has_safe_audit(self, pipeline):
        pipeline.retriever.retrieve.return_value = []
        result = pipeline.run("What is PPF?", [])
        assert isinstance(result.audit, AuditResult)
        assert result.audit.flagged is False

    def test_status_returns_dict(self, pipeline):
        s = pipeline.status()
        assert "loaded" in s and "registry_chunks" in s and "components" in s