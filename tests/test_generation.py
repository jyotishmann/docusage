# tests/test_generation.py -- Part 1: Reranker unit tests (9 tests)
# Mocks CrossEncoder.predict() -- no bge-reranker download, no GPU.
# Run: pytest tests/test_generation.py -v
import pytest
from unittest.mock import MagicMock
from corpus.models import TextChunk
from retrieval import RankedChunk
from generation import Reranker


def make_rc(cid, text, ring=1):
    tc = TextChunk(
        chunk_id=cid, doc_id='d1', doc_title=f'Doc {cid}',
        source_url='https://test.com', governing_body='Test',
        ring=ring, ring_label='Test Ring', chunk_index=0, chunk_text=text,
    )
    return RankedChunk(chunk=tc, rank=1, score=0.5, source='rrf')


@pytest.fixture
def mock_reranker():
    r = Reranker(model_name='BAAI/bge-reranker-v2-m3', top_k=3)
    mock_ce = MagicMock()
    r._model = mock_ce
    return r, mock_ce


class TestReranker:
    def test_higher_score_ranks_first(self, mock_reranker):
        r, ce = mock_reranker
        chunks = [make_rc('c1', 'PPF text'), make_rc('c2', 'NPS text')]
        ce.predict.return_value = [0.3, 0.8]  # c2 scores higher
        result = r.rerank('PPF query', chunks)
        assert result[0].chunk_id == 'c2' and result[1].chunk_id == 'c1'

    def test_output_source_is_reranker(self, mock_reranker):
        r, ce = mock_reranker
        ce.predict.return_value = [0.5]
        assert all(rc.source == 'reranker' for rc in r.rerank('q', [make_rc('c1', 't')]))

    def test_ranks_sequential(self, mock_reranker):
        r, ce = mock_reranker
        chunks = [make_rc(f'c{i}', f't{i}') for i in range(5)]
        ce.predict.return_value = [0.9, 0.7, 0.5, 0.3, 0.1]
        result = r.rerank('q', chunks, top_k=5)
        assert [rc.rank for rc in result] == [1, 2, 3, 4, 5]

    def test_top_k_limits_output(self, mock_reranker):
        r, ce = mock_reranker
        chunks = [make_rc(f'c{i}', f't{i}') for i in range(10)]
        ce.predict.return_value = list(range(10, 0, -1))
        assert len(r.rerank('q', chunks, top_k=3)) == 3

    def test_score_updated_to_reranker_score(self, mock_reranker):
        r, ce = mock_reranker
        ce.predict.return_value = [0.742]
        result = r.rerank('q', [make_rc('c1', 't')])
        assert abs(result[0].score - 0.742) < 1e-6

    def test_empty_candidates_returns_empty(self, mock_reranker):
        r, _ = mock_reranker
        assert r.rerank('q', []) == []

    def test_predict_called_with_correct_pairs(self, mock_reranker):
        r, ce = mock_reranker
        chunks = [make_rc('c1', 'PPF text'), make_rc('c2', 'NPS text')]
        ce.predict.return_value = [0.5, 0.3]
        r.rerank('my query', chunks)
        pairs = ce.predict.call_args[0][0]
        assert pairs[0] == ('my query', 'PPF text')
        assert pairs[1] == ('my query', 'NPS text')

    def test_load_not_called_raises(self):
        r = Reranker(model_name='test')
        with pytest.raises(RuntimeError, match='load'):
            r.rerank('q', [make_rc('c1', 't')])

    def test_rrf_score_overwritten(self, mock_reranker):
        r, ce = mock_reranker
        chunk = make_rc('c1', 'text')
        chunk = RankedChunk(chunk=chunk.chunk, rank=5, score=0.999, source='rrf')
        ce.predict.return_value = [0.1]
        result = r.rerank('q', [chunk])
        assert result[0].score < 0.5 and result[0].source == 'reranker'