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

# tests/test_generation.py -- Part 2: PromptBuilder tests (append to Part 1)
from generation import PromptBuilder


@pytest.fixture
def sample_chunks():
    chunks = []
    for i in range(1, 4):
        tc = TextChunk(
            chunk_id=f'c{i}', doc_id='d1', doc_title=f'Test Document {i}',
            source_url='https://test.com', governing_body=f'Authority {i}',
            ring=i, ring_label=f'Ring {i}', chunk_index=0,
            chunk_text=f'Content for chunk {i} about Indian personal finance.',
            effective_date='2024-01-01',
        )
        chunks.append(RankedChunk(chunk=tc, rank=i, score=0.9, source='reranker'))
    return chunks


class TestPromptBuilder:
    def test_returns_string(self, sample_chunks):
        prompt = PromptBuilder.build('What is PPF?', sample_chunks)
        assert isinstance(prompt, str) and len(prompt) > 0

    def test_query_in_prompt(self, sample_chunks):
        prompt = PromptBuilder.build('What is PPF interest rate?', sample_chunks)
        assert 'What is PPF interest rate?' in prompt

    def test_chunks_numbered_sequentially(self, sample_chunks):
        prompt = PromptBuilder.build('query', sample_chunks)
        assert '[1] Title: Test Document 1' in prompt
        assert '[2] Title: Test Document 2' in prompt
        assert '[3] Title: Test Document 3' in prompt

    def test_chunk_metadata_in_prompt(self, sample_chunks):
        prompt = PromptBuilder.build('query', sample_chunks)
        assert 'Authority 1' in prompt
        assert 'Ring 1' in prompt
        assert '2024-01-01' in prompt

    def test_chunk_text_in_prompt(self, sample_chunks):
        prompt = PromptBuilder.build('query', sample_chunks)
        assert 'Content for chunk 1 about Indian personal finance.' in prompt

    def test_chatml_tokens_present(self, sample_chunks):
        prompt = PromptBuilder.build('query', sample_chunks)
        assert '<|im_start|>system' in prompt
        assert '<|im_end|>' in prompt
        assert '<|im_start|>user' in prompt
        assert '<|im_start|>assistant' in prompt

    def test_system_constraints_present(self, sample_chunks):
        prompt = PromptBuilder.build('query', sample_chunks)
        assert 'ONLY' in prompt   # context-only constraint
        assert '[N]' in prompt    # citation format instruction

    def test_count_chunks_utility(self, sample_chunks):
        prompt = PromptBuilder.build('query', sample_chunks)
        assert PromptBuilder.count_chunks_in_prompt(prompt) == 3

    def test_empty_chunks_valid(self):
        prompt = PromptBuilder.build('What is PPF?', [])
        assert 'What is PPF?' in prompt

    def test_chunk_order_preserved(self, sample_chunks):
        prompt = PromptBuilder.build('query', sample_chunks)
        pos = [prompt.index(f'[{i}] Title: Test Document {i}') for i in range(1, 4)]
        assert pos == sorted(pos)

# tests/test_generation.py -- Part 3: CitationFormatter tests (append)
from generation import CitationFormatter
from generation.models import Citation


class TestCitationFormatter:
    def test_single_valid_marker(self, sample_chunks):
        citations = CitationFormatter.format('PPF rate is 7.1% [1].', sample_chunks)
        assert len(citations) == 1 and citations[0].marker == 1

    def test_multiple_markers(self, sample_chunks):
        answer = 'PPF rate [1]. ELSS lock-in [2]. NPS at 60 [3].'
        citations = CitationFormatter.format(answer, sample_chunks)
        assert [c.marker for c in citations] == [1, 2, 3]

    def test_duplicates_deduplicated(self, sample_chunks):
        answer = 'PPF rate [1] is good. Also PPF [1] allows withdrawal [1].'
        citations = CitationFormatter.format(answer, sample_chunks)
        assert len(citations) == 1 and citations[0].marker == 1

    def test_out_of_range_ignored(self, sample_chunks):
        answer = 'Some claim [99] and another [0].'
        assert CitationFormatter.format(answer, sample_chunks) == []

    def test_no_markers_returns_empty(self, sample_chunks):
        assert CitationFormatter.format('PPF has 15 year lock-in.', sample_chunks) == []

    def test_sorted_by_marker(self, sample_chunks):
        answer = 'Claim [3] and [1] and [2].'
        citations = CitationFormatter.format(answer, sample_chunks)
        assert [c.marker for c in citations] == [1, 2, 3]

    def test_chunk_id_correct_index(self, sample_chunks):
        citations = CitationFormatter.format('Text [2]', sample_chunks)
        assert citations[0].chunk.chunk_id == sample_chunks[1].chunk_id

    def test_citation_properties(self, sample_chunks):
        citations = CitationFormatter.format('Text [1]', sample_chunks)
        c = citations[0]
        assert c.doc_title == 'Test Document 1'
        assert c.governing_body == 'Authority 1'
        assert c.ring_label == 'Ring 1'

    def test_empty_answer_returns_empty(self, sample_chunks):
        assert CitationFormatter.format('', sample_chunks) == []

    def test_empty_chunks_returns_empty(self):
        assert CitationFormatter.format('Text [1]', []) == []

    def test_count_markers_includes_duplicates(self, sample_chunks):
        answer = 'Claim [1] and [1] again and [2] here.'
        assert CitationFormatter.count_markers(answer) == 3

    def test_has_uncited_all_cited(self, sample_chunks):
        answer = 'PPF rate 7.1% [1]. ELSS 3yr lock-in [2]. NPS annuity [3].'
        assert CitationFormatter.has_uncited_claims(answer) is False

    def test_has_uncited_mostly_uncited(self, sample_chunks):
        answer = 'PPF is a scheme. NPS is a pension. ELSS is a fund. LRS 250k [1].'
        assert CitationFormatter.has_uncited_claims(answer) is True