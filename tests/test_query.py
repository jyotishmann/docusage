# tests/test_query.py -- QueryRouter unit tests (23 tests)
import pytest
from query.router import QueryRouter, RouterDecision

@pytest.fixture
def router(): return QueryRouter()

class TestShortQuery:
    def test_very_short_not_decomposed(self, router):
        assert router.route("What is PPF?").decompose is False
    def test_short_with_vs_not_decomposed(self, router):
        assert router.route("PPF vs NPS?").decompose is False
    def test_reason_mentions_word(self, router):
        assert "word" in router.route("What is PPF?").reason.lower()
    def test_empty_not_decomposed(self, router):
        assert router.route("").decompose is False

class TestLongQuery:
    def test_65_words_decomposed(self, router):
        assert router.route(" ".join(["investing"]*65)).decompose is True
    def test_61_words_decomposed(self, router):
        assert router.route(" ".join(["investing"]*61)).decompose is True
    def test_reason_mentions_long(self, router):
        assert "long" in router.route(" ".join(["investing"]*65)).reason.lower()

class TestQuestionMarks:
    def test_two_qmarks(self, router):
        d = router.route("What is the PPF interest rate? Can I withdraw early?")
        assert d.decompose is True and "question mark" in d.reason.lower()
    def test_three_qmarks(self, router):
        assert router.route("What is PPF? What is NPS? Which is better?").decompose is True

class TestComparisons:
    def test_vs_triggers(self, router):
        assert router.route("PPF vs NPS for long term retirement planning India").decompose is True
    def test_compared_to_triggers(self, router):
        assert router.route("How does ELSS compare compared to PPF for tax saving").decompose is True
    def test_difference_between_triggers(self, router):
        assert router.route("What is the difference between NPS Tier 1 and Tier 2").decompose is True
    def test_pros_and_cons_triggers(self, router):
        assert router.route("Explain pros and cons of index funds versus active funds").decompose is True

class TestConjunctions:
    def test_and_also(self, router):
        d = router.route("What is the PPF interest rate and also how do I open an account")
        assert d.decompose is True and "conjunction" in d.reason.lower()
    def test_as_well_as(self, router):
        assert router.route("Explain NPS withdrawal rules as well as the annuity requirement").decompose is True
    def test_in_addition_to(self, router):
        assert router.route("How does LRS work in addition to what are DTAA benefits for investors").decompose is True

class TestMultipleProducts:
    def test_two_schemes(self, router):
        d = router.route("I want to invest in both PPF and ELSS for maximum tax saving")
        assert d.decompose is True
    def test_three_schemes(self, router):
        assert router.route("Compare PPF NPS and ELSS for a salaried professional in India").decompose is True
    def test_single_scheme_no_trigger(self, router):
        d = router.route("What is the current PPF interest rate for this quarter")
        if d.decompose:
            assert "product" not in d.reason.lower()

class TestRouterDecision:
    def test_fields(self, router):
        d = router.route("PPF vs NPS long term retirement India planning")
        assert all(hasattr(d, f) for f in ["decompose", "reason", "confidence"])
    def test_confidence_is_one(self, router):
        assert router.route("What is PPF?").confidence == 1.0
    def test_should_decompose_bool(self, router):
        assert isinstance(router.should_decompose("What is PPF?"), bool)
    def test_reason_nonempty(self, router):
        assert len(router.route("What is PPF?").reason) > 0

# tests/test_query.py -- QueryDecomposer tests (append after Part 1)
import json
from unittest.mock import MagicMock
from query.decomposer import QueryDecomposer, DecomposerModel

ORIG = "Should I invest in PPF or ELSS and can I remit money to US stocks?"

def mock_decomposer(output: str) -> QueryDecomposer:
    m = MagicMock(spec=DecomposerModel)
    m.generate.return_value = output
    return QueryDecomposer(model=m)

class TestDecomposerParsing:
    def test_valid_json(self):
        out = json.dumps(["What are PPF tax benefits and lock-in period in India?",
                          "What is the ELSS expense ratio and tax saving benefit?",
                          "What is the LRS limit for US stock investing from India?"])
        assert len(mock_decomposer(out).decompose(ORIG)) == 3

    def test_preamble_before_json(self):
        out = ('Here are the sub-questions:\n'
               '["What is PPF interest rate and lock-in?", '
               '"What is ELSS expense ratio and tax benefit?", '
               '"What is LRS limit for US stock investments?"]')
        assert len(mock_decomposer(out).decompose(ORIG)) == 3

    def test_trailing_text(self):
        out = ('["What is PPF interest rate in India?", '
               '"What is ELSS lock-in period for 80C?"]\nHope this helps!')
        assert len(mock_decomposer(out).decompose(ORIG)) == 2

    def test_single_quotes(self):
        out = "['What is the PPF interest rate in India?', 'What is the ELSS lock-in?']"
        assert len(mock_decomposer(out).decompose(ORIG)) == 2

    def test_garbled_with_quoted_strings(self):
        out = ('Sub-questions: "What is the PPF interest rate in India?" '
               'and "What is the ELSS lock-in period for section 80C tax saving?"')
        assert len(mock_decomposer(out).decompose(ORIG)) == 2

    def test_unparseable_fallback(self):
        assert mock_decomposer("I do not understand.").decompose(ORIG) == [ORIG]

    def test_model_exception_fallback(self):
        m = MagicMock(spec=DecomposerModel)
        m.generate.side_effect = RuntimeError("CUDA OOM error")
        assert QueryDecomposer(model=m).decompose(ORIG) == [ORIG]

class TestDecomposerValidation:
    def test_single_item_fallback(self):
        assert mock_decomposer('["What is PPF?"]').decompose(ORIG) == [ORIG]

    def test_short_items_filtered(self):
        assert mock_decomposer('["PPF?", "NPS?", "OK"]').decompose(ORIG) == [ORIG]

    def test_duplicates_removed(self):
        q = "What is the PPF interest rate for FY 2024-25 in India today?"
        out = json.dumps([q, q, "What is the ELSS lock-in for section 80C?"])
        result = mock_decomposer(out).decompose(ORIG)
        assert len(result) == 2 and len(set(r.lower() for r in result)) == 2

    def test_original_filtered(self):
        out = json.dumps([ORIG, "What is ELSS lock-in for 80C tax saving?"])
        assert mock_decomposer(out).decompose(ORIG) == [ORIG]

    def test_more_than_4_truncated(self):
        items = [f"What is Indian financial regulation number {i} for investors?" for i in range(6)]
        assert len(mock_decomposer(json.dumps(items)).decompose(ORIG)) <= 4

    def test_non_strings_filtered(self):
        out = json.dumps([42, "What is the PPF interest rate in India?",
                          {"k": "v"}, "What is the ELSS lock-in period for 80C?"])
        assert len(mock_decomposer(out).decompose(ORIG)) == 2

    def test_empty_query_short_circuits(self):
        assert mock_decomposer("[]").decompose("") == [""]

class TestPromptBuilding:
    def test_prompt_contains_query(self):
        m = MagicMock(spec=DecomposerModel)
        m.generate.return_value = '["What is PPF rate?", "What is ELSS rate?"]' 
        QueryDecomposer(model=m).decompose("test query injection check")
        assert "test query injection check" in m.generate.call_args[0][0]

    def test_prompt_has_chatml_tokens(self):
        m = MagicMock(spec=DecomposerModel)
        m.generate.return_value = '["What is PPF rate?", "What is ELSS rate?"]'
        QueryDecomposer(model=m).decompose("PPF versus NPS retirement planning India")
        prompt = m.generate.call_args[0][0]
        assert "<|im_start|>" in prompt and "<|im_end|>" in prompt