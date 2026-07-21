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