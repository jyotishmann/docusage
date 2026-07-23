# tests/test_audit.py -- Part 1: Models and sentence utility tests (22 tests)
# Pure Python, no model loading, ~0.1s.
# Run: pytest tests/test_audit.py -v
import pytest
from audit.models  import SentenceAudit, AuditResult
from audit.auditor import HallucinationAuditor


class TestSentenceAuditModel:
    def test_is_supported_true(self):
        assert SentenceAudit("t", "SUPPORTED", 0.85, [1]).is_supported is True
    def test_is_supported_false_contradicted(self):
        assert SentenceAudit("t", "CONTRADICTED", 0.2, [1]).is_supported is False
    def test_is_supported_false_uncertain(self):
        assert SentenceAudit("t", "UNCERTAIN", 0.55, [1]).is_supported is False
    def test_is_flagged_contradicted(self):
        assert SentenceAudit("t", "CONTRADICTED", 0.2, [1]).is_flagged is True
    def test_is_flagged_unsupported(self):
        assert SentenceAudit("t", "UNSUPPORTED", 0.0, []).is_flagged is True
    def test_not_flagged_supported(self):
        assert SentenceAudit("t", "SUPPORTED", 0.9, [1]).is_flagged is False


class TestAuditResultModel:
    def _res(self, statuses, sr=0.5, flagged=False):
        audits = [SentenceAudit(f"s{i}", st, 0.5, [1]) for i, st in enumerate(statuses)]
        return AuditResult(audits, flagged, sr, "answer")

    def test_supported_count(self):
        assert self._res(["SUPPORTED", "SUPPORTED", "CONTRADICTED"]).supported_count == 2
    def test_contradicted_count(self):
        assert self._res(["CONTRADICTED", "CONTRADICTED"]).contradicted_count == 2
    def test_unsupported_count(self):
        assert self._res(["UNSUPPORTED", "SUPPORTED"]).unsupported_count == 1
    def test_flag_reason_contradicted(self):
        r = self._res(["CONTRADICTED"], flagged=True)
        assert "contradicted" in r.flag_reason.lower()
    def test_flag_reason_low_support(self):
        r = AuditResult([], True, 0.2, "")
        assert "support" in r.flag_reason.lower()
    def test_flag_reason_empty_when_not_flagged(self):
        assert self._res(["SUPPORTED"], sr=1.0, flagged=False).flag_reason == ""


class TestSentenceSplitter:
    def test_basic_split(self):
        text = "PPF has 15 year lock-in. NPS allows withdrawal at 60. ELSS has 3 years."
        assert len(HallucinationAuditor.split_sentences(text)) == 3

    def test_question_mark_split(self):
        sents = HallucinationAuditor.split_sentences(
            "What is the PPF rate? The rate is 7.1 percent per annum.")
        assert len(sents) == 2

    def test_citation_before_next_sentence(self):
        text = "PPF rate is 7.1 percent [1]. NPS provides annuity at retirement [2]."
        sents = HallucinationAuditor.split_sentences(text)
        assert len(sents) == 2
        assert "[1]" in sents[0]

    def test_short_fragments_filtered(self):
        sents = HallucinationAuditor.split_sentences(
            "Ok. Yes. ELSS has a 3 year lock-in period under section 80C.")
        assert all(len(s) >= 15 for s in sents)

    def test_single_sentence_no_split(self):
        text = "PPF has a 15 year lock-in with partial withdrawal allowed from year 7."
        assert len(HallucinationAuditor.split_sentences(text)) == 1

    def test_empty_text(self):
        assert HallucinationAuditor.split_sentences("") == []


class TestExtractCitedMarkers:
    def test_single(self):
        assert HallucinationAuditor.extract_cited_markers("PPF rate is 7.1% [1].") == [1]
    def test_multiple(self):
        assert HallucinationAuditor.extract_cited_markers("A [1] B [2] C [3].") == [1, 2, 3]
    def test_none(self):
        assert HallucinationAuditor.extract_cited_markers("PPF rate is 7.1%.") == []
    def test_duplicate_raw(self):
        result = HallucinationAuditor.extract_cited_markers("A [1] B [1] C [1].")
        assert result == [1, 1, 1]  # raw extraction, dedup done in audit()
    def test_two_digit(self):
        assert HallucinationAuditor.extract_cited_markers("Source [12].") == [12]