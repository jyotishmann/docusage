# generation/citation_formatter.py
# Parses [N] citation markers from generated text into Citation objects.
from __future__ import annotations
import re
from config import get_logger
from retrieval import RankedChunk
from generation.models import Citation

logger = get_logger(__name__)

_CITATION_RE = re.compile(r'\[(\d+)\]')


class CitationFormatter:
    '''Parses [N] markers from generated answer. All methods static.'''

    @staticmethod
    def format(answer: str, context_chunks: list[RankedChunk]) -> list[Citation]:
        '''
        Extract [N] markers, validate, deduplicate, and map to chunks.

        Args:
            answer: Generated answer text with [N] inline citations.
            context_chunks: Chunks provided as context (1-indexed in prompt).

        Returns:
            List of unique Citation objects sorted by marker number.
        '''
        if not answer or not context_chunks:
            return []
        raw = _CITATION_RE.findall(answer)
        if not raw:
            return []

        seen: set[int] = set()
        citations: list[Citation] = []

        for s in raw:
            n = int(s)
            if n < 1 or n > len(context_chunks):
                continue
            if n in seen:
                continue
            seen.add(n)
            citations.append(Citation(marker=n, chunk=context_chunks[n - 1]))

        citations.sort(key=lambda c: c.marker)
        logger.debug('Citations extracted', raw=len(raw), unique=len(citations))
        return citations

    @staticmethod
    def count_markers(answer: str) -> int:
        '''Total [N] occurrences including duplicates.'''
        return len(_CITATION_RE.findall(answer))

    @staticmethod
    def has_uncited_claims(answer: str) -> bool:
        '''Heuristic: True if >50% of sentences lack a [N] marker.'''
        sentences = [s.strip() for s in re.split(r'[.!?]', answer) if s.strip()]
        if not sentences:
            return False
        uncited = sum(1 for s in sentences if not _CITATION_RE.search(s))
        return uncited > len(sentences) * 0.5