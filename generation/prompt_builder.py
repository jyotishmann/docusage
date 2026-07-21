# generation/prompt_builder.py
# Formats reranked chunks + query into ChatML prompt with [N] citation markers.
from __future__ import annotations
import re
from config import get_logger
from retrieval import RankedChunk

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful Indian personal finance assistant. "
    "Answer the question using ONLY the provided context documents. "
    "For each factual claim, cite the source with an inline [N] marker "
    "(e.g. 'PPF interest rate is 7.1% [1]') where N is the document number. "
    "If the context does not contain sufficient information, say so clearly. "
    "Be concise and accurate. Use bullet points for multi-part answers."
)


class PromptBuilder:
    '''Formats reranked chunks and query into a Qwen2.5-3B ChatML prompt.'''

    @staticmethod
    def build(query: str, context_chunks: list[RankedChunk]) -> str:
        context_block = PromptBuilder._format_context(context_chunks)
        user_message  = PromptBuilder._format_user(query, context_block)
        prompt = (
            f"<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
"
            f"<|im_start|>user
{user_message}<|im_end|>
"
            f"<|im_start|>assistant
"
        )
        logger.debug('Prompt built', chunks=len(context_chunks), chars=len(prompt))
        return prompt

    @staticmethod
    def _format_context(chunks: list[RankedChunk]) -> str:
        sections = []
        for i, chunk in enumerate(chunks, start=1):
            header = (f"[{i}] Title: {chunk.doc_title} | "
                      f"Authority: {chunk.governing_body} | "
                      f"Domain: {chunk.ring_label}")
            if chunk.effective_date:
                header += f" | Date: {chunk.effective_date}"
            if chunk.circular_ref:
                header += f" | Ref: {chunk.circular_ref}"
            sections.append(f"{header}
{chunk.chunk_text}")
        return "

".join(sections)

    @staticmethod
    def _format_user(query: str, context_block: str) -> str:
        return f"Context documents:

{context_block}

Question: {query}"

    @staticmethod
    def count_chunks_in_prompt(prompt: str) -> int:
        '''Count [N] Title: headers -- used in tests.'''
        return len(re.findall(r"\[\d+\] Title:", prompt))