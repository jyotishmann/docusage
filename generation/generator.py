# generation/generator.py
# 4-bit NF4 Qwen2.5-3B-Instruct for grounded answer generation.
from __future__ import annotations
import time
from config import get_logger, settings
from retrieval import RankedChunk
from generation.models import GenerationResult
from generation.citation_formatter import CitationFormatter
from generation.prompt_builder import PromptBuilder

logger = get_logger(__name__)


class Generator:
    '''4-bit NF4 Qwen2.5-3B-Instruct. Accepts context chunks, returns GenerationResult.'''

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.GENERATOR_MODEL
        self._tok = self._model = None

    def load(self) -> 'Generator':
        '''Load in 4-bit NF4. ~2GB VRAM after quantisation.'''
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch
        logger.info('Loading generator', model=self.model_name)
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        self._tok   = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=False)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, quantization_config=bnb,
            device_map='auto', trust_remote_code=False)
        self._model.eval()
        logger.info('Generator model ready')
        return self

    def generate(
        self,
        query: str,
        context_chunks: list[RankedChunk],
        sub_queries: list[str] | None = None,
        max_new_tokens: int = settings.GENERATOR_MAX_NEW_TOKENS,
        temperature: float  = settings.GENERATOR_TEMPERATURE,
    ) -> GenerationResult:
        '''
        Generate a grounded answer from context chunks.

        Args:
            query: Original user query.
            context_chunks: Top-K reranked RankedChunks (typically top-5).
            sub_queries: Sub-queries from QueryDecomposer (for display).
            max_new_tokens: Max tokens to generate (default 512).
            temperature: Sampling temperature (default 0.1 = near-deterministic).

        Returns:
            GenerationResult with answer text and structured Citation list.
        '''
        if self._model is None:
            raise RuntimeError('Call .load() before .generate()')

        import torch
        t0     = time.perf_counter()
        prompt = PromptBuilder.build(query, context_chunks)
        inputs = self._tok(prompt, return_tensors='pt',
                           truncation=True, max_length=4096)
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        in_len = inputs['input_ids'].shape[1]

        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=(temperature > 0.0),
                top_p=0.9,
                pad_token_id=self._tok.eos_token_id,
                eos_token_id=self._tok.eos_token_id,
                repetition_penalty=1.1,
            )

        answer     = self._tok.decode(out[0][in_len:], skip_special_tokens=True).strip()
        n_tokens   = len(out[0]) - in_len
        latency_ms = (time.perf_counter() - t0) * 1000

        logger.info('Generation complete', tokens=n_tokens,
                    latency_ms=round(latency_ms), chars=len(answer))

        citations = CitationFormatter.format(answer, context_chunks)

        return GenerationResult(
            answer=answer, citations=citations, query=query,
            sub_queries=sub_queries or [], context_chunks=context_chunks,
            tokens_generated=n_tokens, latency_ms=latency_ms,
        )