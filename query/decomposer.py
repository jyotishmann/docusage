# query/decomposer.py -- Part 1: Prompt template and DecomposerModel
from __future__ import annotations
from config import get_logger, settings
logger = get_logger(__name__)

DECOMPOSE_SYSTEM = (
    "You are a query decomposer for an Indian personal finance system.\n\n"
    "Knowledge base domains:\n"
    "- Domain 1 (Market Investments): stocks, mutual funds, ELSS, SIP\n"
    "- Domain 2 (Govt Schemes): PPF, NPS, EPFO, SGB, APY, SCSS\n"
    "- Domain 3 (Banking and RBI): fixed deposits, DICGC, KYC\n"
    "- Domain 4 (Foreign Investments): FEMA, LRS, DTAA, NRE/NRO\n\n"
    "Task: break query into 2-4 focused sub-questions, one domain each.\n"
    "Output: ONLY a valid JSON array of strings. No explanation.\n"
    "Example: [\"What are PPF tax benefits?\", \"What is LRS limit?\"]"
)
DECOMPOSE_USER = (
    "User query: {query}\n\n"
    "Respond ONLY with a JSON array of 2-4 sub-questions.\n"
    "JSON array:"
)


class DecomposerModel:
    """Loads Qwen2.5-1.5B-Instruct in fp16. Injected into QueryDecomposer."""
    def __init__(self, model_name=None, device=None):
        self.model_name = model_name or settings.DECOMPOSER_MODEL
        self.device     = device or settings.DEVICE
        self._tok = self._model = None

    def load(self):
        """Load fp16. First run downloads ~3GB (HF cache after)."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        logger.info("Loading decomposer", model=self.model_name)
        self._tok   = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=False)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.float16,
            device_map="auto", trust_remote_code=False)
        self._model.eval()
        logger.info("Decomposer model ready")
        return self

    def generate(self, prompt,
                 max_new_tokens=settings.DECOMPOSER_MAX_NEW_TOKENS,
                 temperature=settings.DECOMPOSER_TEMPERATURE):
        """Generate; returns only newly generated tokens."""
        if self._model is None: raise RuntimeError("Call .load() first")
        import torch
        inputs = self._tok(prompt, return_tensors="pt",
                           truncation=True, max_length=1024)
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        in_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = self._model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                temperature=temperature, do_sample=True, top_p=0.9,
                pad_token_id=self._tok.eos_token_id,
                eos_token_id=self._tok.eos_token_id,
            )
        return self._tok.decode(out[0][in_len:], skip_special_tokens=True).strip()
    
# query/decomposer.py -- Part 2: QueryDecomposer (append after Part 1)
import json, re

class QueryDecomposer:
    """Decomposes complex queries into 2-4 sub-queries via Qwen2.5-1.5B."""

    def __init__(self, model: DecomposerModel):
        self.model = model

    def decompose(self, query: str) -> list[str]:
        """Returns 2-4 sub-queries. Falls back to [original_query]."""
        if not query.strip(): return [query]
        prompt = self._build_prompt(query)
        try:
            raw = self.model.generate(prompt)
            logger.debug("Decomposer raw", output=raw[:200])
        except Exception as e:
            logger.error("Decomposer failed", error=str(e))
            return [query]
        return self._parse_and_validate(raw, query)

    def _build_prompt(self, query: str) -> str:
        return (
            f"<|im_start|>system\n{DECOMPOSE_SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{DECOMPOSE_USER.format(query=query)}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def _parse_and_validate(self, raw, orig):
        for fn in [self._direct, self._extract, self._normalise, self._strings]:
            parsed = fn(raw)
            if parsed is not None:
                valid = self._validate(parsed, orig)
                if valid:
                    logger.info("Decomposed", count=len(valid))
                    return valid
        logger.warning("Decomposer fallback", raw=raw[:80])
        return [orig]

    @staticmethod
    def _direct(t):
        try:
            r = json.loads(t.strip())
            return r if isinstance(r, list) else None
        except Exception: return None

    @staticmethod
    def _extract(t):
        m = re.search(r"\[.*?\]", t, re.DOTALL)
        if not m: return None
        try: return json.loads(m.group(0))
        except Exception: return None

    @staticmethod
    def _normalise(t):
        n = re.sub(r"'([^']*)'", r'"\1"', t)
        try:
            r = json.loads(n.strip())
            return r if isinstance(r, list) else None
        except Exception: pass
        m = re.search(r"\[.*?\]", n, re.DOTALL)
        if m:
            try: return json.loads(m.group(0))
            except Exception: pass
        return None

    @staticmethod
    def _strings(t):
        s = re.findall(r'"([^"]{10,})"', t)
        return s if len(s) >= 2 else None

    @staticmethod
    def _validate(parsed, original, min_i=2, max_i=4, min_c=10):
        if not isinstance(parsed, list): return None
        seen, out = set(), []
        for item in parsed:
            if not isinstance(item, str): continue
            item = item.strip()
            if len(item) < min_c: continue
            lo = item.lower()
            if lo in seen or lo == original.strip().lower(): continue
            seen.add(lo); out.append(item)
        if len(out) < min_i: return None
        return out[:max_i]