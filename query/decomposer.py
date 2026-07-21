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