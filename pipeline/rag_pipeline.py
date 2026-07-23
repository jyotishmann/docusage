# pipeline/rag_pipeline.py -- Part 1: Construction and lazy model loading
from __future__ import annotations
import time
import threading

from config import get_logger, settings as default_settings, Settings
from corpus import CorpusRegistry
from indexing import BM25IndexLoader, FAISSIndexLoader
from retrieval import BM25Retriever, DenseRetriever, HybridRetriever
from query import QueryRouter, QueryDecomposer, DecomposerModel
from generation import Reranker, Generator
from audit import HallucinationAuditor
from pipeline.models import PipelineResult

logger = get_logger(__name__)


class RAGPipeline:
    # Full DocuSage RAG pipeline orchestrator.

    def __init__(self, cfg: Settings | None = None):
        self.cfg     = cfg or default_settings
        self._loaded = False
        self._lock   = threading.Lock()

        # Eager: load indices (fast, no VRAM)
        logger.info("Loading corpus registry and indices")
        self.registry  = CorpusRegistry.load(self.cfg.REGISTRY_PATH)
        bm25_loader    = BM25IndexLoader(self.cfg.INDEX_DIR).load_all()
        faiss_loader   = FAISSIndexLoader(self.cfg.INDEX_DIR).load_all()
        logger.info("Indices loaded", chunks=len(self.registry))

        # Stateless component wired immediately
        self.router = QueryRouter(
            max_words_simple=self.cfg.ROUTER_MAX_WORDS_SIMPLE,
            min_words_complex=self.cfg.ROUTER_MIN_WORDS_COMPLEX,
        )

        # Store loaders for model wiring in load()
        self._bm25_loader  = bm25_loader
        self._faiss_loader = faiss_loader

        # Placeholders -- populated in load()
        self.decomposer: QueryDecomposer      | None = None
        self.retriever:  HybridRetriever      | None = None
        self.reranker:   Reranker             | None = None
        self.generator:  Generator            | None = None
        self.auditor:    HallucinationAuditor | None = None

    def load(self) -> "RAGPipeline":
        # Load all model weights. Thread-safe via _lock.
        with self._lock:
            if self._loaded:
                return self
            t0 = time.perf_counter()
            logger.info("Loading all models...")

            # 1. BGE-M3 (shared -- load first for largest contiguous VRAM block)
            from sentence_transformers import SentenceTransformer
            dense_model = SentenceTransformer(
                self.cfg.EMBED_MODEL, device=self.cfg.DEVICE)
            logger.info("BGE-M3 loaded")

            # 2. Decomposer (fp16 Qwen2.5-1.5B)
            self.decomposer = QueryDecomposer(
                DecomposerModel(self.cfg.DECOMPOSER_MODEL).load())
            logger.info("Decomposer loaded")

            # 3. Wire retrievers
            self.retriever = HybridRetriever(
                bm25_retriever=BM25Retriever(self._bm25_loader, self.registry),
                dense_retriever=DenseRetriever(
                    self._faiss_loader, self.registry, dense_model),
            )
            logger.info("HybridRetriever wired")

            # 4. Reranker
            self.reranker = Reranker(
                model_name=self.cfg.RERANKER_MODEL,
                top_k=self.cfg.GENERATOR_CONTEXT_K,
            ).load()
            logger.info("Reranker loaded")

            # 5. Generator (4-bit NF4 Qwen2.5-3B)
            self.generator = Generator(self.cfg.GENERATOR_MODEL).load()
            logger.info("Generator loaded")

            # 6. Auditor (DeBERTa NLI)
            self.auditor = HallucinationAuditor(
                model_name=self.cfg.AUDITOR_MODEL,
                entailment_threshold=self.cfg.AUDITOR_ENTAILMENT_THRESHOLD,
                flag_threshold=self.cfg.AUDITOR_FLAG_THRESHOLD,
            ).load()
            logger.info("Auditor loaded")

            self._loaded = True
            logger.info("All models ready",
                        total_load_ms=round((time.perf_counter()-t0)*1000))
        return self

    def load_once(self) -> "RAGPipeline":
        # Idempotent: load only if not already loaded.
        if not self._loaded:
            self.load()
        return self