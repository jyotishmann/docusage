#!/usr/bin/env python3
# scripts/build_index.py
# One-time: build BM25 + FAISS indices from corpus_registry.json.
# Usage: python scripts/build_index.py [--force] [--bm25-only] [--faiss-only]

from __future__ import annotations
import argparse, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_logger, settings
from corpus import CorpusRegistry
from indexing import BM25Indexer, EmbeddingEncoder, FAISSIndexer

logger = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Build BM25 + FAISS indices.")
    p.add_argument("--force",      action="store_true", help="Rebuild even if indices exist")
    p.add_argument("--bm25-only",  action="store_true", help="BM25 only (skip FAISS)")
    p.add_argument("--faiss-only", action="store_true", help="FAISS only (skip BM25)")
    return p.parse_args()


def main():
    args  = parse_args()
    t_all = time.time()

    print("📂 Loading corpus registry...")
    registry = CorpusRegistry.load()
    chunks   = registry.get_all()
    print(f"   {len(chunks)} chunks | rings: {registry.get_ring_counts()}\n")

    if len(chunks) == 0:
        print("❌ Empty registry. Run download_corpus.py first.")
        sys.exit(1)

    # ── BM25 ────────────────────────────────────────────────────────────────
    if not args.faiss_only:
        print("🔎 Building BM25 indices (combined + 4 ring sub-indices)...")
        t0    = time.time()
        paths = BM25Indexer(settings.INDEX_DIR).build(chunks=chunks, force=args.force)
        print(f"   ✅ {len(paths)} indices in {time.time()-t0:.1f}s")
        for k, v in paths.items():
            print(f"      {k}: {v}")
        print()

    # ── FAISS ────────────────────────────────────────────────────────────────
    if not args.bm25_only:
        print("🧠 Loading BGE-M3 embedding model (first run ~3min download)...")
        t0 = time.time()
        encoder = EmbeddingEncoder()
        encoder.load()
        print(f"   Loaded in {time.time()-t0:.1f}s\n")

        print(f"⚡ Encoding {len(chunks)} chunks...")
        t0   = time.time()
        embs = encoder.encode_chunks(chunks)
        elapsed = time.time() - t0
        print(f"   ✅ {elapsed:.1f}s ({len(chunks)/elapsed:.0f} chunks/sec)\n")

        print("🗂  Building FAISS IndexFlatIP...")
        t0 = time.time()
        ip, mp = FAISSIndexer(settings.INDEX_DIR).build(embs, chunks, force=args.force)
        print(f"   ✅ {time.time()-t0:.2f}s")
        print(f"      Index: {ip}")
        print(f"      ID map: {mp}\n")

    print(f"🎉 Total: {time.time()-t_all:.1f}s | Index dir: {settings.INDEX_DIR}")
    print("Next step: python app.py")


if __name__ == "__main__":
    main()