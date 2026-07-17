# 🏦 DocuSage — Indian Personal Finance Intelligence

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![HF Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/<YOUR_USERNAME>/docusage)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A production-grade RAG pipeline that answers Indian personal finance
> questions — grounded in authoritative sources, with citations and a
> hallucination audit layer.

---

## The Problem

500 million Indians navigate a complex financial landscape spanning equity
markets, government savings schemes (PPF, NPS, EPFO), RBI banking regulations,
and FEMA foreign investment rules. The information is scattered across dozens
of official portals, written in dense regulatory language, and frequently
misrepresented in SEO-optimised blogs.

## What DocuSage Does

DocuSage retrieves answers from authoritative sources (Zerodha Varsity,
SEBI, PFRDA, RBI, FEMA), grounds every claim in a cited source chunk, and
independently verifies that the generated answer is entailed by the retrieved
context.

## Live Demo

👉 [Try on Hugging Face Spaces](https://huggingface.co/spaces/<YOUR_USERNAME>/docusage)

---

## Architecture

Query → Router → Decomposer (Qwen2.5-1.5B)
 → Hybrid Retrieval (BM25 + FAISS/BGE-M3)
 → RRF Fusion
 → Cross-Encoder Reranker (BGE-reranker-v2-m3)
 → Generator (Qwen2.5-3B, 4-bit)
 → Hallucination Audit (NLI DeBERTa-v3)
 → Gradio UI

 **Four corpus rings:**
1. Market Investments (Zerodha Varsity, SEBI)
2. Government Schemes (PPF, NPS, EPFO, SGB)
3. Banking & RBI (DICGC, Retail Direct, Master Directions)
4. Foreign Investments (FEMA, LRS, DTAA, NRI accounts)

## Tech Stack

| Component | Model / Library |
|---|---|
| Embeddings | `BAAI/bge-m3` (1024-dim) |
| Sparse retrieval | `rank-bm25` BM25Okapi |
| Vector DB | `faiss-cpu` IndexFlatIP |
| Reranker | `BAAI/bge-reranker-v2-m3` |
| Query decomposer | `Qwen/Qwen2.5-1.5B-Instruct` |
| Generator | `Qwen/Qwen2.5-3B-Instruct` (4-bit NF4) |
| Hallucination audit | `cross-encoder/nli-deberta-v3-small` |
| Frontend | Gradio 4.x |
| Deployment | Hugging Face Spaces |

---

## Quick Start

```bash
git clone https://github.com/<YOUR_USERNAME>/docusage.git
cd docusage
pip install -r requirements.txt
cp .env.example .env          # Add your HF_TOKEN
python scripts/download_corpus.py   # ~10 min, downloads ~550 pages
python scripts/build_index.py       # ~20 min on T4 GPU
python app.py                        # Launches Gradio on localhost:7860
```

Project Structure

```bash
docusage/
├── config/         Settings (Pydantic BaseSettings)
├── corpus/         Document ingestion + chunking
├── indexing/       BM25 + FAISS index builders
├── retrieval/      BM25 + dense + hybrid + RRF
├── reranking/      Cross-encoder reranker
├── query/          Router + decomposer
├── generation/     Prompt builder + LLM + citation formatter
├── audit/          Hallucination auditor
├── pipeline/       End-to-end orchestrator
├── frontend/       Gradio UI
└── scripts/        One-time corpus download + index build
```
