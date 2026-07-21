#!/usr/bin/env python3
# scripts/download_corpus.py
# One-time corpus build: download → chunk → tag → save registry.
# Usage: python scripts/download_corpus.py [--rings 1 2 3 4] [--dry-run]

from __future__ import annotations
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm
from config import get_logger, settings
from corpus import (
    ALL_SOURCES, SOURCES_BY_RING, CorpusRegistry,
    DocumentChunker, MetadataTagger, SourceDocument,
)
from corpus.downloader import HTMLDownloader, PDFDownloader

logger = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Download and chunk DocuSage corpus.")
    p.add_argument("--rings", nargs="+", type=int, choices=[1,2,3,4],
                   default=[1,2,3,4])
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def download_source(
    doc: SourceDocument,
    pdf_dl: PDFDownloader,
    html_dl: HTMLDownloader,
) -> list[SourceDocument]:
    """Download one source entry; expands html_module into chapter pages."""
    try:
        urls = html_dl.get_chapter_links(doc.source_url)
        # ... rest unchanged
    except Exception as e:
        # logger.warning("Skipping source entirely", source=doc.doc_title, error=str(e))
        import traceback
        logger.warning(
            "Skipping source entirely",
            source=doc.title,
            url=doc.source_url,
            error=str(e),
        )
        print(f"FAILED: {doc.title}")
        print(f"   URL: {doc.source_url}")
        print(f" Error: {e}")
        traceback.print_exc()
        # logger.warning("Skipping source entirely", source=doc.title, error=str(e))
        return []
    if doc.file_format == "html_module":
        urls = html_dl.get_chapter_links(doc.source_url)
        results = []
        for url in urls:
            ch = SourceDocument(
                doc_id=SourceDocument.make_doc_id(url),
                title=f"{doc.title} — {url.split('/')[-2].replace('-',' ').title()}",
                source_url=url, governing_body=doc.governing_body,
                ring=doc.ring, ring_label=doc.ring_label,
                effective_date=doc.effective_date, file_format="html",
            )
            results.append(html_dl.download(ch))
        return results
    elif doc.file_format == "pdf":
        return [pdf_dl.download(doc)]
    else:
        return [html_dl.download(doc)]


def main():
    args = parse_args()
    sources = []
    for ring in args.rings:
        sources.extend(SOURCES_BY_RING.get(ring, []))

    if args.dry_run:
        for doc in sources:
            print(f"  [Ring {doc.ring}] {doc.title}")
        print(f"\nTotal entries: {len(sources)} (html_modules expand further)")
        return

    pdf_dl   = PDFDownloader(output_dir=settings.RAW_DOCS_DIR)
    html_dl  = HTMLDownloader(output_dir=settings.RAW_DOCS_DIR)
    chunker  = DocumentChunker()
    tagger   = MetadataTagger()
    registry = CorpusRegistry()

    downloaded: list[SourceDocument] = []
    for doc in tqdm(sources, desc="Downloading", unit="source"):
        downloaded.extend(download_source(doc, pdf_dl, html_dl))

    valid = [d for d in downloaded if d.raw_text.strip()]
    logger.info("Download complete", valid=len(valid), failed=len(downloaded)-len(valid))

    all_chunks = chunker.chunk_corpus(valid)
    tagger.tag_corpus(all_chunks)
    registry.add_chunks(all_chunks)
    registry.save()

    print(f"\n✅ {len(all_chunks)} chunks → {settings.CORPUS_REGISTRY_PATH}")
    print("Next: python scripts/build_index.py")


if __name__ == "__main__":
    main()