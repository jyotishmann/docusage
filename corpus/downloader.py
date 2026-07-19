# corpus/downloader.py — Part 1: BaseDownloader with retry + rate limiting
# Abstract base class. Concrete subclasses implement parse_to_text().

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import get_logger, settings
from corpus.models import SourceDocument

logger = get_logger(__name__)


def _build_session(max_retries: int = 3, backoff_factor: float = 1.0) -> requests.Session:
    """
    Build a requests.Session with automatic retry on 5xx and connection errors.
    backoff_factor=1.0: retries at 1s, 2s, 4s (exponential).
    """
    session = requests.Session()
    retry = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,           # exponential backoff
        status_forcelist=[500, 502, 503, 504],   # retry on these HTTP codes
        allowed_methods=["GET", "HEAD"],          # only safe methods
        raise_on_status=False,                   # we handle status manually
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)  # apply to all HTTPS requests
    session.mount("http://", adapter)
    session.headers.update({
        "User-Agent": "DocuSage-FinanceRAG/1.0 (research; contact via GitHub)"
    })
    return session


class BaseDownloader(ABC):
    """
    Abstract base for all corpus downloaders.
    Handles HTTP fetching, retry, rate limiting, and idempotent saves.
    Subclasses implement parse_to_text() for their file format.
    """

    def __init__(
        self,
        output_dir: Path | None = None,
        delay_seconds: float = 1.5,
        max_retries: int = 3,
    ):
        self.output_dir = Path(output_dir or settings.RAW_DOCS_DIR)
        self.delay_secs = delay_seconds    # polite delay between requests
        self.session    = _build_session(max_retries=max_retries)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download(self, doc: SourceDocument) -> SourceDocument:
        """
        Fetch the document at doc.source_url, save to disk, parse text.
        Populates doc.raw_text and doc.file_path in-place.
        Returns the updated SourceDocument.
        """
        target_path = self._target_path(doc)

        # ── Idempotent: skip if already downloaded ─────────────────────────
        if target_path.exists():
            logger.debug("Skip download (already exists)", path=str(target_path))
            doc.file_path = str(target_path)
            doc.raw_text  = self._read_cached(target_path)
            return doc

        logger.info("Downloading", title=doc.title, url=doc.source_url)

        # ── Fetch with retry ───────────────────────────────────────────────
        try:
            response = self.session.get(
                doc.source_url,
                timeout=30,
                stream=False,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.error("Download failed", url=doc.source_url, error=str(exc))
            return doc  # return doc with empty raw_text; caller handles it

        # ── Validate response body ─────────────────────────────────────────
        if not self._validate_response(response):
            logger.warning("Invalid response body", url=doc.source_url)
            return doc

        # ── Save raw bytes to disk ─────────────────────────────────────────
        target_path.write_bytes(response.content)
        doc.file_path = str(target_path)
        logger.info("Saved", path=str(target_path), bytes=len(response.content))

        # ── Parse to plain text ────────────────────────────────────────────
        try:
            doc.raw_text = self.parse_to_text(target_path, response.content)
            logger.debug("Parsed text", chars=len(doc.raw_text))
        except Exception as exc:
            logger.error("Parse failed", path=str(target_path), error=str(exc))

        # ── Polite rate limiting ───────────────────────────────────────────
        time.sleep(self.delay_secs)  # be kind to government servers

        return doc

    def _target_path(self, doc: SourceDocument) -> Path:
        """Derive a safe local filename from the document title and doc_id."""
        safe_title = "".join(
            c if c.isalnum() or c in "-_ " else "_"
            for c in doc.title[:60]
        ).strip().replace(" ", "_")
        ext = ".pdf" if doc.file_format == "pdf" else ".html"
        return self.output_dir / f"{safe_title}_{doc.doc_id[:8]}{ext}"

    def _read_cached(self, path: Path) -> str:
        """Read text from a cached file by re-parsing it."""
        try:
            return self.parse_to_text(path, path.read_bytes())
        except Exception as exc:
            logger.warning("Re-parse of cached file failed", path=str(path), error=str(exc))
            return ""

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Normalise whitespace in extracted text.
        Collapses space runs, limits consecutive newlines to 2.
        """
        lines = text.splitlines()
        cleaned = [re.sub(r"[ \t]+", " ", line).strip() for line in lines]
        text = "\n".join(cleaned)
        text = re.sub(r"\n{3,}", "\n\n", text)  # max 2 consecutive newlines
        return text.strip()

    @abstractmethod
    def _validate_response(self, response: requests.Response) -> bool:
        """Return True if response.content is a valid file of this type."""

    @abstractmethod
    def parse_to_text(self, file_path: Path, content: bytes) -> str:
        """Extract plain text from raw file content."""

# corpus/downloader.py — Part 2: Concrete PDF and HTML downloaders
# Append these classes to corpus/downloader.py after BaseDownloader.

import pdfplumber
import requests
from bs4 import BeautifulSoup


class PDFDownloader(BaseDownloader):
    """Downloads and parses PDF documents (govt circulars, scheme brochures)."""

    def _validate_response(self, response: requests.Response) -> bool:
        """Check for PDF magic bytes at the start of the response."""
        return response.content[:4] == b"%PDF"  # standard PDF file signature

    def parse_to_text(self, file_path: Path, content: bytes) -> str:
        """
        Extract text page-by-page using pdfplumber.
        Per-page try/except ensures one corrupt page does not fail the document.
        """
        if not file_path.exists():
            file_path.write_bytes(content)

        pages_text: list[str] = []
        with pdfplumber.open(str(file_path)) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                try:
                    text = page.extract_text() or ""
                    text = self.clean_text(text)
                    if len(text) < 50:
                        logger.warning(
                            "Low text on page (may be scanned image)",
                            file=file_path.name, page=page_num, chars=len(text),
                        )
                    if text:
                        pages_text.append(text)
                except Exception as exc:
                    logger.error(
                        "Failed to extract page",
                        file=file_path.name, page=page_num, error=str(exc),
                    )
        return "\n\n".join(pages_text)  # double newline separates pages


class HTMLDownloader(BaseDownloader):
    """Downloads and parses HTML documents (Zerodha Varsity, govt web pages)."""

    # CSS selectors tried in priority order to find the main article body
    CONTENT_SELECTORS = [
        "article",
        "main",
        ".chapter-content",    # Zerodha Varsity chapter body class
        ".entry-content",
        ".post-content",
        "#content",
        ".content",
    ]

    # Tags stripped before text extraction (navigation noise)
    STRIP_TAGS = [
        "script", "style", "nav", "header", "footer",
        "aside", "form", "button",
    ]

    def _validate_response(self, response: requests.Response) -> bool:
        """Check that response looks like HTML."""
        snippet = response.content[:500].lower()
        return b"<html" in snippet or b"<!doctype" in snippet

    def parse_to_text(self, file_path: Path, content: bytes) -> str:
        """Extract article body text, stripping navigation and boilerplate."""
        soup = BeautifulSoup(content, "lxml")

        # Strip noise elements first
        for tag in self.STRIP_TAGS:
            for element in soup.select(tag):
                element.decompose()  # remove from parse tree

        # Find article body using priority selector list
        body = None
        for selector in self.CONTENT_SELECTORS:
            body = soup.select_one(selector)
            if body:
                break
        if body is None:
            body = soup.find("body") or soup  # final fallback

        return self.clean_text(body.get_text(separator="\n"))

    def get_chapter_links(self, module_url: str) -> list[str]:
        """
        Scrape a Zerodha Varsity module index page and return chapter URLs.
        Each module URL contains links to individual chapter pages.
        """
        response = self.session.get(module_url, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "lxml")

        chapter_links: list[str] = []
        for link in soup.select("a[href*='/chapter/']"):
            href = link.get("href", "")
            if href.startswith("/"):
                href = f"https://zerodha.com{href}"  # make absolute
            if href and href not in chapter_links:
                chapter_links.append(href)

        logger.info("Found chapters", module_url=module_url, count=len(chapter_links))
        return chapter_links
