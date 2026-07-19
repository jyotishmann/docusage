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