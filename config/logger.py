# config/logger.py
# Central loguru configuration for DocuSage.
# Import pattern in every module:
#   from config.logger import get_logger
#   logger = get_logger(__name__)

import sys
from pathlib import Path

from loguru import logger as _loguru_logger

# Track whether logger has already been configured (module-level singleton guard)
_configured = False


def configure_logging(
    log_level: str = "INFO",
    log_file: Path | None = None,
    json_mode: bool = False,
) -> None:
    """
    Configure loguru handlers. Should be called once at application startup.
    Subsequent calls are no-ops (idempotent via _configured guard).
    """
    global _configured
    if _configured:
        return

    # Remove loguru's default handler so we fully control the format
    _loguru_logger.remove()

    # ── Console handler ────────────────────────────────────────────────────
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    _loguru_logger.add(
        sys.stdout,
        level=log_level,
        format=console_format if not json_mode else "{message}",
        colorize=not json_mode,
        serialize=json_mode,   # JSON mode for structured log consumers
    )

    # ── File handler (optional, for production deployment) ─────────────────
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        _loguru_logger.add(
            str(log_file),
            level=log_level,
            rotation="1 day",    # New file every day
            retention="7 days",  # Keep last 7 days
            compression="zip",   # Compress old logs
            serialize=json_mode,
        )

    _configured = True


def get_logger(name: str):
    """
    Return a loguru logger bound with the caller's module name.
    Usage:
        from config.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Retrieval completed", chunks=5)
    """
    # loguru bind() attaches key-value context to all messages from this logger
    return _loguru_logger.bind(module=name)  # module name in every log line