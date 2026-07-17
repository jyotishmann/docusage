# config/__init__.py
# Re-exports the singleton Settings instance as the package's public API.
# Usage in other modules: from config import settings

from config.settings import Settings as _Settings
from config.logger import configure_logging, get_logger

# Singleton: instantiate once, import everywhere
settings = _Settings()

# ── Configure logging on first import ─────────────────────────────────────
# This runs once. Subsequent imports of config hit the module cache.
configure_logging(log_level=settings.LOG_LEVEL)

# ── Ensure data directories exist ─────────────────────────────────────────
settings.ensure_dirs()

__all__ = ["settings", "get_logger"]

