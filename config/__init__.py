# config/__init__.py
# Re-exports the singleton Settings instance as the package's public API.
# Usage in other modules: from config import settings

from config.settings import Settings as _Settings # type: ignore

# Singleton: instantiate once, import everywhere
settings = _Settings()

__all__ = ["settings"]
