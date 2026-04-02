"""Embedder factory — Gemini backend only.

Provides backward-compatible top-level functions (embed_video_chunk,
embed_query) that delegate to the Gemini backend.
Re-exports error classes from gemini_embedder for existing import sites.
"""

from .base_embedder import BaseEmbedder
from .gemini_embedder import GeminiAPIKeyError, GeminiQuotaError  # noqa: F401

_current_embedder: BaseEmbedder | None = None


def get_embedder(backend: str = "gemini", **kwargs) -> BaseEmbedder:
    """Get or create the active embedder (Gemini only)."""
    global _current_embedder
    if _current_embedder is None:
        if backend != "gemini":
            raise ValueError(f"Only 'gemini' backend is supported, got: {backend}")
        from .gemini_embedder import GeminiEmbedder

        _current_embedder = GeminiEmbedder()
    return _current_embedder


def reset_embedder():
    """Reset the cached embedder."""
    global _current_embedder
    _current_embedder = None


# Convenience functions — backward compatible
def embed_video_chunk(chunk_path: str, verbose: bool = False) -> list[float]:
    return get_embedder().embed_video_chunk(chunk_path, verbose=verbose)


def embed_query(query_text: str, verbose: bool = False) -> list[float]:
    return get_embedder().embed_query(query_text, verbose=verbose)
