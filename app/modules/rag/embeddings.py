"""
Embeddings Module — FastEmbed (local ONNX embeddings)
Downloads model to cache on first use; no external API calls at inference time.
Supports Arabic + English via multilingual model.
"""
import logging
import os
from typing import List
from . import config

logger = logging.getLogger(__name__)

_client = None
_client_lock = None

def _get_lock():
    global _client_lock
    if _client_lock is None:
        from threading import Lock
        _client_lock = Lock()
    return _client_lock

def _get_cache_dir() -> str:
    """Return a writable cache directory for the embedding model."""
    _is_serverless = bool(os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"))
    if _is_serverless:
        cache_dir = os.path.join(os.environ.get("TMPDIR", "/tmp"), "fastembed_cache")
    else:
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".embedding_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

def _get_client():
    global _client
    if _client is None:
        with _get_lock():
            if _client is None:
                from fastembed import TextEmbedding
                _client = TextEmbedding(
                    model_name=config.EMBEDDING_MODEL,
                    cache_dir=_get_cache_dir(),
                )
                logger.info("Embedding model '%s' loaded", config.EMBEDDING_MODEL)
    return _client

def embed_texts(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """Generate embeddings for a list of texts with batching support."""
    client = _get_client()
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        results.extend(list(e) for e in client.embed(batch))
        logger.debug("Embedded batch %d/%d", i // batch_size + 1, (len(texts) + batch_size - 1) // batch_size)
    return results

def embed_query(query: str) -> List[float]:
    """Generate embedding for a single query string."""
    client = _get_client()
    return list(next(client.embed([query])))