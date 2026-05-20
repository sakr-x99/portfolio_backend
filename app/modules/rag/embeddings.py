"""
Embeddings Module — FastEmbed (local ONNX embeddings)
Downloads model to /tmp on first use; no external API calls at inference time.
"""
from typing import List
from . import config

_client = None

def _get_client():
    global _client
    if _client is None:
        from fastembed import TextEmbedding
        _client = TextEmbedding(
            model_name=config.EMBEDDING_MODEL,
            cache_dir="/tmp/fastembed_cache",
        )
    return _client

def embed_texts(texts: List[str]) -> List[List[float]]:
    client = _get_client()
    return [list(e) for e in client.embed(texts)]

def embed_query(query: str) -> List[float]:
    client = _get_client()
    return list(next(client.embed([query])))