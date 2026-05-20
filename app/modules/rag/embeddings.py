"""
Embeddings Module — FastEmbed (ONNX-based, local, no external API)
Uses sentence-transformers models via Qdrant's FastEmbed (no PyTorch needed).
"""
from typing import List
from . import config

_client = None

def _get_client():
    global _client
    if _client is None:
        from fastembed import TextEmbedding
        _client = TextEmbedding(model_name=config.EMBEDDING_MODEL)
    return _client

def embed_texts(texts: List[str]) -> List[List[float]]:
    client = _get_client()
    return [list(e) for e in client.embed(texts)]

def embed_query(query: str) -> List[float]:
    client = _get_client()
    return list(next(client.embed([query])))