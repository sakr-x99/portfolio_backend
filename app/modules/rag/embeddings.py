"""
Embeddings Module — Gemini Integration
Generates vector embeddings using Google's GenAI SDK (google.genai).
Model: text-embedding-004 (768 dimensions)
"""
from typing import List
from . import config
from app.core.config import settings

_client = None

def _get_client():
    global _client
    if _client is None:
        from google import genai
        _client = genai.Client(api_key=settings.GEMINI_API_KEY, api_version="v1")
    return _client

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a batch of texts using Gemini.
    """
    client = _get_client()
    result = client.models.embed_content(
        model=config.EMBEDDING_MODEL,
        contents=texts,
    )
    return [e.values for e in result.embeddings]

def embed_query(query: str) -> List[float]:
    """
    Generate embedding for a single search query using Gemini.
    """
    client = _get_client()
    result = client.models.embed_content(
        model=config.EMBEDDING_MODEL,
        contents=query,
    )
    return result.embeddings[0].values
