"""
Embeddings Module — Gemini Integration
Generates vector embeddings via the Gemini v1 REST API (httpx).
Model: text-embedding-004 (768 dimensions)
"""
from typing import List
import httpx
from . import config
from app.core.config import settings

BASE_URL = "https://generativelanguage.googleapis.com/v1"
_client = httpx.Client(timeout=60)

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a batch of texts using Gemini v1 API.
    """
    requests = [
        {
            "model": config.EMBEDDING_MODEL,
            "content": {"parts": [{"text": t}]},
        }
        for t in texts
    ]
    response = _client.post(
        f"{BASE_URL}/{config.EMBEDDING_MODEL}:batchEmbedContents",
        params={"key": settings.GEMINI_API_KEY},
        json={"requests": requests},
    )
    response.raise_for_status()
    data = response.json()
    return [e["values"] for e in data["embeddings"]]

def embed_query(query: str) -> List[float]:
    """
    Generate embedding for a single search query using Gemini v1 API.
    """
    response = _client.post(
        f"{BASE_URL}/{config.EMBEDDING_MODEL}:embedContent",
        params={"key": settings.GEMINI_API_KEY},
        json={
            "model": config.EMBEDDING_MODEL,
            "content": {"parts": [{"text": query}]},
        },
    )
    response.raise_for_status()
    data = response.json()
    return data["embedding"]["values"]