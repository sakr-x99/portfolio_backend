"""
Embeddings Module — Gemini Integration
Generates vector embeddings using Google's Generative AI.
Model: gemini-embedding-2 (768 dimensions)
"""
import google.generativeai as genai
from typing import List
from . import config
from app.core.config import settings

def _ensure_configured():
    """Ensure Gemini is configured."""
    genai.configure(api_key=settings.GEMINI_API_KEY)

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a batch of texts using Gemini.
    """
    _ensure_configured()
    # Gemini allows batching
    result = genai.embed_content(
        model="models/gemini-embedding-2",
        content=texts,
        task_type="retrieval_document"
    )
    return result['embedding']

def embed_query(query: str) -> List[float]:
    """
    Generate embedding for a single search query using Gemini.
    """
    _ensure_configured()
    result = genai.embed_content(
        model="models/gemini-embedding-2",
        content=query,
        task_type="retrieval_query"
    )
    return result['embedding']
