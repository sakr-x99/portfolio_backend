"""
Embeddings Module — Gemini Integration
Generates vector embeddings using Google's Generative AI.
Model: text-embedding-004 (768 dimensions)
"""
from typing import List
from . import config
from app.core.config import settings

_genai = None

def _get_genai():
    global _genai
    if _genai is None:
        import google.generativeai as genai
        genai.configure(api_key=settings.GEMINI_API_KEY)
        _genai = genai
    return _genai

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a batch of texts using Gemini.
    """
    genai = _get_genai()
    result = genai.embed_content(
        model=config.EMBEDDING_MODEL,
        content=texts,
        task_type="retrieval_document"
    )
    return result['embedding']

def embed_query(query: str) -> List[float]:
    """
    Generate embedding for a single search query using Gemini.
    """
    genai = _get_genai()
    result = genai.embed_content(
        model=config.EMBEDDING_MODEL,
        content=query,
        task_type="retrieval_query"
    )
    return result['embedding']
