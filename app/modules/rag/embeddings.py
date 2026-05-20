"""
Embeddings Module — HuggingFace Inference API
Generates embeddings via free HuggingFace Inference API (no API key, no model download).
"""
from typing import List
import httpx

HF_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

def embed_texts(texts: List[str]) -> List[List[float]]:
    resp = httpx.post(
        HF_API_URL,
        json={"inputs": texts, "options": {"wait_for_model": True}},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()

def embed_query(query: str) -> List[float]:
    resp = httpx.post(
        HF_API_URL,
        json={"inputs": [query], "options": {"wait_for_model": True}},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()[0]