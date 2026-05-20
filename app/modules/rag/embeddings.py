"""
Embeddings Module — HuggingFace Inference API
Generates embeddings via free HuggingFace Inference API.
"""
import json
import urllib.request
from typing import List

HF_API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"

HEADERS = {"Content-Type": "application/json", "User-Agent": "Portfolio-RAG/1.0"}

def _call_hf(texts) -> list:
    data = json.dumps({"inputs": texts, "options": {"wait_for_model": True}}).encode()
    req = urllib.request.Request(HF_API_URL, data=data, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())

def embed_texts(texts: List[str]) -> List[List[float]]:
    return _call_hf(texts)

def embed_query(query: str) -> List[float]:
    return _call_hf([query])[0]