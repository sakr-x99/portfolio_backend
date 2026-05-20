"""
Embeddings Module — HuggingFace Inference API
Generates embeddings via free HuggingFace Inference API.
"""
import json
import urllib.request
from typing import List

HF_API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"

def embed_texts(texts: List[str]) -> List[List[float]]:
    data = json.dumps({"inputs": texts}).encode("utf-8")
    req = urllib.request.Request(HF_API_URL, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())

def embed_query(query: str) -> List[float]:
    data = json.dumps({"inputs": [query]}).encode("utf-8")
    req = urllib.request.Request(HF_API_URL, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())[0]