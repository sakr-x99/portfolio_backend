"""
RAG Configuration
Centralized settings for the Retrieval-Augmented Generation pipeline.
"""
import os
import tempfile
from app.core.config import settings

# ── Vector Database ──────────────────────────────────────────────────────────
QDRANT_URL = settings.QDRANT_URL
QDRANT_API_KEY = settings.QDRANT_API_KEY
QDRANT_COLLECTION = settings.QDRANT_COLLECTION

# ── Embedding Model ──────────────────────────────────────────────────────────
# Multilingual model supported by FastEmbed (local ONNX, no API key needed)
# Supports Arabic + English + 50+ languages, 384 dimensions
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIMENSIONS = 384

# ── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE = 1500         # Characters per chunk
CHUNK_OVERLAP = 200       # Overlap between chunks for context continuity

# ── Retrieval ────────────────────────────────────────────────────────────────
TOP_K = 3                 # Number of chunks to retrieve
SCORE_THRESHOLD = 0.35    # Minimum similarity score — lowered for cross-lingual (Arabic query → English content)

# ── Knowledge Base ───────────────────────────────────────────────────────────
# On Vercel/serverless, only /tmp/ is writable. On local, use a persistent dir.
_is_serverless = bool(os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"))

_knownledge_base = getattr(settings, "KNOWLEDGE_DIR", None)
if _knownledge_base:
    KNOWLEDGE_DIR = _knownledge_base
elif _is_serverless:
    KNOWLEDGE_DIR = os.path.join(tempfile.gettempdir(), "sakr_portfolio_knowledge")
else:
    KNOWLEDGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_knowledge")
