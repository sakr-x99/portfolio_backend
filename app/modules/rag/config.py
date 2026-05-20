"""
RAG Configuration
Centralized settings for the Retrieval-Augmented Generation pipeline.
"""
import os
from app.core.config import settings

# ── Vector Database ──────────────────────────────────────────────────────────
QDRANT_URL = settings.QDRANT_URL
QDRANT_API_KEY = settings.QDRANT_API_KEY
QDRANT_COLLECTION = settings.QDRANT_COLLECTION

# ── Embedding Model ──────────────────────────────────────────────────────────
# Using multilingual model for Arabic/English support via FastEmbed (local ONNX)
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
EMBEDDING_DIMENSIONS = 384

# ── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE = 1500         # Characters per chunk
CHUNK_OVERLAP = 200       # Overlap between chunks for context continuity

# ── Retrieval ────────────────────────────────────────────────────────────────
TOP_K = 3                 # Number of chunks to retrieve
SCORE_THRESHOLD = 0.5     # Minimum similarity score (0-1) — was 0.3, raised to reduce noise

# ── Knowledge Base ───────────────────────────────────────────────────────────
_knownledge_base = getattr(settings, "KNOWLEDGE_DIR", None)
if _knownledge_base:
    KNOWLEDGE_DIR = _knownledge_base
else:
    KNOWLEDGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_knowledge")
