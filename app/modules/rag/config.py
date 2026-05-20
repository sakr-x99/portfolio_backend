"""
RAG Configuration
Centralized settings for the Retrieval-Augmented Generation pipeline.
"""
from app.core.config import settings

# ── Vector Database ──────────────────────────────────────────────────────────
QDRANT_URL = settings.QDRANT_URL
QDRANT_API_KEY = settings.QDRANT_API_KEY
QDRANT_COLLECTION = settings.QDRANT_COLLECTION

# ── Embedding Model ──────────────────────────────────────────────────────────
# Using BAAI/bge-small-en-v1.5 via FastEmbed (local ONNX, no API key needed)
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSIONS = 384

# ── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE = 1500         # Characters per chunk
CHUNK_OVERLAP = 200       # Overlap between chunks for context continuity

# ── Retrieval ────────────────────────────────────────────────────────────────
TOP_K = 3                 # Number of chunks to retrieve
SCORE_THRESHOLD = 0.3     # Minimum similarity score (0-1)

# ── Knowledge Base ───────────────────────────────────────────────────────────
import os
import tempfile
KNOWLEDGE_DIR = os.path.join(tempfile.gettempdir(), "sakr_portfolio_knowledge")
