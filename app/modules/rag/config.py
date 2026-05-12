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
# Using Gemini embedding-001
EMBEDDING_MODEL = "models/gemini-embedding-2"
EMBEDDING_DIMENSIONS = 3072

# ── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE = 500          # Characters per chunk
CHUNK_OVERLAP = 50        # Overlap between chunks for context continuity

# ── Retrieval ────────────────────────────────────────────────────────────────
TOP_K = 5                 # Number of chunks to retrieve
SCORE_THRESHOLD = 0.3     # Minimum similarity score (0-1)

# ── Knowledge Base ───────────────────────────────────────────────────────────
import os
import tempfile
KNOWLEDGE_DIR = os.path.join(tempfile.gettempdir(), "sakr_portfolio_knowledge")
