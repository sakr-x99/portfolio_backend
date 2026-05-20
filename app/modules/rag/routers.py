"""
RAG API Endpoints
Provides HTTP endpoints for the RAG pipeline: chat, indexing, and health checks.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from . import pipeline, vector_store

router = APIRouter()


# ── Schemas ──────────────────────────────────────────────────────────────────

class RAGChatMessage(BaseModel):
    role: str
    content: str

class RAGChatRequest(BaseModel):
    messages: List[RAGChatMessage]

class RAGChatResponse(BaseModel):
    content: str
    sources: List[str] = []
    chunks_used: int = 0

class IndexResponse(BaseModel):
    status: str
    files_generated: int = 0
    chunks_created: int = 0
    embeddings_generated: int = 0
    points_indexed: int = 0
    message: str = ""


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/chat", response_model=RAGChatResponse)
async def rag_chat(request: RAGChatRequest):
    """
    RAG chat endpoint — delegates to the unified AI chat system.
    """
    from app.modules.ai.routers import _rag_chat
    try:
        if not request.messages:
            raise HTTPException(status_code=400, detail="No messages provided")
        content = await _rag_chat(request)
        return RAGChatResponse(content=content)
    except HTTPException:
        raise
    except Exception as e:
        print(f"RAG Chat Error: {e}")
        raise HTTPException(status_code=500, detail=f"RAG pipeline error: {str(e)}")


@router.post("/index", response_model=IndexResponse)
async def index_knowledge():
    """
    Trigger full knowledge re-indexing (runs synchronously).
    Extracts data from DB → generates Markdown → chunks → embeds → indexes in Qdrant.
    NOTE: Runs synchronously because Vercel serverless kills background tasks.
    """
    try:
        result = pipeline.index_knowledge()
        return IndexResponse(
            status=result.get("status", "success"),
            files_generated=result.get("files_generated", 0),
            chunks_created=result.get("chunks_created", 0),
            embeddings_generated=result.get("embeddings_generated", 0),
            points_indexed=result.get("points_indexed", 0),
            message="Indexing completed successfully"
        )
    except Exception as e:
        print(f"Indexing Error: {e}")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")


@router.get("/health")
async def rag_health():
    """
    Health check for the RAG system.
    Returns Qdrant collection status and stats.
    """
    info = vector_store.get_collection_info()
    return {
        "rag_system": "operational" if info["status"] == "healthy" else "degraded",
        "qdrant": info,
    }


@router.post("/debug", response_model=List[dict])
async def debug_retrieve(request: RAGChatRequest):
    """
    Debug endpoint to see what chunks are retrieved for a given query.
    """
    try:
        if not request.messages:
            return []
        question = request.messages[-1].content
        return pipeline.retrieve_context(question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
