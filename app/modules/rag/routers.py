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
    Main RAG chat endpoint.
    Takes conversation messages, retrieves relevant context from Qdrant,
    and generates a grounded answer using the LLM.
    """
    try:
        if not request.messages:
            raise HTTPException(status_code=400, detail="No messages provided")

        # Extract the latest user question
        user_messages = [m for m in request.messages if m.role == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user message found")

        question = user_messages[-1].content

        # Build conversation history (everything except the last user message)
        history = [{"role": m.role, "content": m.content} for m in request.messages[:-1]]

        # Run RAG pipeline
        result = await pipeline.generate_answer(question, history)

        return RAGChatResponse(
            content=result["content"],
            sources=result.get("sources", []),
            chunks_used=result.get("chunks_used", 0),
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"RAG Chat Error: {e}")
        raise HTTPException(status_code=500, detail=f"RAG pipeline error: {str(e)}")


@router.post("/index", response_model=IndexResponse)
async def index_knowledge():
    """
    Trigger full knowledge re-indexing.
    Extracts data from DB → generates Markdown → chunks → embeds → indexes in Qdrant.
    Call this after updating portfolio content in the admin panel.
    """
    try:
        result = pipeline.index_knowledge()
        return IndexResponse(**result)
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
