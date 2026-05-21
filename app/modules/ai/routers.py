"""
AI Chat Router
Routes chat requests through the RAG pipeline for context-aware answers.
Falls back to basic chat if RAG is unavailable.
"""
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import List
import json
from . import schemas

logger = logging.getLogger(__name__)

router = APIRouter()

# Fallback system prompt (used only if RAG pipeline fails)
FALLBACK_SYSTEM_PROMPT = """
[LANGUAGE & STYLE]
CRITICAL: You MUST speak ONLY in Egyptian Arabic (Masri / Cairo style). NEVER use Modern Standard Arabic (Fusha/MSA).
CRITICAL: You MUST use at least one or two emojis in every response.
CRITICAL: Keep technical terms in English (e.g. Backend, API, React, Python). NEVER write them in Arabic letters.

You are Sakr AI, the assistant for Mohamed Sakr, a Backend-focused developer.
Mohamed specializes in Python, FastAPI, PostgreSQL, REST APIs, and AI/ML integration.
He also has experience with React, Next.js, TypeScript, Docker, and AWS.
If asked about specific skills or projects, provide what you know and suggest checking the portfolio for details.
"""


@router.post("/chat", response_model=schemas.ChatResponse)
async def chat(request: schemas.ChatRequest):
    """
    Main chat endpoint. Tries RAG pipeline first, falls back to basic chat.
    """
    try:
        # Try RAG pipeline first
        result = await _rag_chat(request)
        return schemas.ChatResponse(content=result)
    except Exception as rag_error:
        logger.warning("RAG pipeline unavailable (%s), falling back to basic chat...", rag_error)
        try:
            result = await _basic_chat(request)
            return schemas.ChatResponse(content=result)
        except Exception as basic_error:
            raise HTTPException(
                status_code=503,
                detail=f"AI service unavailable. RAG error: {rag_error}. LLM error: {basic_error}"
            )

@router.post("/chat/stream")
async def stream_chat(request: schemas.ChatRequest):
    """
    Streaming chat endpoint — uses full RAG pipeline (retrieval + lead capture),
    then streams the final AI response.
    """
    async def event_generator():
        try:
            async for chunk in _rag_chat_stream(request):
                yield f"data: {json.dumps({'content': chunk})}\n\n"
        except Exception as e:
            logger.warning("RAG stream failed (%s), falling back to basic stream...", e)
            async for chunk in _basic_chat_stream(request):
                yield f"data: {json.dumps({'content': chunk})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/explain-repo/stream")
async def explain_repo_stream(request: schemas.ExplainRepoRequest):
    """
    Explain a GitHub repo using RAG on its indexed README chunks.
    The prompt is fixed on the backend — frontend just sends full_name.
    """
    async def event_generator():
        try:
            async for chunk in _explain_repo_stream(request):
                yield f"data: {json.dumps({'content': chunk})}\n\n"
        except Exception as e:
            logger.error("Explain-repo stream failed: %s", e, exc_info=True)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


async def _rag_chat_stream(request: schemas.ChatRequest):
    """Streaming chat using the RAG pipeline — no duplicated graph logic."""
    from app.modules.rag import pipeline, prompt as rag_prompt
    from app.services.ai.manager import ai_manager

    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        raise ValueError("No user message found")

    question = user_messages[-1].content
    history = [{"role": m.role, "content": m.content} for m in request.messages[:-1]]

    state = await pipeline.prepare_rag_state(question, history)

    lead = state.get("lead_data", {})
    missing = []
    if not lead.get("name"):
        missing.append("name")
    if not lead.get("email") and not lead.get("phone"):
        missing.append("contact info")
    if not lead.get("meeting_time"):
        missing.append("meeting time (7-11 PM)")

    messages = rag_prompt.build_rag_prompt(
        state["context"],
        state["history"],
        state.get("summary", ""),
        lead_data=lead,
        missing_fields=missing,
    )

    messages.append({"role": "user", "content": question})

    async for chunk in ai_manager.generate_stream(
        messages=messages,
        temperature=0.3,
        max_tokens=512
    ):
        yield chunk


async def _basic_chat_stream(request: schemas.ChatRequest):
    """Fallback streaming chat without RAG."""
    from app.services.ai.manager import ai_manager
    
    messages = [{"role": "system", "content": FALLBACK_SYSTEM_PROMPT}]
    for msg in request.messages:
        messages.append({"role": msg.role, "content": msg.content})

    async for chunk in ai_manager.generate_stream(
        messages=messages,
        temperature=0.7,
        max_tokens=512
    ):
        yield chunk


EXPLAIN_REPO_SYSTEM_PROMPT = """
[LANGUAGE & STYLE]
CRITICAL: You MUST speak ONLY in Egyptian Arabic (Masri / Cairo style). NEVER use Modern Standard Arabic (Fusha/MSA).
CRITICAL: Use 1-2 emojis per response naturally.
CRITICAL: Keep technical terms in English (e.g. Backend, API, React, Python). NEVER write them in Arabic letters.
CRITICAL: NEVER mention you are answering based on "context" or "retrieved information". Just answer naturally.

You are Sakr AI, an expert GitHub repository analyst.
Your job is to explain repositories clearly and helpfully.

When explaining a repo, cover:
1. What the project is about (ببساطة المشروع ده إيه؟)
2. The problem it solves (المشكلة اللي بيحلها)
3. Architecture overview (إزاي بيشتغل من جوه)
4. Key features (أهم المميزات)
5. Use cases (أشهر حالات الاستخدام)
6. Your honest developer opinion (رأيك فيه كـ developer)

If you don't have enough context about the repo, admit it and suggest what the user can look for.

CONTEXT FROM REPOSITORY README:
{context}
"""


async def _explain_repo_stream(request: schemas.ExplainRepoRequest):
    """Stream an AI explanation of a GitHub repo using RAG on its README chunks."""
    from app.services.ai.manager import ai_manager
    from app.modules.github_trends.service import GitHubTrendsService
    from app.modules.rag.embeddings import embed_query
    from app.modules.rag.vector_store import search

    # 1. Fetch repo from MongoDB
    svc = GitHubTrendsService()
    repo = await svc.get_repo_by_full_name(request.full_name)
    if not repo:
        yield f"❌ مش لاقي الريبو {request.full_name}"
        return

    # 2. Search Qdrant for README chunks from this repo
    repo_query = request.question or f"شرح ريبو {request.full_name}"
    query_embedding = embed_query(repo_query)
    chunks = search(
        query_embedding,
        top_k=10,
        source_filter="repo_readme",
        extra_filters={"full_name": request.full_name},
    )

    # 3. Build context from retrieved chunks
    if chunks:
        context_parts = []
        for c in chunks:
            context_parts.append(f"[Chunk {c.get('chunk_index', 0)}] {c['text']}")
        context_str = "\n\n".join(context_parts)
    else:
        # Fallback: use full README from MongoDB (or description)
        readme = repo.get("readme_content", "")
        if readme and len(readme) > 8000:
            context_str = readme[:8000] + "\n\n...(اختصار)"
        elif readme:
            context_str = readme
        else:
            context_str = repo.get("description", "No README available.")

    # 4. Build fixed system prompt with context
    system_content = EXPLAIN_REPO_SYSTEM_PROMPT.format(context=context_str)

    user_message = request.question or f"شرحتلي الـ GitHub repository {request.full_name} بالعامية المصرية"

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_message},
    ]

    async for chunk in ai_manager.generate_stream(
        messages=messages,
        temperature=0.5,
        max_tokens=1024
    ):
        yield chunk


async def _rag_chat(request: schemas.ChatRequest) -> str:
    """Chat using the RAG pipeline (retrieval-augmented generation)."""
    from app.modules.rag import pipeline

    # Extract the latest user question
    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        raise ValueError("No user message found")

    question = user_messages[-1].content

    # Build conversation history (everything except the last user message)
    history = [{"role": m.role, "content": m.content} for m in request.messages[:-1]]

    # Run RAG pipeline
    result = await pipeline.generate_answer(question, history)
    return result["content"]


async def _basic_chat(request: schemas.ChatRequest) -> str:
    """Basic chat without RAG (fallback mode) using the AI Provider System."""
    from app.services.ai.manager import ai_manager
    
    messages = [{"role": "system", "content": FALLBACK_SYSTEM_PROMPT}]
    for msg in request.messages:
        messages.append({"role": msg.role, "content": msg.content})

    return await ai_manager.generate(
        messages=messages,
        temperature=0.7,
        max_tokens=512
    )
