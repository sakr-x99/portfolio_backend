"""
AI Chat Router
Routes chat requests through the RAG pipeline for context-aware answers.
Falls back to basic chat if RAG is unavailable.
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import List
import json
from . import schemas

router = APIRouter()

# Fallback system prompt (used only if RAG pipeline fails)
FALLBACK_SYSTEM_PROMPT = """
[LANGUAGE & STYLE]
CRITICAL: You MUST speak ONLY in Egyptian Arabic (Masri / Cairo style). NEVER use Modern Standard Arabic (Fusha/MSA).
CRITICAL: You MUST use at least one or two emojis in every response.
CRITICAL: Keep technical terms in English (e.g. Backend, API, React, Python). NEVER write them in Arabic letters.

You are Sakr AI, the assistant for Mohamed Sakr, a Backend-focused developer.
Because your knowledge base is currently syncing, DO NOT guess or invent any skills, projects, or technical stacks.
If asked about his skills, politely say that the data is syncing and suggest checking the portfolio sections directly.
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
        print(f"  ⚠ RAG pipeline unavailable ({rag_error}), falling back to basic chat...")
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
            print(f"  ⚠ RAG stream failed ({e}), falling back to basic stream...")
            async for chunk in _basic_chat_stream(request):
                yield f"data: {json.dumps({'content': chunk})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


async def _rag_chat_stream(request: schemas.ChatRequest):
    """Streaming chat using the RAG pipeline — no duplicated graph logic."""
    from app.modules.rag import pipeline, prompt as rag_prompt
    from app.services.ai.manager import ai_manager
    import json as json_lib

    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        raise ValueError("No user message found")

    question = user_messages[-1].content
    history = [{"role": m.role, "content": m.content} for m in request.messages[:-1]]

    # Use the shared RAG state preparation (calls graph nodes internally)
    state = await pipeline.prepare_rag_state(question, history)

    # Build prompt with retrieved context
    messages = rag_prompt.build_rag_prompt(state["context"], state["history"], state.get("summary", ""))

    # Add booking protocol if hiring intent
    if state["intent"] == "hiring":
        lead = state.get("lead_data", {})
        missing = []
        if not lead.get("name"): missing.append("name")
        if not lead.get("email") and not lead.get("phone"): missing.append("contact info")
        if not lead.get("meeting_time"): missing.append("meeting time (7-11 PM)")

        booking_add = (
            "\n[BOOKING MODE] Collect: Name, Contact, Time (7-11 PM Egypt)."
            f" Lead: {json_lib.dumps(lead, ensure_ascii=False)}."
            f" Missing: {', '.join(missing) if missing else 'None — confirm booking'}.\n"
        )
        messages[0]["content"] += booking_add

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
