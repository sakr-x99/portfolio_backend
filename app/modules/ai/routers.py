"""
AI Chat Router
Routes chat requests through the multi-agent system with memory and caching.
"""
import logging
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
import json
from . import schemas
from app.agents import AgentResult

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Lazy imports for agent system ──────────────────────────────────────────

def _get_registry():
    from app.agents import agent_registry
    return agent_registry

def _get_memory():
    from app.memory import conversation_memory
    return conversation_memory

def _get_cache():
    from app.core.cache import get_ai_cache_key, get_cached_ai_response, set_cached_ai_response
    return get_ai_cache_key, get_cached_ai_response, set_cached_ai_response


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


def _ensure_session(request) -> str:
    """Ensure we have a session_id (generate one if not provided)."""
    from app.memory import conversation_memory
    session_id = request.session_id
    if not session_id:
        session_id = conversation_memory.generate_session_id()
    return session_id


# ═══════════════════════════════════════════════════════════════════════════════
# CONVERSATIONS ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/conversations")
async def list_conversations(limit: int = Query(default=10, le=50)):
    """List all conversation sessions with titles and timestamps."""
    from app.memory import conversation_memory
    sessions = await conversation_memory.get_all_sessions()
    sessions = sessions[-limit:]

    result = []
    for sid in sessions:
        try:
            msgs = await conversation_memory.get_messages(sid, limit=2)
            meta = await conversation_memory.get_metadata(sid)
            title = meta.get("title", "") or (msgs[0]["content"][:60] if msgs else "New Chat")
            result.append({
                "session_id": sid,
                "title": title,
                "message_count": len(await conversation_memory.get_messages(sid)),
                "last_active": meta.get("last_active", ""),
                "created_at": meta.get("created_at", ""),
            })
        except Exception:
            continue

    result.sort(key=lambda x: x.get("last_active", ""), reverse=True)
    return result


@router.delete("/conversations/{session_id}")
async def delete_conversation(session_id: str):
    """Delete a conversation session."""
    from app.memory import conversation_memory
    await conversation_memory.clear(session_id)
    return {"status": "deleted", "session_id": session_id}


# ═══════════════════════════════════════════════════════════════════════════════
# CHAT ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/chat", response_model=schemas.ChatResponse)
async def chat(request: schemas.ChatRequest):
    """
    Main chat endpoint. Uses multi-agent system with memory + cache.
    Falls back to basic chat if agent system fails.
    """
    try:
        result = await _agent_chat(request)
        return schemas.ChatResponse(
            content=result.content,
            session_id=request.session_id,
            agent=result.agent_name,
            sources=result.sources,
            cached=result.cached,
        )
    except Exception as agent_error:
        logger.warning("Agent chat failed (%s), falling back to basic chat...", agent_error)
        try:
            result = await _basic_chat(request)
            return schemas.ChatResponse(
                content=result,
                session_id=request.session_id,
            )
        except Exception as basic_error:
            raise HTTPException(
                status_code=503,
                detail=f"AI service unavailable. Agent error: {agent_error}. LLM error: {basic_error}"
            )


@router.post("/chat/stream")
async def stream_chat(request: schemas.ChatRequest):
    """
    Streaming chat endpoint — uses multi-agent system with memory + cache.
    """
    async def event_generator():
        try:
            async for chunk in _agent_chat_stream(request):
                yield f"data: {json.dumps({'content': chunk})}\n\n"
        except Exception as e:
            logger.warning("Agent stream failed (%s), falling back to basic stream...", e)
            async for chunk in _basic_chat_stream(request):
                yield f"data: {json.dumps({'content': chunk})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/explain-repo/stream")
async def explain_repo_stream(request: schemas.ExplainRepoRequest):
    """
    Explain a GitHub repo using the ExplainRepo agent.
    """
    async def event_generator():
        try:
            async for chunk in _explain_repo_stream(request):
                yield f"data: {json.dumps({'content': chunk})}\n\n"
        except Exception as e:
            logger.error("Explain-repo stream failed: %s", e, exc_info=True)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ═══════════════════════════════════════════════════════════════════════════════
# INTERNAL: Agent-based chat (with memory + cache)
# ═══════════════════════════════════════════════════════════════════════════════

async def _agent_chat(request: schemas.ChatRequest):
    """Chat using the multi-agent system with memory and cache."""
    from app.modules.rag import pipeline

    session_id = _ensure_session(request)
    memory = _get_memory()

    # Store session_id on request for response
    request.session_id = session_id

    # Extract the latest user question
    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        raise ValueError("No user message found")

    question = user_messages[-1].content
    history = [{"role": m.role, "content": m.content} for m in request.messages[:-1]]

    # Load memory context
    memory_messages = await memory.get_messages(session_id, limit=6)
    if memory_messages:
        history = memory_messages + history

    mem_summary = await memory.get_summary(session_id)
    mem_lead = await memory.get_lead_data(session_id)

    # Check cache
    cache_ttl = 300  # 5 min default
    get_ai_cache_key, get_cached_ai_response, set_cached_ai_response = _get_cache()

    # Try RAG context
    try:
        state = await pipeline.prepare_rag_state(question, history)
        context = state.get("context", [])
        lead_data = {**mem_lead, **state.get("lead_data", {})}
        summary = state.get("summary", "") or mem_summary
    except Exception as e:
        logger.warning("RAG prep failed: %s", e)
        context = []
        lead_data = mem_lead
        summary = mem_summary

    # Try cache hit
    top_context = context[0].get("text", "")[:200] if context else ""
    cache_key = get_ai_cache_key("router", question, top_context)
    cached_response = get_cached_ai_response(cache_key, cache_ttl)
    if cached_response:
        await memory.add_message(session_id, "user", question)
        await memory.add_message(session_id, "assistant", cached_response)
        return AgentResult(content=cached_response, agent_name="cache", cached=True)

    # Route through agent system
    registry = _get_registry()
    result = await registry.process(
        question=question,
        history=history,
        context=context,
        lead_data=lead_data,
        summary=summary,
    )

    # Cache the response
    set_cached_ai_response(cache_key, result.content, cache_ttl)

    # Save to memory
    await memory.add_message(session_id, "user", question)
    await memory.add_message(session_id, "assistant", result.content)
    if result.lead_data:
        await memory.set_lead_data(session_id, result.lead_data)

    # Set conversation title from first user message
    meta = await memory.get_metadata(session_id)
    if not meta.get("title"):
        await memory.set_metadata(session_id, title=question[:60])

    # Auto-summarize if conversation is long
    if len(history) >= 8:
        await memory.auto_summarize(session_id)

    return result


async def _agent_chat_stream(request: schemas.ChatRequest):
    """Streaming chat using the multi-agent system with memory."""
    from app.modules.rag import pipeline

    session_id = _ensure_session(request)
    memory = _get_memory()

    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        raise ValueError("No user message found")

    question = user_messages[-1].content
    history = [{"role": m.role, "content": m.content} for m in request.messages[:-1]]

    # Load memory context
    memory_messages = await memory.get_messages(session_id, limit=6)
    if memory_messages:
        history = memory_messages + history

    mem_summary = await memory.get_summary(session_id)
    mem_lead = await memory.get_lead_data(session_id)

    # Try RAG context
    try:
        state = await pipeline.prepare_rag_state(question, history)
        context = state.get("context", [])
        lead_data = {**mem_lead, **state.get("lead_data", {})}
        summary = state.get("summary", "") or mem_summary
    except Exception as e:
        logger.warning("RAG prep failed in stream: %s", e)
        context = []
        lead_data = mem_lead
        summary = mem_summary

    # Stream through agent system
    registry = _get_registry()
    full_response = ""
    async for chunk in registry.process_stream(
        question=question,
        history=history,
        context=context,
        lead_data=lead_data,
        summary=summary,
    ):
        full_response += chunk
        yield chunk

    # Save to memory after streaming
    await memory.add_message(session_id, "user", question)
    await memory.add_message(session_id, "assistant", full_response)

    # Set conversation title from first user message
    meta = await memory.get_metadata(session_id)
    if not meta.get("title"):
        await memory.set_metadata(session_id, title=question[:60])

    # Cache the full response
    cache_ttl = 300
    top_context = context[0].get("text", "")[:200] if context else ""
    get_ai_cache_key, _, set_cached_ai_response = _get_cache()
    cache_key = get_ai_cache_key("router", question, top_context)
    set_cached_ai_response(cache_key, full_response, cache_ttl)

    if len(history) >= 8:
        await memory.auto_summarize(session_id)


# ═══════════════════════════════════════════════════════════════════════════════
# INTERNAL: Explain Repo
# ═══════════════════════════════════════════════════════════════════════════════

async def _explain_repo_stream(request: schemas.ExplainRepoRequest):
    """Stream an AI explanation of a GitHub repo using ExplainRepo agent."""
    from app.agents import ExplainRepoAgent

    session_id = _ensure_session(request)
    memory = _get_memory()

    agent = ExplainRepoAgent()
    full_response = ""
    async for chunk in agent.process_stream(
        full_name=request.full_name,
        question=request.question,
    ):
        full_response += chunk
        yield chunk

    # Save to memory
    user_msg = request.question or f"شرحتلي repo {request.full_name}"
    await memory.add_message(session_id, "user", user_msg)
    await memory.add_message(session_id, "assistant", full_response)

    # Set conversation title from first interaction
    meta = await memory.get_metadata(session_id)
    if not meta.get("title"):
        await memory.set_metadata(session_id, title=user_msg[:60])


# ═══════════════════════════════════════════════════════════════════════════════
# INTERNAL: Fallback basic chat (no RAG, no agents)
# ═══════════════════════════════════════════════════════════════════════════════

async def _basic_chat(request: schemas.ChatRequest) -> str:
    """Basic chat without RAG (fallback mode)."""
    from app.services.ai.manager import ai_manager

    messages = [{"role": "system", "content": FALLBACK_SYSTEM_PROMPT}]
    for msg in request.messages:
        messages.append({"role": msg.role, "content": msg.content})

    return await ai_manager.generate(messages=messages, temperature=0.7, max_tokens=512)


async def _basic_chat_stream(request: schemas.ChatRequest):
    """Fallback streaming chat without RAG."""
    from app.services.ai.manager import ai_manager

    messages = [{"role": "system", "content": FALLBACK_SYSTEM_PROMPT}]
    for msg in request.messages:
        messages.append({"role": msg.role, "content": msg.content})

    async for chunk in ai_manager.generate_stream(messages=messages, temperature=0.7, max_tokens=512):
        yield chunk
