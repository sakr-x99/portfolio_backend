import json
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, TypedDict
from . import vector_store, prompt, embeddings

logger = logging.getLogger(__name__)

def _get_ai_manager():
    from app.services.ai.manager import ai_manager
    return ai_manager

# ═══════════════════════════════════════════════════════════════════════════════
# STATE DEFINITION
# ═══════════════════════════════════════════════════════════════════════════════

class RAGState(TypedDict):
    """The state of our RAG graph."""
    question: str
    original_question: str  # Preserved original question before rewrite
    history: List[Dict]
    context: List[Dict]
    answer: str
    sources: List[str]
    intent: str  # 'chat' or 'hiring'
    lead_data: Dict  # Extracted lead info
    summary: str  # Conversation summary

# ═══════════════════════════════════════════════════════════════════════════════
# NODES
# ═══════════════════════════════════════════════════════════════════════════════

async def classify_intent_node(state: RAGState) -> Dict:
    """Classify user intent: are they interested in hiring/freelance or just chatting?"""
    logger.info("--- CLASSIFYING INTENT ---")
    question = state["question"]
    history = state["history"]

    prompt_text = (
        "Classify the user's intent. "
        "If they are asking about hiring, freelance services, meeting with Mohamed, "
        "booking a call, or expressing professional interest (in English or Arabic), classify as 'hiring'. "
        "Arabic keywords to look for: 'خدمة', 'ميتنج', 'اجتماع', 'شغل', 'توظيف', 'عايز اقابلك', 'بروجكت'. "
        "Otherwise, classify as 'chat'. "
        "Return ONLY the single word 'hiring' or 'chat'."
    )

    messages = [{"role": "system", "content": prompt_text}]
    messages.extend(history[-4:])
    messages.append({"role": "user", "content": question})

    try:
        intent_raw = await _get_ai_manager().generate(
            messages=messages,
            temperature=0.0,
            max_tokens=10
        )
        intent = intent_raw.strip().lower()
        is_hiring = intent == "hiring"
        logger.info("Intent classified: %s (raw: '%s')", "hiring" if is_hiring else "chat", intent)
        return {"intent": "hiring" if is_hiring else "chat"}
    except Exception as e:
        logger.error("Intent classification failed: %s", e)
        return {"intent": "chat"}  # Default to chat on failure

async def rewrite_node(state: RAGState) -> Dict:
    """Rewrite the user question to optimize for vector search."""
    logger.info("--- REWRITING QUERY ---")
    question = state["original_question"]
    history = state["history"]

    rewrite_prompt = (
        "You are a search query optimizer. Given a conversation history and a new user question, "
        "rewrite the question to be a standalone search query that captures the full context. "
        "Return ONLY the rewritten query, nothing else."
    )

    messages = [{"role": "system", "content": rewrite_prompt}]
    for msg in history[-3:]:
        messages.append(msg)
    messages.append({"role": "user", "content": question})

    try:
        rewritten_query = await _get_ai_manager().generate(
            messages=messages,
            temperature=0.0,
            max_tokens=50
        )
        # Fall back to original if rewrite is empty or too long
        cleaned = rewritten_query.strip()
        if not cleaned or len(cleaned) > len(question) * 3:
            logger.warning("Rewrite rejected (empty/too long), using original")
            return {}
        logger.info("Query rewritten: '%s' → '%s'", question[:50], cleaned[:50])
        return {"question": cleaned}
    except Exception as e:
        logger.error("Query rewrite failed: %s", e)
        return {}  # Keep original question

async def retrieve_node(state: RAGState) -> Dict:
    """Retrieve relevant context from vector store (async wrapper)."""
    logger.info("--- RETRIEVING CONTEXT ---")
    query = state["question"]
    from .config import TOP_K

    def _sync_search(q, k):
        qe = embeddings.embed_query(q)
        return vector_store.search(qe, top_k=k)

    try:
        loop = asyncio.get_event_loop()
        retrieved_chunks = await loop.run_in_executor(None, _sync_search, query, TOP_K)
        sources = list(set(c["source"] for c in retrieved_chunks))
        logger.info("Retrieved %d chunks from %d sources", len(retrieved_chunks), len(sources))
        return {
            "context": retrieved_chunks,
            "sources": sources
        }
    except Exception as e:
        logger.error("Retrieval failed: %s", e)
        return {"context": [], "sources": []}

async def lead_capture_node(state: RAGState) -> Dict:
    """If intent is 'hiring', extract lead information and save to DB."""
    if state["intent"] != "hiring":
        return {}

    logger.info("--- EXTRACTING LEAD DATA ---")

    extract_prompt = (
        "Extract lead information from the conversation. "
        "Return ONLY a valid JSON object with these exact lowercase keys: "
        "name, email, phone, inquiry_type, message, meeting_time. "
        "Use null for missing fields. Do not include any other text."
    )

    messages = [{"role": "system", "content": extract_prompt}]
    messages.extend(state["history"][-6:])
    messages.append({"role": "user", "content": state["original_question"]})

    try:
        extracted_str = await _get_ai_manager().generate(
            messages=messages, temperature=0.0, max_tokens=300
        )

        json_str = extracted_str.strip().replace("```json", "").replace("```", "").strip()
        raw_data = json.loads(json_str)

        lead_data = {k.lower(): v for k, v in raw_data.items()}
        logger.info("Extracted lead data: %s", lead_data)

        if lead_data.get("email") or lead_data.get("phone") or lead_data.get("name"):
            logger.info("Saving lead in DB...")

            def _save_lead_sync(ld, question, summary):
                from app.core.database import SessionLocal
                from app.modules.biz_management.models import Lead
                db = SessionLocal()
                try:
                    new_lead = Lead(
                        name=ld.get("name"),
                        email=ld.get("email"),
                        phone=ld.get("phone"),
                        inquiry_type=ld.get("inquiry_type") or "General",
                        message=ld.get("message") or question,
                        meeting_time=ld.get("meeting_time"),
                        summary=summary,
                        created_at=datetime.now().isoformat()
                    )
                    db.add(new_lead)
                    db.commit()
                    logger.info("Lead saved! Name=%s, Phone=%s", new_lead.name, new_lead.phone)
                except Exception as db_err:
                    db.rollback()
                    logger.error("DB save error: %s", db_err, exc_info=True)
                finally:
                    db.close()

            await asyncio.to_thread(
                _save_lead_sync, lead_data, state["original_question"], state.get("summary", "")
            )
        else:
            logger.info("No contact info found yet, skipping save.")

        return {"lead_data": lead_data}
    except Exception as e:
        logger.warning("Error parsing lead JSON: %s", e)
        return {}

async def generate_node(state: RAGState) -> Dict:
    """Generate answer using retrieved context and lead awareness."""
    logger.info("--- GENERATING ANSWER ---")

    lead = state.get("lead_data", {})
    missing = []
    if not lead.get("name"):
        missing.append("name")
    if not lead.get("email") and not lead.get("phone"):
        missing.append("contact info (email or phone)")
    if not lead.get("meeting_time"):
        missing.append("preferred meeting time (between 7 PM - 11 PM)")

    # Build base prompt
    messages = prompt.build_rag_prompt(
        retrieved_chunks=state["context"],
        history=state["history"],
        summary=state.get("summary", ""),
        lead_data=lead,
        missing_fields=missing,
    )

    try:
        answer = await _get_ai_manager().generate(
            messages=messages, temperature=0.3, max_tokens=1024
        )
        return {"answer": answer}
    except Exception as e:
        logger.error("Generation failed: %s", e)
        return {"answer": "عذرًا، حصل مشكلة في النظام. حاول تاني بعد شوية."}

async def summarize_node(state: RAGState) -> Dict:
    """Summarize the conversation so far (only if there's history)."""
    history = state["history"]
    if not history:
        return {}

    logger.info("--- SUMMARIZING ---")

    summary_prompt = "Summarize the following conversation in one short paragraph."
    messages = [{"role": "system", "content": summary_prompt}]
    messages.extend(history[-10:])

    try:
        summary = await _get_ai_manager().generate(
            messages=messages,
            temperature=0.3,
            max_tokens=150
        )
        logger.info("Summary generated (%d chars)", len(summary))
        return {"summary": summary}
    except Exception as e:
        logger.error("Summarization failed: %s", e)
        return {}

# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════

def _should_summarize(state: RAGState) -> str:
    """Conditional edge: only summarize if there is conversation history."""
    from langgraph.graph import END
    return "summarize" if state.get("history") else END


def create_rag_graph():
    from langgraph.graph import StateGraph, START, END
    workflow = StateGraph(RAGState)

    workflow.add_node("classify", classify_intent_node)
    workflow.add_node("rewrite", rewrite_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("lead_capture", lead_capture_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("summarize", summarize_node)

    workflow.add_edge(START, "classify")
    workflow.add_edge("classify", "rewrite")
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("retrieve", "lead_capture")
    workflow.add_edge("lead_capture", "generate")
    workflow.add_conditional_edges("generate", _should_summarize)
    workflow.add_edge("summarize", END)

    return workflow.compile()


_rag_app_instance = None
_rag_app_lock = None

def _get_rag_lock():
    global _rag_app_lock
    if _rag_app_lock is None:
        from threading import Lock
        _rag_app_lock = Lock()
    return _rag_app_lock


def get_rag_app():
    global _rag_app_instance
    if _rag_app_instance is None:
        with _get_rag_lock():
            if _rag_app_instance is None:
                _rag_app_instance = create_rag_graph()
    return _rag_app_instance
