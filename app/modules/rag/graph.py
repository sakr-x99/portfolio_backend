import json
from datetime import datetime
from typing import List, Dict, TypedDict
from langgraph.graph import StateGraph, START, END
from app.modules.biz_management.models import Lead
from app.core.database import SessionLocal
from app.services.ai.manager import ai_manager
from . import vector_store, prompt, embeddings

# ═══════════════════════════════════════════════════════════════════════════════
# STATE DEFINITION
# ═══════════════════════════════════════════════════════════════════════════════

class RAGState(TypedDict):
    """The state of our RAG graph."""
    question: str
    history: List[Dict]
    context: List[Dict]
    answer: str
    sources: List[str]
    intent: str # 'chat' or 'hiring'
    lead_data: Dict # Extracted lead info
    summary: str # Conversation summary

# ═══════════════════════════════════════════════════════════════════════════════
# NODES
# ═══════════════════════════════════════════════════════════════════════════════

async def classify_intent_node(state: RAGState) -> Dict:
    """Classify user intent: are they interested in hiring/freelance or just chatting?"""
    print(f"--- CLASSIFYING INTENT ---")
    question = state["question"]
    history = state["history"]
    
    prompt = (
        "Classify the user's intent. "
        "If they are asking about hiring, freelance services, meeting with Mohamed, "
        "booking a call, or expressing professional interest (in English or Arabic), classify as 'hiring'. "
        "Arabic keywords to look for: 'خدمة', 'ميتنج', 'اجتماع', 'شغل', 'توظيف', 'عايز اقابلك', 'بروجكت'. "
        "Otherwise, classify as 'chat'. "
        "Return ONLY the word 'hiring' or 'chat'."
    )
    
    messages = [{"role": "system", "content": prompt}]
    messages.extend(history[-4:]) # More context
    messages.append({"role": "user", "content": question})
    
    intent_raw = await ai_manager.generate(
        messages=messages, 
        temperature=0.0, 
        max_tokens=10
    )
    intent = intent_raw.strip().lower()
    
    return {"intent": "hiring" if any(k in intent for k in ["hiring", "hire", "service", "meeting"]) else "chat"}

async def rewrite_node(state: RAGState) -> Dict:
    """Rewrite the user question to optimize for vector search."""
    # (Existing logic)
    print(f"--- REWRITING QUERY ---")
    question = state["question"]
    history = state["history"]
    
    rewrite_prompt = (
        "You are a search query optimizer. Given a conversation history and a new user question, "
        "rewrite the question to be a standalone search query that captures the full context. "
        "Return ONLY the rewritten query, nothing else."
    )
    
    messages = [{"role": "system", "content": rewrite_prompt}]
    for msg in history[-3:]: messages.append(msg)
    messages.append({"role": "user", "content": question})
    
    rewritten_query = await ai_manager.generate(
        messages=messages, 
        temperature=0.0, 
        max_tokens=50
    )
    return {"question": rewritten_query}

def retrieve_node(state: RAGState) -> Dict:
    """Retrieve relevant context from vector store."""
    print(f"--- RETRIEVING CONTEXT ---")
    query = state["question"]
    query_embedding = embeddings.embed_query(query)
    retrieved_chunks = vector_store.search(query_embedding, top_k=3) # Minimized context
    return {
        "context": retrieved_chunks,
        "sources": list(set(c["source"] for c in retrieved_chunks))
    }

async def lead_capture_node(state: RAGState) -> Dict:
    """If intent is 'hiring', extract lead information and save to DB."""
    if state["intent"] != "hiring":
        return {}

    print(f"--- EXTRACTING LEAD DATA ---")
    
    # Extract data using LLM
    extract_prompt = (
        "Extract lead information from the conversation. "
        "Return ONLY a valid JSON object with these exact lowercase keys: "
        "name, email, phone, inquiry_type, message, meeting_time. "
        "Use null for missing fields. Do not include any other text."
    )
    
    messages = [{"role": "system", "content": extract_prompt}]
    messages.extend(state["history"][-6:])
    messages.append({"role": "user", "content": state["question"]})
    
    extracted_str = await ai_manager.generate(messages=messages, temperature=0.0, max_tokens=300)
    
    try:
        json_str = extracted_str.strip().replace("```json", "").replace("```", "").strip()
        raw_data = json.loads(json_str)
        
        # Normalize all keys to lowercase to handle LLM inconsistency
        lead_data = {k.lower(): v for k, v in raw_data.items()}
        print(f"  → Extracted lead data: {lead_data}")
        
        # Save if we have any useful info
        if lead_data.get("email") or lead_data.get("phone") or lead_data.get("name"):
            print(f"  → Saving lead in DB...")
            db = SessionLocal()
            try:
                new_lead = Lead(
                    name=lead_data.get("name"),
                    email=lead_data.get("email"),
                    phone=lead_data.get("phone"),
                    inquiry_type=lead_data.get("inquiry_type") or "General",
                    message=lead_data.get("message") or state["question"],
                    meeting_time=lead_data.get("meeting_time"),
                    summary=state.get("summary", ""),
                    created_at=datetime.now().isoformat()
                )
                db.add(new_lead)
                db.commit()
                print(f"  ✓ Lead saved! Name={new_lead.name}, Phone={new_lead.phone}")
            except Exception as db_err:
                db.rollback()
                print(f"  ⚠ DB save error: {db_err}")
            finally:
                db.close()
        else:
            print(f"  → No contact info found yet, skipping save.")
        
        return {"lead_data": lead_data}
    except Exception as e:
        print(f"  ⚠ Error parsing lead JSON: {e}\n  Raw response: {extracted_str[:200]}")
        return {}

async def generate_node(state: RAGState) -> Dict:
    """Generate answer using retrieved context and lead awareness."""
    print(f"--- GENERATING ANSWER ---")
    
    # Identify missing info
    lead = state.get("lead_data", {})
    missing = []
    if not lead.get("name"): missing.append("name")
    if not lead.get("email") and not lead.get("phone"): missing.append("contact info (email or phone)")
    if not lead.get("meeting_time"): missing.append("preferred meeting time (between 7 PM - 11 PM)")

    # Customize system instructions based on intent
    system_add = (
        "\n\n[BOOKING AGENT PROTOCOL]\n"
        "1. You ARE the official booking agent for Mohamed Sakr. Do NOT refer users to other forms or buttons.\n"
        "2. If the user wants to hire or meet Mohamed, you MUST collect: Name, Contact Info (Email or Phone), and a Meeting Time.\n"
        "3. AVAILABILITY: Mohamed is ONLY available between 7:00 PM and 11:00 PM Egypt Time. If they suggest another time, politely inform them of his availability.\n"
        "4. If info is missing, ask for it directly and professionally in the user's language.\n"
        "5. ARABIC GUIDANCE: If speaking Arabic, ask for missing info like this:\n"
        "   - Name: 'ممكن تقولي اسمك بالكامل؟'\n"
        "   - Contact: 'إيه وسيلة التواصل المناسبة (رقم تليفون أو إيميل)؟'\n"
        "   - Time: 'إمتى الموعد المناسب ليك؟ (محمد متاح من 7 لـ 11 مساءً بتوقيت مصر)'.\n"
        "6. Once all info is provided, confirm that 'the request has been recorded in the system and Mohamed will contact you soon'.\n"
        f"7. CURRENT STATUS OF LEAD DATA: {json.dumps(lead, ensure_ascii=False)}.\n"
        f"8. MISSING FIELDS: {', '.join(missing) if missing else 'None'}.\n"
    )

    # Build prompt with summary and context awareness
    messages = prompt.build_rag_prompt(state["context"], state["history"], state.get("summary", ""))
    messages[0]["content"] += system_add
    
    # Call AI Manager
    answer = await ai_manager.generate(messages=messages, temperature=0.3, max_tokens=1024)
    return {"answer": answer}

async def summarize_node(state: RAGState) -> Dict:
    """Summarize the conversation so far."""
    print(f"--- SUMMARIZING ---")
    history = state["history"]
    if not history: return {}
    
    summary_prompt = "Summarize the following conversation in one short paragraph."
    messages = [{"role": "system", "content": summary_prompt}]
    messages.extend(history[-10:]) # Summarize last 10 turns
    
    summary = await ai_manager.generate(
        messages=messages, 
        temperature=0.3, 
        max_tokens=150
    )
    return {"summary": summary}

# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════

def create_rag_graph():
    workflow = StateGraph(RAGState)
    
    # Add nodes
    workflow.add_node("classify", classify_intent_node)
    workflow.add_node("rewrite", rewrite_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("lead_capture", lead_capture_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("summarize", summarize_node)
    
    # Define edges
    workflow.add_edge(START, "classify")
    workflow.add_edge("classify", "rewrite")
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("retrieve", "lead_capture")
    workflow.add_edge("lead_capture", "generate")
    workflow.add_edge("generate", "summarize")
    workflow.add_edge("summarize", END)
    
    # Compile
    return workflow.compile()

# Global graph instance
rag_app = create_rag_graph()
