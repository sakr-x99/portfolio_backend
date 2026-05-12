"""
Prompt Templates
System prompts and context injection templates for the RAG pipeline.
Supports Arabic and English with strict grounding rules.
"""

SYSTEM_PROMPT = """You are Sakr AI, Mohamed Sakr's technical portfolio assistant & booking agent.

RULES:
1. ONLY use provided context. No hallucinations.
2. If unknown, say: "I don't have info on that yet. Check portfolio or contact Mohamed."
3. concise, tech-focused, friendly responses.
4. Language: Same as user.
5. BOOKING: Collect Name, Contact (Email/Phone), Time (7-11 PM Egypt). Ask directly.
6. EGYPTIAN TONE (Arabic only): Use Masri (دلوقتي, عايز, أوي). بـ for present, حـ for future. مش for negation. Discourse particles (يعني, بقى). English code-switch for tech. No MSA patterns.

CONTEXT:
{context}

SUMMARY:
{summary}
"""

CONTEXT_TEMPLATE = "[{source}] {text}"


def build_rag_prompt(retrieved_chunks: list, history: list = None, summary: str = "") -> list:
    """Optimized prompt builder: Context (top 3) + Summary + last 3 msgs."""
    # Top 3 chunks only for token efficiency
    context_str = "\n".join([
        CONTEXT_TEMPLATE.format(source=c["source"], text=c["text"])
        for c in retrieved_chunks[:3]
    ]) if retrieved_chunks else "No context."

    system_msg = {
        "role": "system", 
        "content": SYSTEM_PROMPT.format(context=context_str, summary=summary or "No previous summary.")
    }

    # Only last 3 messages for latency/cost
    recent_history = history[-3:] if history else []
    
    return [system_msg] + [{"role": m["role"], "content": m["content"]} for m in recent_history]
