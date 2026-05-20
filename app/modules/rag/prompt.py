"""
Prompt Templates
Optimized system prompts and context injection templates
for the RAG pipeline.
"""
import json
import logging

logger = logging.getLogger(__name__)

# ── Core Identity & Style (sent once, compact) ───────────────────────────────

BASE_SYSTEM_PROMPT = """[IDENTITY]
You are Sakr AI, Mohamed Sakr's assistant. أنت مساعد محمد صقر.

[LANGUAGE]
CRITICAL: Speak ONLY Egyptian Arabic (Masri). NEVER use Fusha/MSA.
CRITICAL: Keep technical terms in English (e.g. Backend, API, React).
CRITICAL: Use 1-2 emojis per response naturally.

Use Masri: دلوقتي, عايز, تمام, ماشي, بص, يعني, خلاص, إزيك, أهلاً بيك
Avoid MSA: كيف يمكنني, هل تود, أريد, سوف, لكي

[RAG RULES]
- Use ONLY provided context. Never invent pricing, timelines, or skills.
- If "[NO RELEVANT CONTEXT FOUND]", admit you don't know.
- Empty context → never fabricate, say you don't have that info.

[BEHAVIOR]
- One question at a time. No CV dumps — give a teaser then ask.
- If HR: focus on problem-solving, team fit.
- If Client: understand business needs first.

CONTEXT:
{context}

SUMMARY:
{summary}
"""

# ── Booking Protocol (appended only when intent == 'hiring') ─────────────────

BOOKING_PROTOCOL = """
[BOOKING AGENT PROTOCOL]
1. You ARE the official booking agent. Do NOT refer users to forms/buttons.
2. Collect: Name, Contact (Email or Phone), Meeting Time.
3. AVAILABILITY: 7:00 PM - 11:00 PM Egypt Time only.
4. If info missing, ask professionally in user's language.
5. Arabic: 'ممكن تقولي اسمك بالكامل؟', 'إيه وسيلة التواصل المناسبة؟', 'إمتى الموعد المناسب؟ (محمد متاح من 7 لـ 11 مساءً)'.
6. Once complete: 'تم تسجيل الطلب، محمد هيتواصل معاك قريبًا'.
7. LEAD STATUS: {lead_json}
8. MISSING: {missing}
"""

CONTEXT_TEMPLATE = "\nSource: {source}\n{text}\n"


def build_rag_prompt(
    retrieved_chunks: list,
    history: list = None,
    summary: str = "",
    lead_data: dict = None,
    missing_fields: list = None,
) -> list:
    """
    Build the RAG prompt with context, history, summary, and optional booking protocol.

    Returns a list of message dicts for the AI provider.
    """
    top_chunks = retrieved_chunks[:3] if retrieved_chunks else []

    if top_chunks:
        context_str = "\n".join([
            CONTEXT_TEMPLATE.format(
                source=chunk.get("source", "Unknown"),
                text=chunk.get("text", "")
            )
            for chunk in top_chunks
        ])
    else:
        context_str = (
            "[NO RELEVANT CONTEXT FOUND] "
            "You have no information about this topic. "
            "Politely say you don't know and suggest the portfolio."
        )

    system_content = BASE_SYSTEM_PROMPT.format(
        context=context_str,
        summary=summary or "No previous summary."
    )

    # Append booking protocol if we have lead context
    if lead_data or missing_fields:
        try:
            booking_block = BOOKING_PROTOCOL.format(
                lead_json=json.dumps(lead_data or {}, ensure_ascii=False),
                missing=", ".join(missing_fields) if missing_fields else "None — confirm booking",
            )
            system_content += booking_block
        except Exception as e:
            logger.warning("Failed to append booking protocol: %s", e)

    messages = [{"role": "system", "content": system_content}]

    recent_history = history[-3:] if history else []
    for msg in recent_history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    return messages
