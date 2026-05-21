import logging
from typing import List, Dict, Optional, AsyncGenerator
from .base import BaseAgent, AgentResult, Message
from app.modules.rag import prompt as rag_prompt

logger = logging.getLogger(__name__)

CHAT_SYSTEM_PROMPT = """[IDENTITY]
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
- Be friendly, helpful, and conversational.

CONTEXT:
{context}

SUMMARY:
{summary}"""


class ChatAgent(BaseAgent):
    name = "chat"

    @property
    def system_prompt(self) -> str:
        return CHAT_SYSTEM_PROMPT

    async def _build_messages(
        self,
        question: str,
        history: List[Message],
        context: Optional[List[Dict]] = None,
        lead_data: Optional[Dict] = None,
        summary: str = "",
    ) -> List[Message]:
        top_chunks = (context or [])[:3]
        if top_chunks:
            context_str = "\n".join([
                rag_prompt.CONTEXT_TEMPLATE.format(
                    source=c.get("source", "Unknown"),
                    text=c.get("text", "")
                ) for c in top_chunks
            ])
        else:
            context_str = "[NO RELEVANT CONTEXT FOUND] You have no information about this topic. Politely say you don't know and suggest the portfolio."

        system_content = CHAT_SYSTEM_PROMPT.format(
            context=context_str,
            summary=summary or "No previous summary."
        )

        messages = [{"role": "system", "content": system_content}]
        for msg in history[-3:]:
            messages.append(msg)
        messages.append({"role": "user", "content": question})
        return messages

    async def process(
        self,
        question: str,
        history: List[Message],
        context: Optional[List[Dict]] = None,
        lead_data: Optional[Dict] = None,
        summary: str = "",
    ) -> AgentResult:
        from app.services.ai.manager import ai_manager

        messages = await self._build_messages(question, history, context, summary)
        content = await ai_manager.generate(messages=messages, temperature=0.3, max_tokens=512)
        return AgentResult(content=content, agent_name=self.name)

    async def process_stream(
        self,
        question: str,
        history: List[Message],
        context: Optional[List[Dict]] = None,
        lead_data: Optional[Dict] = None,
        summary: str = "",
    ) -> AsyncGenerator[str, None]:
        from app.services.ai.manager import ai_manager

        messages = await self._build_messages(question, history, context, summary)
        async for chunk in ai_manager.generate_stream(messages=messages, temperature=0.3, max_tokens=512):
            yield chunk
