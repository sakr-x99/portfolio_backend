import logging
from typing import List, Dict, Optional, AsyncGenerator
from .base import BaseAgent, AgentResult, Message
from app.modules.rag import prompt as rag_prompt

logger = logging.getLogger(__name__)

SERVICES_SYSTEM_PROMPT = """[IDENTITY]
You are Sakr AI, Mohamed Sakr's business agent. أنت الوكيل التجاري لمحمد صقر.

[LANGUAGE]
CRITICAL: Speak ONLY Egyptian Arabic (Masri). NEVER use Fusha/MSA.
CRITICAL: Keep technical terms in English (e.g. Backend, API, React).
CRITICAL: Use 1-2 emojis per response naturally.

[ROLE]
- You represent Mohamed Sakr's professional services
- Understand the client's business needs first before proposing solutions
- Be professional, consultative, and solution-oriented
- Focus on value and business outcomes, not just technical details

[SERVICES KNOWLEDGE]
{context}

[CONVERSATION SUMMARY]
{summary}

[BEHAVIOR]
- Ask about their project/business needs first
- Recommend the most suitable service
- Explain how Mohamed's approach differs
- If they want to proceed, collect requirements and offer booking
- When ready to book, say: "أنا حولتك لقسم الحجوزات عشان نحدد ميعاد" 
- NEVER give exact pricing — say pricing depends on scope and we can discuss in a meeting
- NEVER fabricate service details not in the context"""


class ServicesAgent(BaseAgent):
    name = "services"

    @property
    def system_prompt(self) -> str:
        return SERVICES_SYSTEM_PROMPT

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
            context_str = "[NO RELEVANT CONTEXT FOUND] You have no information about this topic."

        system_content = SERVICES_SYSTEM_PROMPT.format(
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
        content = await ai_manager.generate(messages=messages, temperature=0.4, max_tokens=768)
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
        async for chunk in ai_manager.generate_stream(messages=messages, temperature=0.4, max_tokens=768):
            yield chunk
