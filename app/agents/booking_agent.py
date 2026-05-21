import json
import logging
from datetime import datetime
from typing import List, Dict, Optional, AsyncGenerator
from .base import BaseAgent, AgentResult, Message
from app.modules.rag import prompt as rag_prompt

logger = logging.getLogger(__name__)

BOOKING_SYSTEM_PROMPT = """[IDENTITY]
You are Sakr AI's booking agent. أنت مسؤول الحجوزات لمحمد صقر.

[LANGUAGE]
CRITICAL: Speak ONLY Egyptian Arabic (Masri). NEVER use Fusha/MSA.
CRITICAL: Use 1-2 emojis per response naturally.

[ROLE]
- You collect lead information for Mohamed Sakr
- You schedule consultation meetings
- You DO NOT answer technical questions — redirect to general chat

[BOOKING PROTOCOL]
1. You ARE the official booking agent. Do NOT refer users to forms/buttons.
2. Collect: Name, Contact (Email or Phone), Meeting Time.
3. AVAILABILITY: 7:00 PM - 11:00 PM Egypt Time only.
4. If info missing, ask professionally in user's language.
5. Arabic questions: 'ممكن تقولي اسمك بالكامل؟', 'إيه وسيلة التواصل المناسبة؟', 'إمتى الموعد المناسب؟ (محمد متاح من 7 لـ 11 مساءً)'.
6. Once complete: 'تم تسجيل الطلب ✅، محمد هيتواصل معاك قريبًا لتأكيد الموعد.'
7. LEAD STATUS: {lead_json}
8. MISSING INFO: {missing}

[CONTEXT]
{context}

[SUMMARY]
{summary}"""


class BookingAgent(BaseAgent):
    name = "booking"

    @property
    def system_prompt(self) -> str:
        return BOOKING_SYSTEM_PROMPT

    async def _build_messages(
        self,
        question: str,
        history: List[Message],
        context: Optional[List[Dict]] = None,
        lead_data: Optional[Dict] = None,
        summary: str = "",
    ) -> List[Message]:
        lead = lead_data or {}
        missing = []
        if not lead.get("name"):
            missing.append("name")
        if not lead.get("email") and not lead.get("phone"):
            missing.append("contact info (email or phone)")
        if not lead.get("meeting_time"):
            missing.append("preferred meeting time (between 7 PM - 11 PM)")

        context_str = "[NO RELEVANT CONTEXT FOUND]"
        if context:
            top = context[:2]
            context_str = "\n".join([
                rag_prompt.CONTEXT_TEMPLATE.format(
                    source=c.get("source", "Unknown"),
                    text=c.get("text", "")
                ) for c in top
            ])

        system_content = BOOKING_SYSTEM_PROMPT.format(
            lead_json=json.dumps(lead, ensure_ascii=False),
            missing=", ".join(missing) if missing else "None — confirm booking",
            context=context_str,
            summary=summary or "No previous summary."
        )

        messages = [{"role": "system", "content": system_content}]
        for msg in history[-3:]:
            messages.append(msg)
        messages.append({"role": "user", "content": question})
        return messages, lead, missing

    async def _save_lead(self, lead_data: Dict, question: str, summary: str = ""):
        """Save lead to database if we have contact info."""
        if not lead_data.get("email") and not lead_data.get("phone") and not lead_data.get("name"):
            return

        try:
            from app.core.database import SessionLocal
            from app.modules.biz_management.models import Lead

            db = SessionLocal()
            try:
                new_lead = Lead(
                    name=lead_data.get("name"),
                    email=lead_data.get("email"),
                    phone=lead_data.get("phone"),
                    inquiry_type=lead_data.get("inquiry_type") or "Booking",
                    message=lead_data.get("message") or question,
                    meeting_time=lead_data.get("meeting_time"),
                    summary=summary,
                    created_at=datetime.now().isoformat()
                )
                db.add(new_lead)
                db.commit()
                logger.info("Lead saved: %s", new_lead.name)
            except Exception as e:
                db.rollback()
                logger.error("Lead save error: %s", e, exc_info=True)
            finally:
                db.close()
        except Exception as e:
            logger.error("Lead save failed: %s", e)

    async def _extract_lead(self, question: str, history: List[Message]) -> Dict:
        """Extract lead info from conversation."""
        from app.services.ai.manager import ai_manager

        extract_prompt = (
            "Extract lead information from the conversation. "
            "Return ONLY a valid JSON object with keys: "
            "name, email, phone, inquiry_type, message, meeting_time. "
            "Use null for missing fields. No other text."
        )

        messages = [{"role": "system", "content": extract_prompt}]
        messages.extend(history[-6:])
        messages.append({"role": "user", "content": question})

        try:
            extracted = await ai_manager.generate(
                messages=messages, temperature=0.0, max_tokens=300
            )
            json_str = extracted.strip().replace("```json", "").replace("```", "").strip()
            return json.loads(json_str)
        except Exception as e:
            logger.warning("Lead extraction failed: %s", e)
            return {}

    async def process(
        self,
        question: str,
        history: List[Message],
        context: Optional[List[Dict]] = None,
        lead_data: Optional[Dict] = None,
        summary: str = "",
    ) -> AgentResult:
        from app.services.ai.manager import ai_manager

        extracted = await self._extract_lead(question, history)
        merged = {**(lead_data or {}), **extracted}

        messages, _, _ = await self._build_messages(question, history, context, merged, summary)
        content = await ai_manager.generate(messages=messages, temperature=0.3, max_tokens=512)

        await self._save_lead(merged, question, summary)

        return AgentResult(content=content, agent_name=self.name, lead_data=merged)

    async def process_stream(
        self,
        question: str,
        history: List[Message],
        context: Optional[List[Dict]] = None,
        lead_data: Optional[Dict] = None,
        summary: str = "",
    ) -> AsyncGenerator[str, None]:
        from app.services.ai.manager import ai_manager

        extracted = await self._extract_lead(question, history)
        merged = {**(lead_data or {}), **extracted}

        messages, _, _ = await self._build_messages(question, history, context, merged, summary)

        async for chunk in ai_manager.generate_stream(messages=messages, temperature=0.3, max_tokens=512):
            yield chunk

        await self._save_lead(merged, question, summary)
