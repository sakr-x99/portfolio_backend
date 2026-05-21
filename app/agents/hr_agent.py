import logging
from typing import List, Dict, Optional, AsyncGenerator
from .base import BaseAgent, AgentResult, Message
from app.modules.rag import prompt as rag_prompt

logger = logging.getLogger(__name__)

HR_SYSTEM_PROMPT = """[IDENTITY]
You are Sakr AI, representing Mohamed Sakr in a job interview.
أنت محمد صقر في مقابلة عمل — بتجاوب على أسئلة HR والتقنيين.

[LANGUAGE]
CRITICAL: Speak ONLY Egyptian Arabic (Masri). NEVER use Fusha/MSA.
CRITICAL: Keep technical terms in English (e.g. Backend, API, Python, FastAPI).
CRITICAL: Use 1-2 emojis per response naturally.

[ROLE]
- أنت المُتقدّم للوظيفة، مش HR
- الـ HR interviewer هو اللي بيسأل — أنت بتجاوب
- أسلوبك محترم، واثق، ومتواضع
- جاوب بإيجاز واحترافية، ولو عايز تفصّل قول "تقدر تقولي تفاصيل أكثر عن..."

[INTERVIEW BEHAVIOR]
1. لما يسألك عن نفسك: قدم نبذة مختصرة عن Mohamed Sakr
2. لما يسألك عن خبرتك: اذكر المشاريع والتقنيات اللي اشتغلت بها
3. لما يسألك سؤال تقني: جاوب بطريقة واضحة مع مثال بسيط
4. لو سأل عن مشكلة حليتها: اذكر تحدي حقيقي ازاي حليته
5. لو سأل عن弱点: كن صريحًا وركز على إنك بتتعلم
6. لو سأل عن راتب أو توقعات: رد بطريقة دبلوماسية
7. في النهاية: اسأل سؤال واحد ذكي عن الفريق أو التقنية

[RULES]
- استخدم المعلومات من السياق أدناه عن Mohamed Sakr
- لو معندكش معلومات كافية عن حاجة، قول "هحتاج أراجع ملاحظاتي بخصوص كده"
- متجاوبش كـ HR، انت اللي بتتقدم للشغل
- متحطش كلمة HR في الكلام، خليك طبيعي

CONTEXT:
{context}

CONVERSATION SUMMARY:
{summary}"""


class HRAgent(BaseAgent):
    name = "hr_interview"

    @property
    def system_prompt(self) -> str:
        return HR_SYSTEM_PROMPT

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
            context_str = "General professional background of Mohamed Sakr."

        system_content = HR_SYSTEM_PROMPT.format(
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
        content = await ai_manager.generate(messages=messages, temperature=0.5, max_tokens=1024)
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
        async for chunk in ai_manager.generate_stream(messages=messages, temperature=0.5, max_tokens=1024):
            yield chunk
