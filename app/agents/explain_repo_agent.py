import logging
from typing import List, Dict, Optional, AsyncGenerator
from .base import BaseAgent, AgentResult, Message

logger = logging.getLogger(__name__)

EXPLAIN_REPO_SYSTEM_PROMPT = """
[LANGUAGE & STYLE]
CRITICAL: You MUST speak ONLY in Egyptian Arabic (Masri / Cairo style). NEVER use Modern Standard Arabic (Fusha/MSA).
CRITICAL: Use 1-2 emojis per response naturally.
CRITICAL: Keep technical terms in English (e.g. Backend, API, React, Python). NEVER write them in Arabic letters.
CRITICAL: NEVER mention you are answering based on "context" or "retrieved information". Just answer naturally.

You are Sakr AI, an expert GitHub repository analyst.
Your job is to explain repositories clearly and helpfully.

When explaining a repo, cover:
1. What the project is about
2. The problem it solves
3. Architecture overview
4. Key features
5. Use cases
6. Your honest developer opinion

If you don't have enough context about the repo, admit it and suggest what the user can look for.

CONTEXT FROM REPOSITORY README:
{context}
"""


class ExplainRepoAgent(BaseAgent):
    name = "explain_repo"

    @property
    def system_prompt(self) -> str:
        return EXPLAIN_REPO_SYSTEM_PROMPT

    def _strip_images(self, text: str) -> str:
        import re
        text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', '', text)
        text = re.sub(r'https?://\S+\.(png|jpg|jpeg|gif|svg|webp|ico)(\?\S*)?', '', text, flags=re.IGNORECASE)
        text = re.sub(r'data:image/\w+;base64,[A-Za-z0-9+/=]+', '', text)
        text = re.sub(r'<img[^>]+>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'<svg[^>]*>.*?</svg>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<figure[^>]*>.*?</figure>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'\b\S+\.(png|jpg|jpeg|gif|svg|webp|ico|bmp|tiff?)\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b\S+%20(png|jpg|jpeg|gif|svg|webp|ico|bmp|tiff?)\b', '', text, flags=re.IGNORECASE)
        return text

    async def _fetch_repo_context(self, full_name: str, question: str = "") -> str:
        from app.modules.github_trends.service import GitHubTrendsService
        from app.modules.rag.embeddings import embed_query
        from app.modules.rag.vector_store import search

        svc = GitHubTrendsService()
        repo = await svc.get_repo_by_full_name(full_name)
        if not repo:
            return None, f"❌ مش لاقي الريبو {full_name}"

        repo_query = question or f"شرح ريبو {full_name}"
        query_embedding = embed_query(repo_query)
        chunks = search(
            query_embedding,
            top_k=10,
            source_filter="repo_readme",
            extra_filters={"full_name": full_name},
        )

        if chunks:
            context_parts = []
            for c in chunks:
                sanitized = self._strip_images(c['text'])
                context_parts.append(f"[Chunk {c.get('chunk_index', 0)}] {sanitized}")
            context_str = "\n\n".join(context_parts)
        else:
            readme = repo.get("readme_content", "")
            if readme:
                readme = self._strip_images(readme)
            if readme and len(readme) > 8000:
                context_str = readme[:8000] + "\n\n...(اختصار)"
            elif readme:
                context_str = readme
            else:
                context_str = self._strip_images(repo.get("description", "") or "No README available.")

        return repo, context_str

    async def _build_messages(
        self,
        question: str,
        history: List[Message],
        context: Optional[List[Dict]] = None,
        lead_data: Optional[Dict] = None,
        summary: str = "",
    ) -> List[Message]:
        pass  # Not used — we use custom message building

    async def process(
        self,
        question: str,
        history: List[Message],
        context: Optional[List[Dict]] = None,
        lead_data: Optional[Dict] = None,
        summary: str = "",
    ) -> AgentResult:
        from app.services.ai.manager import ai_manager

        repo, context_str = await self._fetch_repo_context(question)
        if repo is None:
            return AgentResult(content=context_str, agent_name=self.name)

        system_content = EXPLAIN_REPO_SYSTEM_PROMPT.format(context=context_str)
        user_message = f"شرحتلي الـ GitHub repository {question} بالعامية المصرية"

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_message},
        ]

        content = await ai_manager.generate(messages=messages, temperature=0.5, max_tokens=1024)
        return AgentResult(content=content, agent_name=self.name)

    async def process_stream(
        self,
        full_name: str,
        question: str = "",
    ):
        import logging
        logger = logging.getLogger(__name__)

        try:
            repo, context_str = await self._fetch_repo_context(full_name, question)
        except Exception as e:
            logger.error("Fetch repo context failed for '%s': %s", full_name, e, exc_info=True)
            yield f"⚠️ فشل في جلب بيانات الريبو: {e}"
            return

        if repo is None:
            yield context_str
            return

        user_message = question or f"شرحتلي الـ GitHub repository {full_name} بالعامية المصرية"
        desc = self._strip_images(repo.get("description", "") or "No description")

        contexts_to_try = [
            ("full readme", context_str),
            ("description only", desc),
            ("no context", f"Repository: {full_name}"),
        ]

        from app.services.ai.manager import ai_manager

        for ctx_label, ctx_text in contexts_to_try:
            try:
                system_content = EXPLAIN_REPO_SYSTEM_PROMPT.format(context=ctx_text)
                async for chunk in ai_manager.generate_stream(
                    messages=[{"role": "system", "content": system_content}, {"role": "user", "content": user_message}],
                    temperature=0.5,
                    max_tokens=1024,
                ):
                    yield chunk
                return
            except Exception as e:
                logger.warning("explain_repo with %s failed for '%s': %s", ctx_label, full_name, e)
                continue

        yield f"\n\n⚠️ عذرًا، حصل مشكلة في الاتصال بـ AI بعد محاولة كل المستويات."
