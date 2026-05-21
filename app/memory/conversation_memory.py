import json
import hashlib
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

Message = Dict[str, str]


class ConversationMemory:
    """
    Conversation memory using Redis (fallback to in-memory dict).
    Stores:
    - messages: list of {role, content}
    - summary: conversation summary (auto-generated)
    - lead_data: extracted lead info
    - metadata: session metadata (created_at, last_active, etc.)
    """

    MEMORY_TTL = 86400 * 7  # 7 days

    def __init__(self):
        self._memory_store: Dict[str, Dict] = {}
        self._redis = None

    def _get_redis(self):
        if self._redis is None:
            try:
                from app.core.cache import get_redis
                self._redis = get_redis()
            except Exception:
                self._redis = None
        return self._redis

    def _session_key(self, session_id: str, suffix: str = "data") -> str:
        return f"memory:{session_id}:{suffix}"

    async def get_messages(self, session_id: str, limit: int = 20) -> List[Message]:
        data = await self._load(session_id)
        messages = data.get("messages", [])
        return messages[-limit:] if limit else messages

    async def add_message(self, session_id: str, role: str, content: str):
        data = await self._load(session_id)
        messages = data.get("messages", [])
        messages.append({"role": role, "content": content})
        data["messages"] = messages
        data["last_active"] = datetime.now().isoformat()
        await self._save(session_id, data)

    async def get_summary(self, session_id: str) -> str:
        data = await self._load(session_id)
        return data.get("summary", "")

    async def set_summary(self, session_id: str, summary: str):
        data = await self._load(session_id)
        data["summary"] = summary
        await self._save(session_id, data)

    async def get_lead_data(self, session_id: str) -> Dict:
        data = await self._load(session_id)
        return data.get("lead_data", {})

    async def set_lead_data(self, session_id: str, lead_data: Dict):
        data = await self._load(session_id)
        existing = data.get("lead_data", {})
        existing.update(lead_data)
        data["lead_data"] = existing
        await self._save(session_id, data)

    async def get_metadata(self, session_id: str) -> Dict:
        data = await self._load(session_id)
        return data.get("metadata", {})

    async def set_metadata(self, session_id: str, **kwargs):
        data = await self._load(session_id)
        meta = data.get("metadata", {})
        meta.update(kwargs)
        data["metadata"] = meta
        await self._save(session_id, data)

    async def clear(self, session_id: str):
        r = self._get_redis()
        if r:
            try:
                keys = r.keys(f"memory:{session_id}:*")
                if keys:
                    r.delete(*keys)
            except Exception:
                pass
        self._memory_store.pop(session_id, None)

    async def get_all_sessions(self) -> List[str]:
        r = self._get_redis()
        if r:
            try:
                keys = r.keys("memory:*:data")
                return list(set(k.split(":")[1] for k in keys))
            except Exception:
                pass
        return list(self._memory_store.keys())

    async def auto_summarize(self, session_id: str) -> str:
        """Auto-generate a summary of the conversation using AI."""
        messages = await self.get_messages(session_id, limit=30)
        if len(messages) < 4:
            return await self.get_summary(session_id)

        from app.services.ai.manager import ai_manager

        summary_prompt = "Summarize the following conversation in one short paragraph in English."
        chat_messages = [{"role": "system", "content": summary_prompt}]
        chat_messages.extend(messages[-10:])

        try:
            summary = await ai_manager.generate(
                messages=chat_messages, temperature=0.3, max_tokens=150
            )
            await self.set_summary(session_id, summary)
            logger.info("Auto-summary for %s: %d chars", session_id, len(summary))
            return summary
        except Exception as e:
            logger.error("Auto-summarize failed: %s", e)
            return await self.get_summary(session_id)

    async def _load(self, session_id: str) -> Dict:
        r = self._get_redis()
        if r:
            try:
                key = self._session_key(session_id)
                data = r.get(key)
                if data:
                    parsed = json.loads(data)
                    # Refresh TTL on access
                    r.expire(key, self.MEMORY_TTL)
                    return parsed
            except Exception as e:
                logger.warning("Redis load failed for %s: %s", session_id, e)

        return self._memory_store.get(session_id, {
            "messages": [],
            "summary": "",
            "lead_data": {},
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "last_active": datetime.now().isoformat(),
            }
        })

    async def _save(self, session_id: str, data: Dict):
        r = self._get_redis()
        if r:
            try:
                key = self._session_key(session_id)
                r.setex(key, self.MEMORY_TTL, json.dumps(data, ensure_ascii=False))
                return
            except Exception as e:
                logger.warning("Redis save failed for %s: %s", session_id, e)

        self._memory_store[session_id] = data

    @staticmethod
    def generate_session_id() -> str:
        import uuid
        return uuid.uuid4().hex[:16]


conversation_memory = ConversationMemory()
