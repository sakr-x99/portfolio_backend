import logging
from typing import List, Dict, Any, AsyncGenerator
from .base import BaseAIProvider, Message
from .exceptions import APIKeyMissingError, GenerationError, StreamingError
from app.core.config import settings

logger = logging.getLogger(__name__)

class GroqProvider(BaseAIProvider):
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or settings.GROQ_API_KEY
        self.model_name = model or settings.GROQ_MODEL
        self._client = None

    def _check_key(self):
        if not self.api_key:
            raise APIKeyMissingError("Groq")

    def _get_client(self):
        self._check_key()
        if self._client is None:
            from groq import AsyncGroq
            self._client = AsyncGroq(api_key=self.api_key)
        return self._client

    async def generate(self, messages: List[Message], **kwargs) -> str:
        try:
            client = self._get_client()
            completion = await client.chat.completions.create(
                model=kwargs.get("model", self.model_name),
                messages=messages,
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 1024),
            )
            return completion.choices[0].message.content
        except APIKeyMissingError:
            raise
        except Exception as e:
            raise GenerationError("Groq", e) from e

    async def generate_stream(
        self, messages: List[Message], **kwargs
    ) -> AsyncGenerator[str, None]:
        try:
            client = self._get_client()
            stream = await client.chat.completions.create(
                model=kwargs.get("model", self.model_name),
                messages=messages,
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 1024),
                stream=True,
            )
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except APIKeyMissingError:
            raise
        except Exception as e:
            raise StreamingError("Groq", e) from e

    def get_name(self) -> str:
        return "groq"

    async def close(self):
        if self._client and hasattr(self._client, 'close'):
            await self._client.close()
        self._client = None
