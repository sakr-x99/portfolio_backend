from typing import List, Dict, Any
from groq import AsyncGroq
from .base import BaseAIProvider
from app.core.config import settings

class GroqProvider(BaseAIProvider):
    def __init__(self, api_key: str = None, model: str = None):
        self.client = AsyncGroq(api_key=api_key or settings.GROQ_API_KEY)
        self.model = model or settings.GROQ_MODEL

    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        try:
            completion = await self.client.chat.completions.create(
                model=kwargs.get("model", self.model),
                messages=messages,
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 1024),
            )
            return completion.choices[0].message.content
        except Exception as e:
            raise Exception(f"Groq Error: {str(e)}")

    async def generate_stream(self, messages: List[Dict[str, str]], **kwargs):
        try:
            stream = await self.client.chat.completions.create(
                model=kwargs.get("model", self.model),
                messages=messages,
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 1024),
                stream=True,
            )
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            raise Exception(f"Groq Stream Error: {str(e)}")

    def get_name(self) -> str:
        return "groq"
