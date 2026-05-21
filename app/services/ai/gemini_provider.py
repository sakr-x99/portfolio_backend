import logging
from typing import List, Dict, Any, AsyncGenerator
from .base import BaseAIProvider, Message
from .exceptions import APIKeyMissingError, GenerationError, StreamingError
from app.core.config import settings

logger = logging.getLogger(__name__)

class GeminiProvider(BaseAIProvider):
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model_name = model or settings.GEMINI_MODEL
        self._genai = None
        self._model_cache: dict = {}

    def _check_key(self):
        if not self.api_key:
            raise APIKeyMissingError("Gemini")

    def _get_genai(self):
        self._check_key()
        if self._genai is None:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._genai = genai
        return self._genai

    def _convert_messages(self, messages: List[Message]) -> tuple[str, list]:
        genai = self._get_genai()
        system_instruction = ""
        contents = []
        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            else:
                role = "user" if msg["role"] == "user" else "model"
                contents.append({
                    "role": role,
                    "parts": [genai.types.Part(text=msg["content"])],
                })
        return system_instruction, contents

    def _get_model(self, model_name: str, system_instruction: str = ""):
        key = (model_name, system_instruction)
        if key not in self._model_cache:
            genai = self._get_genai()
            kwargs = {"model_name": model_name}
            if system_instruction:
                kwargs["system_instruction"] = system_instruction
            self._model_cache[key] = genai.GenerativeModel(**kwargs)
        return self._model_cache[key]

    def _build_config(self, kwargs: dict):
        genai = self._get_genai()
        return genai.types.GenerationConfig(
            temperature=kwargs.get("temperature", 0.7),
            max_output_tokens=kwargs.get("max_tokens", 1024),
        )

    async def generate(self, messages: List[Message], **kwargs) -> str:
        try:
            genai = self._get_genai()
            model_name = kwargs.get("model", self.model_name)
            system_instruction, contents = self._convert_messages(messages)
            model = self._get_model(model_name, system_instruction)
            config = self._build_config(kwargs)
            response = await model.generate_content_async(
                contents, generation_config=config
            )
            return response.text
        except APIKeyMissingError:
            raise
        except Exception as e:
            raise GenerationError("Gemini", e) from e

    async def generate_stream(
        self, messages: List[Message], **kwargs
    ) -> AsyncGenerator[str, None]:
        try:
            genai = self._get_genai()
            model_name = kwargs.get("model", self.model_name)
            system_instruction, contents = self._convert_messages(messages)
            model = self._get_model(model_name, system_instruction)
            config = self._build_config(kwargs)
            response = await model.generate_content_async(
                contents, generation_config=config, stream=True
            )
            async for chunk in response:
                if chunk.text:
                    yield chunk.text
        except APIKeyMissingError:
            raise
        except Exception as e:
            raise StreamingError("Gemini", e) from e

    def get_name(self) -> str:
        return "gemini"

    async def close(self):
        self._model_cache.clear()
        self._genai = None
