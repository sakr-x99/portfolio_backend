import logging
from typing import List, Dict, Any, Optional, AsyncGenerator
from .base import BaseAIProvider, Message
from .groq_provider import GroqProvider
from .gemini_provider import GeminiProvider
from .exceptions import (
    AIProviderError,
    APIKeyMissingError,
    ProviderNotFoundError,
    GenerationError,
    StreamingError,
    AllProvidersFailedError,
)
from app.core.config import settings

logger = logging.getLogger(__name__)


class AIManager:
    def __init__(
        self,
        providers: Dict[str, BaseAIProvider] | None = None,
        config_profile: str = "primary",
    ):
        if config_profile == "secondary":
            groq_key = settings.GROQ_API_KEY_2
            gemini_key = settings.GEMINI_API_KEY_2
            groq_model = settings.GROQ_MODEL_2
            gemini_model = settings.GEMINI_MODEL_2
            self.primary = settings.PRIMARY_AI_PROVIDER_2
            self.fallback = settings.FALLBACK_PROVIDER_2
        else:
            groq_key = settings.GROQ_API_KEY
            gemini_key = settings.GEMINI_API_KEY
            groq_model = settings.GROQ_MODEL
            gemini_model = settings.GEMINI_MODEL
            self.primary = settings.PRIMARY_AI_PROVIDER
            self.fallback = settings.FALLBACK_PROVIDER

        if providers is not None:
            self.providers = providers
        else:
            self.providers = {
                "groq": GroqProvider(api_key=groq_key, model=groq_model),
                "gemini": GeminiProvider(api_key=gemini_key, model=gemini_model),
            }

    def register_provider(self, name: str, provider: BaseAIProvider):
        self.providers[name] = provider
        logger.info("Provider '%s' registered (%s)", name, provider.get_name())

    def _resolve(self, name: str) -> BaseAIProvider:
        provider = self.providers.get(name)
        if not provider:
            raise ProviderNotFoundError(name)
        return provider

    async def generate(self, messages: List[Message], **kwargs) -> str:
        target = kwargs.pop("provider", self.primary)

        try:
            provider = self._resolve(target)
            logger.debug("Generating with provider: %s", provider.get_name())
            return await provider.generate(messages, **kwargs)
        except (APIKeyMissingError, ProviderNotFoundError, GenerationError) as e:
            logger.error("Primary provider '%s' failed: %s", target, e)

        try:
            fallback = self._resolve(self.fallback)
            logger.info("Falling back to provider: %s", self.fallback)
            return await fallback.generate(messages, **kwargs)
        except (APIKeyMissingError, ProviderNotFoundError, GenerationError) as fe:
            logger.error("Fallback provider '%s' failed: %s", self.fallback, fe)
            raise AllProvidersFailedError(self.primary, self.fallback, str(fe)) from fe

    async def generate_stream(
        self, messages: List[Message], **kwargs
    ) -> AsyncGenerator[str, None]:
        target = kwargs.pop("provider", self.primary)

        try:
            provider = self._resolve(target)
            logger.debug("Streaming with provider: %s", provider.get_name())
            async for chunk in provider.generate_stream(messages, **kwargs):
                yield chunk
            return
        except (APIKeyMissingError, ProviderNotFoundError, StreamingError) as e:
            logger.error("Primary provider '%s' streaming failed: %s", target, e)

        try:
            fallback = self._resolve(self.fallback)
            logger.info("Falling back stream to provider: %s", self.fallback)
            async for chunk in fallback.generate_stream(messages, **kwargs):
                yield chunk
        except (APIKeyMissingError, ProviderNotFoundError, StreamingError) as fe:
            logger.error("Fallback provider '%s' streaming failed: %s", self.fallback, fe)
            raise AllProvidersFailedError(self.primary, self.fallback, str(fe)) from fe

    async def close(self):
        for name, provider in self.providers.items():
            try:
                await provider.close()
                logger.debug("Closed provider: %s", name)
            except Exception as e:
                logger.warning("Error closing provider '%s': %s", name, e)


ai_manager = AIManager(config_profile="primary")

github_trends_ai = AIManager(config_profile="secondary")
