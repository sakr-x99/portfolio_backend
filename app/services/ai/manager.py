import logging
from typing import List, Dict, Any, Optional
from .base import BaseAIProvider
from .groq_provider import GroqProvider
from .gemini_provider import GeminiProvider
from app.core.config import settings

logger = logging.getLogger(__name__)

class AIManager:
    def __init__(self):
        self.providers: Dict[str, BaseAIProvider] = {
            "groq": GroqProvider(),
            "gemini": GeminiProvider()
        }
        self.primary = settings.PRIMARY_AI_PROVIDER
        self.fallback = settings.FALLBACK_PROVIDER

    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        # Check if a specific provider is requested
        target_provider = kwargs.get("provider", self.primary)
        
        # Try requested provider
        try:
            provider = self.providers.get(target_provider)
            if not provider:
                raise Exception(f"Provider {target_provider} not found")
            
            return await provider.generate(messages, **kwargs)
        
        except Exception as e:
            logger.error(f"Primary AI provider ({self.primary}) failed: {str(e)}")
            
            # Try fallback provider
            try:
                fallback_provider = self.providers.get(self.fallback)
                if not fallback_provider:
                    raise Exception(f"Fallback provider {self.fallback} not found")
                
                logger.info(f"Switching to fallback AI provider: {self.fallback}")
                return await fallback_provider.generate(messages, **kwargs)
            
            except Exception as fe:
                logger.error(f"Fallback AI provider ({self.fallback}) failed: {str(fe)}")
                raise Exception(f"All AI providers failed. Last error: {str(fe)}")

    async def generate_stream(self, messages: List[Dict[str, str]], **kwargs):
        # Try primary provider
        try:
            provider = self.providers.get(self.primary)
            if not provider:
                raise Exception(f"Primary provider {self.primary} not found")
            
            logger.info(f"Streaming with primary AI provider: {self.primary}")
            async for chunk in provider.generate_stream(messages, **kwargs):
                yield chunk
        
        except Exception as e:
            logger.error(f"Primary AI provider ({self.primary}) streaming failed: {str(e)}")
            
            # Try fallback provider
            try:
                fallback_provider = self.providers.get(self.fallback)
                if not fallback_provider:
                    raise Exception(f"Fallback provider {self.fallback} not found")
                
                logger.info(f"Switching stream to fallback AI provider: {self.fallback}")
                async for chunk in fallback_provider.generate_stream(messages, **kwargs):
                    yield chunk
            
            except Exception as fe:
                logger.error(f"Fallback AI provider ({self.fallback}) streaming failed: {str(fe)}")
                raise Exception(f"All AI providers failed during streaming. Last error: {str(fe)}")

# Global instance
ai_manager = AIManager()
