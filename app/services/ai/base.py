from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseAIProvider(ABC):
    @abstractmethod
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate a response from the AI provider."""
        pass

    @abstractmethod
    async def generate_stream(self, messages: List[Dict[str, str]], **kwargs):
        """Generate a streaming response from the AI provider."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return the provider name."""
        pass
