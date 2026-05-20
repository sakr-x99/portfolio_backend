from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncGenerator

Message = Dict[str, str]

class BaseAIProvider(ABC):
    @abstractmethod
    async def generate(self, messages: List[Message], **kwargs: Any) -> str:
        ...

    @abstractmethod
    async def generate_stream(
        self, messages: List[Message], **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        ...

    @abstractmethod
    def get_name(self) -> str:
        ...

    async def close(self):
        pass
