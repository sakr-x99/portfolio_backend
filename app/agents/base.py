from abc import ABC, abstractmethod
from typing import List, Dict, Any, AsyncGenerator, Optional
from dataclasses import dataclass, field

Message = Dict[str, str]


@dataclass
class AgentResult:
    content: str
    agent_name: str
    sources: List[str] = field(default_factory=list)
    lead_data: Dict = field(default_factory=dict)
    summary: str = ""
    cached: bool = False


class BaseAgent(ABC):
    name: str = "base"

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        ...

    @abstractmethod
    async def process(
        self,
        question: str,
        history: List[Message],
        context: Optional[List[Dict]] = None,
        lead_data: Optional[Dict] = None,
        summary: str = "",
    ) -> AgentResult:
        ...

    @abstractmethod
    async def process_stream(
        self,
        question: str,
        history: List[Message],
        context: Optional[List[Dict]] = None,
        lead_data: Optional[Dict] = None,
        summary: str = "",
    ) -> AsyncGenerator[str, None]:
        ...
        yield ""
