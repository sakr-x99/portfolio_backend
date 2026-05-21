import logging
from .base import BaseAgent, AgentResult
from .registry import AgentRegistry, agent_registry
from .chat_agent import ChatAgent
from .services_agent import ServicesAgent
from .hr_agent import HRAgent
from .booking_agent import BookingAgent
from .explain_repo_agent import ExplainRepoAgent

logger = logging.getLogger(__name__)


def register_all_agents():
    """Register all agents with the global registry."""
    agent_registry.register(ChatAgent())
    agent_registry.register(ServicesAgent())
    agent_registry.register(HRAgent())
    agent_registry.register(BookingAgent())
    agent_registry.register(ExplainRepoAgent())
    logger.info("All agents registered: %s", list(agent_registry.agents.keys()))


__all__ = [
    "BaseAgent", "AgentResult",
    "AgentRegistry", "agent_registry",
    "ChatAgent", "ServicesAgent", "HRAgent", "BookingAgent", "ExplainRepoAgent",
    "register_all_agents",
]
