import logging
from typing import Dict, List, Optional, AsyncGenerator
from .base import BaseAgent, AgentResult, Message

logger = logging.getLogger(__name__)


class AgentRegistry:
    def __init__(self):
        self._agents: Dict[str, BaseAgent] = {}

    def register(self, agent: BaseAgent):
        self._agents[agent.name] = agent
        logger.info("Agent registered: %s", agent.name)

    def get(self, name: str) -> Optional[BaseAgent]:
        return self._agents.get(name)

    @property
    def agents(self) -> Dict[str, BaseAgent]:
        return dict(self._agents)

    async def route(self, question: str, history: List[Message]) -> str:
        from app.services.ai.manager import ai_manager

        router_prompt = """You are a smart router. Based on the user's message and conversation history, 
classify which agent should handle this. Return ONLY the agent name, nothing else.

Available agents:
- chat: General questions about Mohamed Sakr's skills, projects, education, experience, background, general conversation
- services: Questions about offered services, pricing, hiring, freelance work, business inquiries, collaboration
- hr_interview: HR-related discussions, interview questions, technical assessment, job applications
- booking: Scheduling meetings, providing contact info, lead capture (name, email, phone, meeting time)
- explain_repo: Explaining GitHub repositories (this is set explicitly, not auto-routed)

Rules:
- If user asks about services OR hiring/freelance → 'services'
- If user provides contact info or wants to book → 'booking'
- If user talks about job interview → 'hr_interview'
- If unsure or general chat → 'chat'
- Return ONLY the single word."""

        messages = [{"role": "system", "content": router_prompt}]
        for msg in history[-4:]:
            messages.append(msg)
        messages.append({"role": "user", "content": question[:500]})

        try:
            result = await ai_manager.generate(
                messages=messages, temperature=0.0, max_tokens=20
            )
            agent_name = result.strip().lower()
            if agent_name in self._agents:
                logger.info("Routed to agent: %s", agent_name)
                return agent_name
            logger.warning("Unknown agent '%s', defaulting to chat", agent_name)
            return "chat"
        except Exception as e:
            logger.error("Routing failed: %s, defaulting to chat", e)
            return "chat"

    async def process(
        self,
        question: str,
        history: List[Message],
        context: Optional[List[Dict]] = None,
        lead_data: Optional[Dict] = None,
        summary: str = "",
        force_agent: Optional[str] = None,
    ) -> AgentResult:
        agent_name = force_agent or await self.route(question, history)
        agent = self._agents.get(agent_name)
        if not agent:
            logger.warning("Agent '%s' not found, falling back to chat", agent_name)
            agent = self._agents["chat"]

        return await agent.process(question, history, context, lead_data, summary)

    async def process_stream(
        self,
        question: str,
        history: List[Message],
        context: Optional[List[Dict]] = None,
        lead_data: Optional[Dict] = None,
        summary: str = "",
        force_agent: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        agent_name = force_agent or await self.route(question, history)
        agent = self._agents.get(agent_name)
        if not agent:
            agent = self._agents["chat"]

        async for chunk in agent.process_stream(question, history, context, lead_data, summary):
            yield chunk


agent_registry = AgentRegistry()
