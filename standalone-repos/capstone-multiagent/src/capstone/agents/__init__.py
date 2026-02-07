"""Specialized AI agents for the multi-agent platform.

Each agent encapsulates domain expertise and exposes a uniform interface
(:class:`BaseAgent`) so the orchestrator can dispatch tasks polymorphically.
"""

from capstone.agents.analyst import AnalystAgent
from capstone.agents.base import AgentResult, AgentTask, BaseAgent
from capstone.agents.coder import CoderAgent
from capstone.agents.researcher import ResearcherAgent
from capstone.agents.writer import WriterAgent

__all__ = [
    "AgentResult",
    "AgentTask",
    "AnalystAgent",
    "BaseAgent",
    "CoderAgent",
    "ResearcherAgent",
    "WriterAgent",
]
