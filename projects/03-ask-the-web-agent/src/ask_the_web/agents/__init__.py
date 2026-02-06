"""Agent modules for the Ask-the-Web system."""

from ask_the_web.agents.fact_checker import FactCheckerAgent
from ask_the_web.agents.router import QueryRouterAgent
from ask_the_web.agents.searcher import WebSearchAgent
from ask_the_web.agents.synthesizer import SynthesisAgent

__all__ = [
    "QueryRouterAgent",
    "WebSearchAgent",
    "SynthesisAgent",
    "FactCheckerAgent",
]
