"""AgentCard implementation following the Google A2A specification.

An AgentCard is a machine-readable manifest that advertises an agent's
identity, capabilities, skills, and authentication requirements.  It is
served at ``/.well-known/agent.json`` so that other agents (or orchestrators)
can discover and interact with the agent dynamically.
"""

from __future__ import annotations

import time
from typing import Any

import httpx
import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Core A2A models (mirrors the Google A2A AgentCard spec)
# ---------------------------------------------------------------------------


class Capability(BaseModel):
    """A single capability offered by an agent."""

    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)


class Skill(BaseModel):
    """A higher-level skill that may combine multiple capabilities."""

    id: str
    name: str
    description: str
    tags: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)


class AuthConfig(BaseModel):
    """Authentication configuration for the agent."""

    type: str = "bearer"  # "bearer" | "api_key" | "oauth2" | "none"
    credentials_url: str | None = None
    scopes: list[str] = Field(default_factory=list)


class AgentProvider(BaseModel):
    """Organization or individual that hosts the agent."""

    organization: str
    url: str | None = None


class AgentCard(BaseModel):
    """Machine-readable agent manifest (served at /.well-known/agent.json).

    See https://google.github.io/A2A/ for the specification.
    """

    name: str
    description: str
    version: str
    url: str
    capabilities: list[Capability] = Field(default_factory=list)
    skills: list[Skill] = Field(default_factory=list)
    authentication: AuthConfig | None = None
    supported_modalities: list[str] = Field(default_factory=lambda: ["text"])
    provider: AgentProvider | None = None
    documentation_url: str | None = None
    default_input_modes: list[str] = Field(default_factory=lambda: ["text"])
    default_output_modes: list[str] = Field(default_factory=lambda: ["text"])


# ---------------------------------------------------------------------------
# Default agent card for this service
# ---------------------------------------------------------------------------


def build_default_agent_card(base_url: str = "http://localhost:8008") -> AgentCard:
    """Construct the default AgentCard for the MCP-A2A service."""
    return AgentCard(
        name="AI Portfolio Agent",
        description=(
            "Multi-capability agent supporting data analysis, web search, "
            "database queries, and general assistance via MCP tools."
        ),
        version="0.1.0",
        url=base_url,
        capabilities=[
            Capability(
                name="math_computation",
                description="Perform mathematical calculations.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "operation": {"type": "string"},
                        "a": {"type": "number"},
                        "b": {"type": "number"},
                    },
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "result": {"type": "number"},
                    },
                },
            ),
            Capability(
                name="weather_lookup",
                description="Look up weather conditions for a city.",
                input_schema={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "temp_f": {"type": "number"},
                        "condition": {"type": "string"},
                    },
                },
            ),
            Capability(
                name="database_query",
                description="Query an employee database.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query_type": {"type": "string", "enum": ["list", "count", "filter"]},
                    },
                },
                output_schema={
                    "type": "object",
                    "properties": {"records": {"type": "array"}},
                },
            ),
            Capability(
                name="web_search",
                description="Search the web for information.",
                input_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
                output_schema={
                    "type": "object",
                    "properties": {"results": {"type": "array"}},
                },
            ),
            Capability(
                name="file_read",
                description="Read file contents (sandboxed).",
                input_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
                output_schema={
                    "type": "object",
                    "properties": {"content": {"type": "string"}},
                },
            ),
        ],
        skills=[
            Skill(
                id="data-analysis",
                name="Data Analysis",
                description="Analyze datasets and produce insights including summaries and statistics.",
                tags=["analysis", "data", "statistics"],
                examples=[
                    "Analyze this CSV data and find trends.",
                    "What are the key statistics for the sales data?",
                ],
            ),
            Skill(
                id="research",
                name="Web Research",
                description="Search the web and compile research summaries.",
                tags=["search", "research", "web"],
                examples=[
                    "Research the latest developments in AI agents.",
                    "Find information about the A2A protocol.",
                ],
            ),
            Skill(
                id="computation",
                name="Mathematical Computation",
                description="Perform mathematical calculations and operations.",
                tags=["math", "computation", "calculator"],
                examples=[
                    "Calculate the compound interest on $1000 at 5% for 10 years.",
                    "What is the square root of 144?",
                ],
            ),
        ],
        authentication=AuthConfig(type="bearer"),
        supported_modalities=["text"],
        provider=AgentProvider(
            organization="AI Engineer Portfolio",
            url="https://github.com/ai-engineer-portfolio",
        ),
    )


# ---------------------------------------------------------------------------
# AgentCard Registry — discovers and stores remote agent cards
# ---------------------------------------------------------------------------


class AgentCardRegistry:
    """Discovers, caches, and queries agent cards from remote agents.

    Agent cards are fetched from the well-known URL
    ``<agent_base_url>/.well-known/agent.json``.
    """

    def __init__(self, cache_ttl_seconds: int = 300) -> None:
        self._cards: dict[str, AgentCard] = {}
        self._fetch_times: dict[str, float] = {}
        self._cache_ttl = cache_ttl_seconds

    # -- Registration --------------------------------------------------------

    def register(self, card: AgentCard) -> None:
        """Register an agent card directly (e.g., for the local agent)."""
        self._cards[card.url] = card
        self._fetch_times[card.url] = time.time()
        logger.info("agent_card.registered", name=card.name, url=card.url)

    def unregister(self, url: str) -> bool:
        """Remove an agent from the registry."""
        removed = self._cards.pop(url, None) is not None
        self._fetch_times.pop(url, None)
        return removed

    # -- Fetching ------------------------------------------------------------

    async def fetch_card(self, base_url: str, force: bool = False) -> AgentCard | None:
        """Fetch an agent card from ``<base_url>/.well-known/agent.json``.

        Results are cached for ``cache_ttl_seconds``.
        """
        # Check cache
        if not force and base_url in self._cards:
            age = time.time() - self._fetch_times.get(base_url, 0)
            if age < self._cache_ttl:
                return self._cards[base_url]

        well_known_url = f"{base_url.rstrip('/')}/.well-known/agent.json"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(well_known_url)
                resp.raise_for_status()
                card = AgentCard.model_validate(resp.json())
                self._cards[base_url] = card
                self._fetch_times[base_url] = time.time()
                logger.info("agent_card.fetched", name=card.name, url=base_url)
                return card
        except Exception as exc:
            logger.warning("agent_card.fetch_failed", url=well_known_url, error=str(exc))
            return None

    # -- Querying ------------------------------------------------------------

    def list_agents(self) -> list[AgentCard]:
        """Return all known agent cards."""
        return list(self._cards.values())

    def find_by_capability(self, capability_name: str) -> list[AgentCard]:
        """Find agents that advertise a capability matching *capability_name*."""
        return [
            card
            for card in self._cards.values()
            if any(c.name == capability_name for c in card.capabilities)
        ]

    def find_by_skill_tag(self, tag: str) -> list[AgentCard]:
        """Find agents that have a skill tagged with *tag*."""
        return [
            card
            for card in self._cards.values()
            if any(tag in skill.tags for skill in card.skills)
        ]

    def find_by_modality(self, modality: str) -> list[AgentCard]:
        """Find agents that support a given modality (text, image, audio)."""
        return [
            card
            for card in self._cards.values()
            if modality in card.supported_modalities
        ]

    def get_card(self, url: str) -> AgentCard | None:
        """Get a cached agent card by URL."""
        return self._cards.get(url)
