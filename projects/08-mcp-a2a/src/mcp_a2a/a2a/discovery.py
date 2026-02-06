"""Agent Discovery Service for the A2A protocol.

Discovers agents by fetching their well-known AgentCard endpoints,
performs capability matching, health checking, and maintains a live
registry of known agents.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
import structlog

from mcp_a2a.a2a.agent_card import AgentCard, AgentCardRegistry

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Health status model
# ---------------------------------------------------------------------------


@dataclass
class AgentHealth:
    """Health status for a discovered agent."""

    url: str
    healthy: bool
    latency_ms: float
    checked_at: float
    error: str | None = None


# ---------------------------------------------------------------------------
# Discovery service
# ---------------------------------------------------------------------------


class AgentDiscoveryService:
    """Discovers, monitors, and indexes A2A-compliant agents.

    Supports:
    - Fetching agent cards from well-known URLs
    - Periodic health checks
    - Capability-based agent matching
    - Dynamic registration and deregistration
    """

    def __init__(
        self,
        registry: AgentCardRegistry | None = None,
        health_check_interval: float = 60.0,
        request_timeout: float = 5.0,
    ) -> None:
        self._registry = registry or AgentCardRegistry()
        self._health_cache: dict[str, AgentHealth] = {}
        self._health_check_interval = health_check_interval
        self._request_timeout = request_timeout
        self._background_task: asyncio.Task[None] | None = None

    @property
    def registry(self) -> AgentCardRegistry:
        return self._registry

    # -- Discovery -----------------------------------------------------------

    async def discover_agent(self, base_url: str) -> AgentCard | None:
        """Discover a single agent by fetching its AgentCard.

        Also performs an initial health check.
        """
        card = await self._registry.fetch_card(base_url, force=True)
        if card is not None:
            await self.check_health(base_url)
            logger.info(
                "discovery.agent_found",
                name=card.name,
                url=base_url,
                capabilities=len(card.capabilities),
                skills=len(card.skills),
            )
        return card

    async def discover_agents(self, urls: list[str]) -> list[AgentCard]:
        """Discover multiple agents concurrently."""
        tasks = [self.discover_agent(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if isinstance(r, AgentCard)]

    # -- Health checking -----------------------------------------------------

    async def check_health(self, base_url: str) -> AgentHealth:
        """Perform a health check against an agent's ``/health`` endpoint."""
        url = f"{base_url.rstrip('/')}/health"
        start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self._request_timeout) as client:
                resp = await client.get(url)
                latency = (time.perf_counter() - start) * 1000
                healthy = resp.status_code == 200
                health = AgentHealth(
                    url=base_url,
                    healthy=healthy,
                    latency_ms=round(latency, 2),
                    checked_at=time.time(),
                )
        except Exception as exc:
            latency = (time.perf_counter() - start) * 1000
            health = AgentHealth(
                url=base_url,
                healthy=False,
                latency_ms=round(latency, 2),
                checked_at=time.time(),
                error=str(exc),
            )

        self._health_cache[base_url] = health
        return health

    async def check_all_health(self) -> list[AgentHealth]:
        """Check health of all registered agents concurrently."""
        agents = self._registry.list_agents()
        if not agents:
            return []
        tasks = [self.check_health(card.url) for card in agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if isinstance(r, AgentHealth)]

    def get_health(self, url: str) -> AgentHealth | None:
        """Get cached health status for an agent."""
        return self._health_cache.get(url)

    # -- Capability matching -------------------------------------------------

    def find_agents_for_task(self, task_description: str) -> list[AgentCard]:
        """Find agents that might handle a task based on keyword matching.

        Uses a simple keyword-to-tag/capability matching heuristic.
        In production this would use embeddings or an LLM for semantic matching.
        """
        description_lower = task_description.lower()
        scored: list[tuple[float, AgentCard]] = []

        for card in self._registry.list_agents():
            score = 0.0

            # Match against capability names/descriptions
            for cap in card.capabilities:
                if cap.name.lower() in description_lower:
                    score += 2.0
                for word in cap.description.lower().split():
                    if len(word) > 3 and word in description_lower:
                        score += 0.5

            # Match against skill names/tags
            for skill in card.skills:
                for tag in skill.tags:
                    if tag.lower() in description_lower:
                        score += 1.5
                if skill.name.lower() in description_lower:
                    score += 2.0

            # Boost healthy agents
            health = self._health_cache.get(card.url)
            if health and health.healthy:
                score += 1.0

            if score > 0:
                scored.append((score, card))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [card for _, card in scored]

    def find_by_capability(self, capability_name: str) -> list[AgentCard]:
        """Find agents with a specific capability."""
        return self._registry.find_by_capability(capability_name)

    def find_by_skill_tag(self, tag: str) -> list[AgentCard]:
        """Find agents that have a skill matching the given tag."""
        return self._registry.find_by_skill_tag(tag)

    # -- Background health monitoring ----------------------------------------

    async def start_health_monitor(self) -> None:
        """Start periodic background health checking."""
        if self._background_task is not None:
            return

        async def _monitor() -> None:
            while True:
                try:
                    await self.check_all_health()
                except Exception as exc:
                    logger.error("discovery.health_monitor.error", error=str(exc))
                await asyncio.sleep(self._health_check_interval)

        self._background_task = asyncio.create_task(_monitor())
        logger.info("discovery.health_monitor.started")

    async def stop_health_monitor(self) -> None:
        """Stop the background health monitor."""
        if self._background_task is not None:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
            self._background_task = None
            logger.info("discovery.health_monitor.stopped")

    # -- Serialization -------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """Return a summary of the discovery service state."""
        agents = self._registry.list_agents()
        return {
            "total_agents": len(agents),
            "healthy_agents": sum(
                1 for a in agents if self._health_cache.get(a.url, AgentHealth(
                    url=a.url, healthy=False, latency_ms=0, checked_at=0
                )).healthy
            ),
            "agents": [
                {
                    "name": a.name,
                    "url": a.url,
                    "capabilities": len(a.capabilities),
                    "skills": len(a.skills),
                    "health": (
                        {
                            "healthy": h.healthy,
                            "latency_ms": h.latency_ms,
                            "checked_at": h.checked_at,
                        }
                        if (h := self._health_cache.get(a.url))
                        else None
                    ),
                }
                for a in agents
            ],
        }
