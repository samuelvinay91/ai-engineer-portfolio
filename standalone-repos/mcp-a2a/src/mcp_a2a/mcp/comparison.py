"""MCP vs Traditional REST API comparison module.

Demonstrates the same functionality implemented through both paradigms and
provides structured metrics to illustrate the trade-offs between MCP's dynamic
tool discovery model and a conventional REST API.
"""

from __future__ import annotations

import time
from typing import Any

import httpx
import structlog
from pydantic import BaseModel

from mcp_a2a.mcp.server import execute_tool, list_tools

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Comparison result models
# ---------------------------------------------------------------------------


class ApproachResult(BaseModel):
    """Outcome of executing a single approach (REST or MCP)."""

    approach: str  # "rest" | "mcp"
    success: bool
    data: Any
    latency_ms: float
    discovery_required: bool
    schema_available: bool
    error: str | None = None


class ComparisonResult(BaseModel):
    """Side-by-side comparison of REST vs MCP for the same operation."""

    operation: str
    rest_result: ApproachResult
    mcp_result: ApproachResult
    analysis: ComparisonAnalysis


class ComparisonAnalysis(BaseModel):
    """Qualitative and quantitative analysis of the two approaches."""

    latency_winner: str
    discoverability: str
    flexibility: str
    type_safety: str
    summary: str


# ---------------------------------------------------------------------------
# REST-style approach (simulates a traditional HTTP API call)
# ---------------------------------------------------------------------------


class TraditionalRESTClient:
    """Simulated REST client that calls a fixed set of endpoints.

    In a real system this would issue HTTP requests; here we simulate the
    request/response cycle to provide a fair latency comparison.
    """

    # Pretend the client "knows" the API surface ahead of time
    _ENDPOINT_MAP: dict[str, str] = {
        "calculator": "/api/v1/calculator",
        "weather": "/api/v1/weather",
        "database_query": "/api/v1/database/query",
        "web_search": "/api/v1/search",
        "file_reader": "/api/v1/files/read",
    }

    async def call_endpoint(
        self,
        operation: str,
        payload: dict[str, Any],
    ) -> ApproachResult:
        """Simulate calling a traditional REST endpoint.

        The implementation re-uses the same handlers as MCP (since the
        business logic is identical) but wraps them to model the REST
        interaction pattern: no discovery, static URLs, payload validation
        done server-side.
        """
        start = time.perf_counter()

        endpoint = self._ENDPOINT_MAP.get(operation)
        if endpoint is None:
            elapsed = (time.perf_counter() - start) * 1000
            return ApproachResult(
                approach="rest",
                success=False,
                data=None,
                latency_ms=round(elapsed, 3),
                discovery_required=False,
                schema_available=False,
                error=f"Unknown endpoint for operation '{operation}'",
            )

        # Simulate REST overhead (serialization + network round-trip)
        _simulate_rest_overhead()

        result = execute_tool(operation, payload)
        elapsed = (time.perf_counter() - start) * 1000

        return ApproachResult(
            approach="rest",
            success=result.success,
            data=result.data,
            latency_ms=round(elapsed, 3),
            discovery_required=False,
            schema_available=False,  # REST: schema comes from docs (out-of-band)
        )


def _simulate_rest_overhead() -> None:
    """Add a small delay to represent serialization + HTTP overhead."""
    time.sleep(0.001)  # 1 ms


# ---------------------------------------------------------------------------
# MCP-style approach
# ---------------------------------------------------------------------------


class MCPApproachClient:
    """Demonstrates the MCP interaction pattern: discover, then invoke."""

    async def call_tool(
        self,
        operation: str,
        arguments: dict[str, Any],
    ) -> ApproachResult:
        """Call a tool the MCP way — discover first, then execute."""
        start = time.perf_counter()

        # Step 1: tool discovery (the defining MCP value-add)
        available_tools = list_tools()
        tool_names = [t["name"] for t in available_tools]

        if operation not in tool_names:
            elapsed = (time.perf_counter() - start) * 1000
            return ApproachResult(
                approach="mcp",
                success=False,
                data=None,
                latency_ms=round(elapsed, 3),
                discovery_required=True,
                schema_available=True,
                error=f"Tool '{operation}' not found; available: {tool_names}",
            )

        # Step 2: invoke
        result = execute_tool(operation, arguments)
        elapsed = (time.perf_counter() - start) * 1000

        return ApproachResult(
            approach="mcp",
            success=result.success,
            data=result.data,
            latency_ms=round(elapsed, 3),
            discovery_required=True,
            schema_available=True,
        )


# ---------------------------------------------------------------------------
# Comparison engine
# ---------------------------------------------------------------------------


async def compare_approaches(
    operation: str,
    arguments: dict[str, Any],
) -> ComparisonResult:
    """Execute the same *operation* via REST and MCP, returning analysis."""
    rest_client = TraditionalRESTClient()
    mcp_client = MCPApproachClient()

    rest_result = await rest_client.call_endpoint(operation, arguments)
    mcp_result = await mcp_client.call_tool(operation, arguments)

    analysis = _analyze(operation, rest_result, mcp_result)

    logger.info(
        "comparison.complete",
        operation=operation,
        rest_ms=rest_result.latency_ms,
        mcp_ms=mcp_result.latency_ms,
    )

    return ComparisonResult(
        operation=operation,
        rest_result=rest_result,
        mcp_result=mcp_result,
        analysis=analysis,
    )


def _analyze(
    operation: str,
    rest: ApproachResult,
    mcp: ApproachResult,
) -> ComparisonAnalysis:
    """Build a qualitative analysis comparing the two approaches."""
    latency_winner = "rest" if rest.latency_ms < mcp.latency_ms else "mcp"

    return ComparisonAnalysis(
        latency_winner=latency_winner,
        discoverability=(
            "MCP provides built-in tool and schema discovery (list_tools), "
            "whereas REST relies on external documentation (OpenAPI/Swagger)."
        ),
        flexibility=(
            "MCP tools can be added or removed at runtime and clients "
            "automatically discover changes. REST requires endpoint deployment "
            "and client-library updates."
        ),
        type_safety=(
            "MCP enforces input schemas via JSON Schema at the protocol level. "
            "REST relies on server-side validation and optional OpenAPI specs."
        ),
        summary=(
            f"For '{operation}': REST latency={rest.latency_ms:.1f}ms, "
            f"MCP latency={mcp.latency_ms:.1f}ms. "
            f"MCP adds discovery overhead but provides dynamic schema introspection, "
            f"standardized error handling, and runtime extensibility."
        ),
    )


# ---------------------------------------------------------------------------
# Full comparison report (all tools)
# ---------------------------------------------------------------------------


async def generate_full_report() -> dict[str, Any]:
    """Run a comparison for every registered tool and compile a report."""
    test_cases: list[tuple[str, dict[str, Any]]] = [
        ("calculator", {"operation": "add", "a": 42, "b": 17}),
        ("weather", {"city": "San Francisco"}),
        ("database_query", {"query_type": "count"}),
        ("web_search", {"query": "MCP protocol", "num_results": 2}),
    ]

    results: list[dict[str, Any]] = []
    for op, args in test_cases:
        comparison = await compare_approaches(op, args)
        results.append(comparison.model_dump())

    # Aggregate stats
    rest_total = sum(r["rest_result"]["latency_ms"] for r in results)
    mcp_total = sum(r["mcp_result"]["latency_ms"] for r in results)

    return {
        "comparisons": results,
        "aggregate": {
            "total_rest_latency_ms": round(rest_total, 3),
            "total_mcp_latency_ms": round(mcp_total, 3),
            "rest_avg_ms": round(rest_total / len(results), 3),
            "mcp_avg_ms": round(mcp_total / len(results), 3),
        },
        "key_differences": {
            "discovery": "MCP provides built-in tool discovery; REST needs OpenAPI/docs.",
            "schema": "MCP embeds input schemas in protocol; REST uses external specs.",
            "transport": "MCP supports stdio, SSE, HTTP; REST is HTTP-only.",
            "session": "MCP maintains stateful sessions; REST is stateless by default.",
            "extensibility": "MCP servers can add tools at runtime; REST requires redeploy.",
        },
    }
