"""MCP Server implementation using the official mcp Python SDK.

Exposes tools, resources, and prompt templates through the Model Context Protocol,
allowing LLM applications to discover and invoke capabilities dynamically.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import structlog
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    EmbeddedResource,
    GetPromptResult,
    Prompt,
    PromptArgument,
    PromptMessage,
    Resource,
    TextContent,
    TextResourceContents,
    Tool,
)
from pydantic import BaseModel

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Domain helpers – these would call real services in production
# ---------------------------------------------------------------------------

MOCK_WEATHER_DATA: dict[str, dict[str, Any]] = {
    "san francisco": {"temp_f": 62, "condition": "Foggy", "humidity": 78},
    "new york": {"temp_f": 45, "condition": "Cloudy", "humidity": 55},
    "london": {"temp_f": 48, "condition": "Rainy", "humidity": 85},
    "tokyo": {"temp_f": 58, "condition": "Clear", "humidity": 40},
    "sydney": {"temp_f": 75, "condition": "Sunny", "humidity": 60},
}

MOCK_DATABASE: list[dict[str, Any]] = [
    {"id": 1, "name": "Alice", "role": "Engineer", "department": "Platform"},
    {"id": 2, "name": "Bob", "role": "Designer", "department": "Product"},
    {"id": 3, "name": "Carol", "role": "Manager", "department": "Engineering"},
    {"id": 4, "name": "Dan", "role": "Engineer", "department": "ML"},
    {"id": 5, "name": "Eve", "role": "SRE", "department": "Platform"},
]

MOCK_KNOWLEDGE_BASE: dict[str, str] = {
    "mission": "To build reliable AI-powered solutions that augment human capabilities.",
    "products": "AI Portfolio Suite: LLM playground, customer-support bot, deep-research agent, RAG pipeline, MCP/A2A integration.",
    "tech_stack": "Python 3.11+, FastAPI, LangChain, LlamaIndex, Qdrant, PostgreSQL, Redis, Kubernetes.",
    "team_size": "42 engineers across 6 teams.",
    "founded": "2023",
}

MOCK_USER_PREFERENCES: dict[str, Any] = {
    "theme": "dark",
    "language": "en",
    "timezone": "America/Los_Angeles",
    "notifications_enabled": True,
    "default_model": "claude-sonnet-4-20250514",
    "max_tokens": 4096,
}


# ---------------------------------------------------------------------------
# Tool result model
# ---------------------------------------------------------------------------

class ToolResult(BaseModel):
    """Structured result returned by every tool handler."""

    success: bool
    data: Any
    execution_time_ms: float
    tool_name: str


# ---------------------------------------------------------------------------
# Tool handler implementations
# ---------------------------------------------------------------------------

def _handle_calculator(arguments: dict[str, Any]) -> ToolResult:
    """Evaluate a basic math expression safely."""
    start = time.perf_counter()
    operation: str = arguments.get("operation", "")
    a: float = float(arguments.get("a", 0))
    b: float = float(arguments.get("b", 0))

    ops: dict[str, Any] = {
        "add": a + b,
        "subtract": a - b,
        "multiply": a * b,
        "divide": a / b if b != 0 else "Error: division by zero",
        "power": a ** b,
        "sqrt": math.sqrt(a) if a >= 0 else "Error: negative sqrt",
        "modulo": a % b if b != 0 else "Error: modulo by zero",
    }

    result = ops.get(operation, f"Unknown operation: {operation}")
    elapsed = (time.perf_counter() - start) * 1000
    return ToolResult(
        success=not isinstance(result, str) or not result.startswith("Error"),
        data={"operation": operation, "a": a, "b": b, "result": result},
        execution_time_ms=round(elapsed, 3),
        tool_name="calculator",
    )


def _handle_weather(arguments: dict[str, Any]) -> ToolResult:
    """Return mock weather data for a city."""
    start = time.perf_counter()
    city = arguments.get("city", "").lower().strip()
    weather = MOCK_WEATHER_DATA.get(city)
    elapsed = (time.perf_counter() - start) * 1000
    if weather is None:
        return ToolResult(
            success=False,
            data={"error": f"No weather data for '{city}'", "available_cities": list(MOCK_WEATHER_DATA)},
            execution_time_ms=round(elapsed, 3),
            tool_name="weather",
        )
    return ToolResult(
        success=True,
        data={"city": city.title(), **weather},
        execution_time_ms=round(elapsed, 3),
        tool_name="weather",
    )


def _handle_database_query(arguments: dict[str, Any]) -> ToolResult:
    """Query the mock employee database."""
    start = time.perf_counter()
    query_type = arguments.get("query_type", "list")
    filter_field = arguments.get("filter_field")
    filter_value = arguments.get("filter_value", "").lower()

    if query_type == "count":
        data: Any = {"total_records": len(MOCK_DATABASE)}
    elif query_type == "filter" and filter_field:
        matches = [
            row for row in MOCK_DATABASE
            if str(row.get(filter_field, "")).lower() == filter_value
        ]
        data = {"matches": matches, "count": len(matches)}
    else:
        data = {"records": MOCK_DATABASE, "count": len(MOCK_DATABASE)}

    elapsed = (time.perf_counter() - start) * 1000
    return ToolResult(success=True, data=data, execution_time_ms=round(elapsed, 3), tool_name="database_query")


def _handle_web_search(arguments: dict[str, Any]) -> ToolResult:
    """Mock web-search returning synthetic results."""
    start = time.perf_counter()
    query = arguments.get("query", "")
    num_results = min(int(arguments.get("num_results", 3)), 10)

    results = [
        {
            "title": f"Result {i + 1} for '{query}'",
            "url": f"https://example.com/search?q={query.replace(' ', '+')}&p={i}",
            "snippet": f"This is a mock search result #{i + 1} relevant to '{query}'.",
        }
        for i in range(num_results)
    ]
    elapsed = (time.perf_counter() - start) * 1000
    return ToolResult(
        success=True,
        data={"query": query, "results": results, "total_results": num_results},
        execution_time_ms=round(elapsed, 3),
        tool_name="web_search",
    )


def _handle_file_reader(arguments: dict[str, Any]) -> ToolResult:
    """Read contents of a file (sandboxed to allowed paths)."""
    start = time.perf_counter()
    file_path = arguments.get("path", "")
    max_lines = int(arguments.get("max_lines", 100))

    # Sandbox: only allow reading from /tmp or current working directory
    resolved = Path(file_path).resolve()
    allowed_roots = [Path("/tmp"), Path.cwd()]
    if not any(str(resolved).startswith(str(root)) for root in allowed_roots):
        elapsed = (time.perf_counter() - start) * 1000
        return ToolResult(
            success=False,
            data={"error": "Access denied — path outside sandbox", "path": str(resolved)},
            execution_time_ms=round(elapsed, 3),
            tool_name="file_reader",
        )

    try:
        lines = resolved.read_text(encoding="utf-8").splitlines()[:max_lines]
        elapsed = (time.perf_counter() - start) * 1000
        return ToolResult(
            success=True,
            data={
                "path": str(resolved),
                "lines": len(lines),
                "content": "\n".join(lines),
            },
            execution_time_ms=round(elapsed, 3),
            tool_name="file_reader",
        )
    except FileNotFoundError:
        elapsed = (time.perf_counter() - start) * 1000
        return ToolResult(
            success=False,
            data={"error": "File not found", "path": str(resolved)},
            execution_time_ms=round(elapsed, 3),
            tool_name="file_reader",
        )
    except Exception as exc:
        elapsed = (time.perf_counter() - start) * 1000
        return ToolResult(
            success=False,
            data={"error": str(exc), "path": str(resolved)},
            execution_time_ms=round(elapsed, 3),
            tool_name="file_reader",
        )


# ---------------------------------------------------------------------------
# Tool dispatcher (used by both the MCP Server and the REST adapter)
# ---------------------------------------------------------------------------

TOOL_HANDLERS: dict[str, Any] = {
    "calculator": _handle_calculator,
    "weather": _handle_weather,
    "database_query": _handle_database_query,
    "web_search": _handle_web_search,
    "file_reader": _handle_file_reader,
}


def execute_tool(tool_name: str, arguments: dict[str, Any]) -> ToolResult:
    """Execute a registered tool by name, returning a structured result."""
    handler = TOOL_HANDLERS.get(tool_name)
    if handler is None:
        return ToolResult(
            success=False,
            data={"error": f"Unknown tool: {tool_name}"},
            execution_time_ms=0.0,
            tool_name=tool_name,
        )
    return handler(arguments)


# ---------------------------------------------------------------------------
# Tool / Resource / Prompt definitions (MCP schema)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: list[Tool] = [
    Tool(
        name="calculator",
        description=(
            "Perform basic mathematical operations. "
            "Supported: add, subtract, multiply, divide, power, sqrt, modulo."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide", "power", "sqrt", "modulo"],
                    "description": "The math operation to perform.",
                },
                "a": {"type": "number", "description": "First operand."},
                "b": {"type": "number", "description": "Second operand (ignored for sqrt)."},
            },
            "required": ["operation", "a"],
        },
    ),
    Tool(
        name="weather",
        description="Look up current weather conditions for a given city.",
        inputSchema={
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name (e.g. 'San Francisco').",
                },
            },
            "required": ["city"],
        },
    ),
    Tool(
        name="database_query",
        description="Query a mock employee database. Supports listing, counting, and filtering records.",
        inputSchema={
            "type": "object",
            "properties": {
                "query_type": {
                    "type": "string",
                    "enum": ["list", "count", "filter"],
                    "description": "Type of query to execute.",
                },
                "filter_field": {
                    "type": "string",
                    "enum": ["name", "role", "department"],
                    "description": "Field to filter on (required when query_type='filter').",
                },
                "filter_value": {
                    "type": "string",
                    "description": "Value to match for the filter field.",
                },
            },
            "required": ["query_type"],
        },
    ),
    Tool(
        name="web_search",
        description="Search the web for information on a given query.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (max 10).",
                    "default": 3,
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="file_reader",
        description="Read the contents of a file. Sandboxed to /tmp and CWD.",
        inputSchema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or relative file path."},
                "max_lines": {
                    "type": "integer",
                    "description": "Maximum number of lines to read.",
                    "default": 100,
                },
            },
            "required": ["path"],
        },
    ),
]

RESOURCE_DEFINITIONS: list[Resource] = [
    Resource(
        uri="resource://company_knowledge",
        name="Company Knowledge Base",
        description="Internal company information including mission, products, and tech stack.",
        mimeType="application/json",
    ),
    Resource(
        uri="resource://user_preferences",
        name="User Preferences",
        description="Current user's settings and preferences.",
        mimeType="application/json",
    ),
]

PROMPT_DEFINITIONS: list[Prompt] = [
    Prompt(
        name="summarize",
        description="Generate a concise summary of the provided text.",
        arguments=[
            PromptArgument(name="text", description="The text to summarize.", required=True),
            PromptArgument(
                name="style",
                description="Summary style: 'brief', 'detailed', or 'bullets'.",
                required=False,
            ),
        ],
    ),
    Prompt(
        name="analyze",
        description="Perform structured analysis of data or text.",
        arguments=[
            PromptArgument(name="content", description="Content to analyze.", required=True),
            PromptArgument(
                name="focus",
                description="Analysis focus: 'sentiment', 'key_themes', 'entities', or 'general'.",
                required=False,
            ),
        ],
    ),
]


# ---------------------------------------------------------------------------
# Helper accessors (used by api.py and client.py)
# ---------------------------------------------------------------------------

def list_tools() -> list[dict[str, Any]]:
    """Return serializable tool definitions."""
    return [tool.model_dump() for tool in TOOL_DEFINITIONS]


def list_resources() -> list[dict[str, Any]]:
    """Return serializable resource definitions."""
    return [res.model_dump() for res in RESOURCE_DEFINITIONS]


def list_prompts() -> list[dict[str, Any]]:
    """Return serializable prompt definitions."""
    return [p.model_dump() for p in PROMPT_DEFINITIONS]


def read_resource(uri: str) -> dict[str, Any]:
    """Read a resource by URI."""
    if uri == "resource://company_knowledge":
        return {"uri": uri, "data": MOCK_KNOWLEDGE_BASE}
    if uri == "resource://user_preferences":
        return {"uri": uri, "data": MOCK_USER_PREFERENCES}
    return {"uri": uri, "error": "Resource not found"}


def get_prompt(name: str, arguments: dict[str, str] | None = None) -> dict[str, Any]:
    """Expand a prompt template with the given arguments."""
    arguments = arguments or {}
    if name == "summarize":
        text = arguments.get("text", "")
        style = arguments.get("style", "brief")
        style_instructions = {
            "brief": "Provide a 2-3 sentence summary.",
            "detailed": "Provide a comprehensive paragraph summary covering all key points.",
            "bullets": "Provide a bulleted list of the key points.",
        }
        return {
            "name": name,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Please summarize the following text.\n"
                        f"Style: {style_instructions.get(style, style_instructions['brief'])}\n\n"
                        f"Text:\n{text}"
                    ),
                }
            ],
        }
    if name == "analyze":
        content = arguments.get("content", "")
        focus = arguments.get("focus", "general")
        return {
            "name": name,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Analyze the following content with a focus on: {focus}.\n"
                        f"Provide structured output with sections for findings, insights, "
                        f"and recommendations.\n\nContent:\n{content}"
                    ),
                }
            ],
        }
    return {"name": name, "error": "Prompt not found"}


# ---------------------------------------------------------------------------
# MCP Server (SDK-based, stdio transport)
# ---------------------------------------------------------------------------

def create_mcp_server(name: str = "ai-portfolio-mcp") -> Server:
    """Build and return a configured :class:`mcp.server.Server` instance.

    The server registers all tools, resources, and prompts so that an MCP
    client can discover and invoke them dynamically.
    """
    server = Server(name)

    # -- Tool listing --------------------------------------------------------
    @server.list_tools()
    async def handle_list_tools() -> list[Tool]:
        logger.info("mcp.list_tools")
        return TOOL_DEFINITIONS

    # -- Tool execution ------------------------------------------------------
    @server.call_tool()
    async def handle_call_tool(name: str, arguments: dict[str, Any] | None) -> list[TextContent]:
        arguments = arguments or {}
        logger.info("mcp.call_tool", tool=name, arguments=arguments)
        result = execute_tool(name, arguments)
        return [TextContent(type="text", text=json.dumps(result.model_dump(), default=str))]

    # -- Resource listing ----------------------------------------------------
    @server.list_resources()
    async def handle_list_resources() -> list[Resource]:
        logger.info("mcp.list_resources")
        return RESOURCE_DEFINITIONS

    # -- Resource reading ----------------------------------------------------
    @server.read_resource()
    async def handle_read_resource(uri: str) -> list[TextResourceContents | EmbeddedResource]:
        logger.info("mcp.read_resource", uri=uri)
        data = read_resource(str(uri))
        return [
            TextResourceContents(
                uri=str(uri),
                text=json.dumps(data, default=str),
                mimeType="application/json",
            )
        ]

    # -- Prompt listing ------------------------------------------------------
    @server.list_prompts()
    async def handle_list_prompts() -> list[Prompt]:
        logger.info("mcp.list_prompts")
        return PROMPT_DEFINITIONS

    # -- Prompt retrieval ----------------------------------------------------
    @server.get_prompt()
    async def handle_get_prompt(name: str, arguments: dict[str, str] | None) -> GetPromptResult:
        logger.info("mcp.get_prompt", prompt=name, arguments=arguments)
        result = get_prompt(name, arguments)
        messages = [
            PromptMessage(role="user", content=TextContent(type="text", text=m["content"]))
            for m in result.get("messages", [])
        ]
        return GetPromptResult(description=f"Prompt: {name}", messages=messages)

    return server


async def run_stdio_server() -> None:
    """Launch the MCP server over stdio transport (used as a subprocess)."""
    server = create_mcp_server()
    logger.info("mcp.server.starting", transport="stdio")

    options = server.create_initialization_options()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, options)
