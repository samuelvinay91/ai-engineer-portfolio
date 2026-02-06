"""MCP Client implementation for connecting to MCP servers.

Provides tool discovery, invocation, resource access, and prompt retrieval
through a session-managed connection to any MCP-compliant server.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import structlog
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import (
    CallToolResult,
    GetPromptResult,
    TextContent,
)

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data classes for typed results
# ---------------------------------------------------------------------------


@dataclass
class DiscoveredTool:
    """A tool discovered from an MCP server."""

    name: str
    description: str | None
    input_schema: dict[str, Any]


@dataclass
class DiscoveredResource:
    """A resource discovered from an MCP server."""

    uri: str
    name: str | None
    description: str | None
    mime_type: str | None


@dataclass
class DiscoveredPrompt:
    """A prompt template discovered from an MCP server."""

    name: str
    description: str | None
    arguments: list[dict[str, Any]]


@dataclass
class ToolCallResult:
    """Result of invoking a tool through MCP."""

    success: bool
    content: list[dict[str, Any]]
    is_error: bool = False
    raw: CallToolResult | None = None


@dataclass
class MCPServerConnection:
    """Holds connection state for a single MCP server."""

    server_command: str
    server_args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None
    session: ClientSession | None = None
    tools: list[DiscoveredTool] = field(default_factory=list)
    resources: list[DiscoveredResource] = field(default_factory=list)
    prompts: list[DiscoveredPrompt] = field(default_factory=list)


class MCPClient:
    """High-level MCP client that manages connections to one or more MCP servers.

    Usage::

        async with MCPClient.connect("python", ["-m", "my_mcp_server"]) as client:
            tools = await client.discover_tools()
            result = await client.call_tool("calculator", {"operation": "add", "a": 1, "b": 2})
    """

    def __init__(self) -> None:
        self._connections: dict[str, MCPServerConnection] = {}
        self._active_session: ClientSession | None = None

    # -- Context-manager based connection ------------------------------------

    @classmethod
    @asynccontextmanager
    async def connect(
        cls,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        server_name: str = "default",
    ) -> AsyncIterator[MCPClient]:
        """Connect to an MCP server process and yield a ready-to-use client.

        Parameters
        ----------
        command:
            The executable to run (e.g. ``"python"``).
        args:
            Arguments for the server process.
        env:
            Optional environment variables for the subprocess.
        server_name:
            A label for this connection (useful when connecting to multiple
            servers simultaneously).
        """
        client = cls()
        server_params = StdioServerParameters(
            command=command,
            args=args or [],
            env=env,
        )

        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                conn = MCPServerConnection(
                    server_command=command,
                    server_args=args or [],
                    env=env,
                    session=session,
                )
                client._connections[server_name] = conn
                client._active_session = session

                # Auto-discover capabilities on connect
                await client.discover_tools(server_name=server_name)
                await client.discover_resources(server_name=server_name)
                await client.discover_prompts(server_name=server_name)

                logger.info(
                    "mcp.client.connected",
                    server=server_name,
                    tools=len(conn.tools),
                    resources=len(conn.resources),
                    prompts=len(conn.prompts),
                )

                yield client

    # -- Session helper ------------------------------------------------------

    def _get_session(self, server_name: str = "default") -> ClientSession:
        conn = self._connections.get(server_name)
        if conn is None or conn.session is None:
            raise RuntimeError(f"No active session for server '{server_name}'")
        return conn.session

    # -- Discovery -----------------------------------------------------------

    async def discover_tools(self, server_name: str = "default") -> list[DiscoveredTool]:
        """List all tools exposed by the connected MCP server."""
        session = self._get_session(server_name)
        result = await session.list_tools()
        conn = self._connections[server_name]
        conn.tools = [
            DiscoveredTool(
                name=t.name,
                description=t.description,
                input_schema=t.inputSchema,
            )
            for t in result.tools
        ]
        return conn.tools

    async def discover_resources(self, server_name: str = "default") -> list[DiscoveredResource]:
        """List all resources exposed by the connected MCP server."""
        session = self._get_session(server_name)
        result = await session.list_resources()
        conn = self._connections[server_name]
        conn.resources = [
            DiscoveredResource(
                uri=str(r.uri),
                name=r.name,
                description=r.description,
                mime_type=r.mimeType,
            )
            for r in result.resources
        ]
        return conn.resources

    async def discover_prompts(self, server_name: str = "default") -> list[DiscoveredPrompt]:
        """List all prompt templates exposed by the connected MCP server."""
        session = self._get_session(server_name)
        result = await session.list_prompts()
        conn = self._connections[server_name]
        conn.prompts = [
            DiscoveredPrompt(
                name=p.name,
                description=p.description,
                arguments=[
                    {"name": a.name, "description": a.description, "required": a.required}
                    for a in (p.arguments or [])
                ],
            )
            for p in result.prompts
        ]
        return conn.prompts

    # -- Tool invocation -----------------------------------------------------

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        server_name: str = "default",
    ) -> ToolCallResult:
        """Invoke a tool on the MCP server and return the parsed result.

        Parameters
        ----------
        tool_name:
            Name of the tool to call (must exist on the server).
        arguments:
            Tool arguments matching the declared input schema.
        server_name:
            Which server connection to use.
        """
        session = self._get_session(server_name)
        logger.info("mcp.client.call_tool", tool=tool_name, arguments=arguments)

        try:
            result: CallToolResult = await session.call_tool(tool_name, arguments or {})
            content_items: list[dict[str, Any]] = []
            for item in result.content:
                if isinstance(item, TextContent):
                    try:
                        content_items.append(json.loads(item.text))
                    except json.JSONDecodeError:
                        content_items.append({"type": "text", "text": item.text})
                else:
                    content_items.append({"type": item.type, "data": str(item)})

            return ToolCallResult(
                success=not result.isError,
                content=content_items,
                is_error=bool(result.isError),
                raw=result,
            )
        except Exception as exc:
            logger.error("mcp.client.call_tool.error", tool=tool_name, error=str(exc))
            return ToolCallResult(
                success=False,
                content=[{"error": str(exc)}],
                is_error=True,
            )

    # -- Resource access -----------------------------------------------------

    async def read_resource(
        self, uri: str, server_name: str = "default"
    ) -> dict[str, Any]:
        """Read a resource from the MCP server by URI."""
        session = self._get_session(server_name)
        logger.info("mcp.client.read_resource", uri=uri)

        try:
            result = await session.read_resource(uri)
            contents: list[dict[str, Any]] = []
            for item in result.contents:
                try:
                    contents.append(json.loads(item.text if hasattr(item, "text") else str(item)))
                except (json.JSONDecodeError, AttributeError):
                    contents.append({"raw": str(item)})
            return {"uri": uri, "contents": contents}
        except Exception as exc:
            logger.error("mcp.client.read_resource.error", uri=uri, error=str(exc))
            return {"uri": uri, "error": str(exc)}

    # -- Prompt access -------------------------------------------------------

    async def get_prompt(
        self,
        name: str,
        arguments: dict[str, str] | None = None,
        server_name: str = "default",
    ) -> dict[str, Any]:
        """Retrieve an expanded prompt template from the MCP server."""
        session = self._get_session(server_name)
        logger.info("mcp.client.get_prompt", prompt=name, arguments=arguments)

        try:
            result: GetPromptResult = await session.get_prompt(name, arguments or {})
            messages = []
            for msg in result.messages:
                content_text = ""
                if isinstance(msg.content, TextContent):
                    content_text = msg.content.text
                elif isinstance(msg.content, str):
                    content_text = msg.content
                else:
                    content_text = str(msg.content)
                messages.append({"role": msg.role, "content": content_text})
            return {"name": name, "description": result.description, "messages": messages}
        except Exception as exc:
            logger.error("mcp.client.get_prompt.error", prompt=name, error=str(exc))
            return {"name": name, "error": str(exc)}

    # -- Introspection -------------------------------------------------------

    def get_cached_tools(self, server_name: str = "default") -> list[DiscoveredTool]:
        """Return locally-cached tools (no server roundtrip)."""
        conn = self._connections.get(server_name)
        return conn.tools if conn else []

    def get_cached_resources(self, server_name: str = "default") -> list[DiscoveredResource]:
        """Return locally-cached resources (no server roundtrip)."""
        conn = self._connections.get(server_name)
        return conn.resources if conn else []

    def get_cached_prompts(self, server_name: str = "default") -> list[DiscoveredPrompt]:
        """Return locally-cached prompts (no server roundtrip)."""
        conn = self._connections.get(server_name)
        return conn.prompts if conn else []
