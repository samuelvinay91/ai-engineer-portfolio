"""Configuration management for MCP & A2A services."""

from __future__ import annotations

from common.config import Settings as BaseSettings


class MCPSettings(BaseSettings):
    """MCP-specific configuration."""

    # Service identity
    service_name: str = "mcp-a2a"
    service_version: str = "0.1.0"
    host: str = "0.0.0.0"
    port: int = 8008

    # MCP server settings
    mcp_server_name: str = "ai-portfolio-mcp"
    mcp_server_version: str = "0.1.0"
    mcp_transport: str = "stdio"  # "stdio" | "sse"
    mcp_sse_port: int = 8009

    # A2A settings
    a2a_base_url: str = "http://localhost:8008"
    a2a_agent_name: str = "AI Portfolio Agent"
    a2a_agent_description: str = (
        "Multi-capability agent supporting data analysis, web search, and general assistance"
    )

    # Security settings
    mcp_rate_limit_per_minute: int = 60
    mcp_max_input_length: int = 10_000
    mcp_enable_sandboxing: bool = True
    mcp_audit_log_enabled: bool = True

    # Discovery settings
    discovery_timeout_seconds: float = 5.0
    discovery_cache_ttl_seconds: int = 300
    known_agent_urls: list[str] = []

    # Redis for task storage
    task_store_ttl_seconds: int = 3600


def get_settings() -> MCPSettings:
    """Get cached settings instance."""
    return MCPSettings()
