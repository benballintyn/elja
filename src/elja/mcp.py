"""MCP wiring: configured servers -> toolsets the agent can use."""

from fastmcp.client.transports import StdioTransport, StreamableHttpTransport
from pydantic_ai.mcp import MCPToolset

from elja.settings import EljaSettings


def build_mcp_toolsets(settings: EljaSettings) -> list[MCPToolset]:
    """Build one toolset per configured MCP server.

    Args:
        settings: Resolved elja settings.

    Returns:
        MCP toolsets in config order, each carrying its server's name as id.
    """
    toolsets: list[MCPToolset] = []
    for name, server in settings.mcp.servers.items():
        transport: StdioTransport | StreamableHttpTransport
        if server.transport == "stdio":
            if server.command is None:  # pragma: no cover - enforced by MCPServerConfig
                raise ValueError(f"MCP server {name!r} has no command")
            transport = StdioTransport(
                command=server.command,
                args=server.args,
                env=server.env or None,
            )
        else:
            if server.url is None:  # pragma: no cover - enforced by MCPServerConfig
                raise ValueError(f"MCP server {name!r} has no url")
            transport = StreamableHttpTransport(server.url)
        toolsets.append(MCPToolset(transport, id=name, max_retries=settings.tools.max_retries))
    return toolsets
