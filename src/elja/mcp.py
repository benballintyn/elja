"""MCP wiring: configured servers -> toolsets the agent can use."""

from collections.abc import Callable

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


async def preflight_mcp_toolsets(
    toolsets: list[MCPToolset],
    on_error: Callable[[str, str], None],
) -> list[MCPToolset]:
    """Connect to each MCP server once, dropping the ones that fail.

    Without this, a single broken ``[mcp.servers.*]`` entry would make every
    agent turn fail (toolsets are entered per run), and connect errors
    wouldn't name the offending server.

    Args:
        toolsets: Toolsets from :func:`build_mcp_toolsets`.
        on_error: Called with (server name, error message) per failed server.

    Returns:
        The toolsets that connected successfully (their stdio subprocesses
        stay warm for the agent thanks to fastmcp's keep-alive).
    """
    ok: list[MCPToolset] = []
    for toolset in toolsets:
        try:
            async with toolset:
                pass
        except Exception as exc:
            on_error(str(toolset.id), str(exc) or repr(exc))
        else:
            ok.append(toolset)
    return ok
