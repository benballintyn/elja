"""MCP wiring: configured servers -> toolsets the agent can use."""

from collections.abc import Callable
from typing import Any

from fastmcp.client.transports import StdioTransport, StreamableHttpTransport
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.toolsets import AbstractToolset

from elja.settings import EljaSettings


def toolset_name(toolset: AbstractToolset[Any]) -> str:
    """The configured server name behind a (possibly wrapped) MCP toolset."""
    inner = toolset
    while hasattr(inner, "wrapped"):
        inner = inner.wrapped
    if inner.id is None:
        raise ValueError(f"toolset {toolset!r} has no server name")
    return str(inner.id)


def build_mcp_toolsets(settings: EljaSettings) -> list[AbstractToolset[Any]]:
    """Build one toolset per configured MCP server.

    Args:
        settings: Resolved elja settings.

    Returns:
        MCP toolsets in config order; recover the server name of each with
        :func:`toolset_name` (prefixed toolsets are wrappers).
    """
    toolsets: list[AbstractToolset[Any]] = []
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
            headers = {k: v.get_secret_value() for k, v in server.headers.items()}
            transport = StreamableHttpTransport(server.url, headers=headers or None)
        toolset: AbstractToolset[Any]
        # Keep the if/else: MCPToolset distinguishes "unset" (5s default) from
        # an explicit None (wait forever) — don't collapse to a single call.
        if server.init_timeout is None:
            toolset = MCPToolset(transport, id=name, max_retries=settings.tools.max_retries)
        else:
            toolset = MCPToolset(
                transport,
                id=name,
                max_retries=settings.tools.max_retries,
                init_timeout=server.init_timeout,
            )
        if server.tool_prefix is not None:
            toolset = toolset.prefixed(server.tool_prefix)
        toolsets.append(toolset)
    return toolsets


async def preflight_mcp_toolsets(
    toolsets: list[AbstractToolset[Any]],
    on_error: Callable[[str, str], None],
) -> list[AbstractToolset[Any]]:
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
    ok: list[AbstractToolset[Any]] = []
    for toolset in toolsets:
        try:
            async with toolset:
                pass
        except Exception as exc:
            on_error(toolset_name(toolset), str(exc) or repr(exc))
        else:
            ok.append(toolset)
    return ok
