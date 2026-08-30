"""A tiny stdio MCP server used by the tests (run as a subprocess)."""

import os

from fastmcp import FastMCP

mcp = FastMCP("echo-server")


@mcp.tool
def echo(text: str) -> str:
    """Echo the given text back, wrapped in markers."""
    return f"echo:{text}"


@mcp.tool
def pid() -> int:
    """Return the server process id."""
    return os.getpid()


if __name__ == "__main__":
    mcp.run()
