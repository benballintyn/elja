"""A tiny stdio MCP server used by the tests (run as a subprocess)."""

from fastmcp import FastMCP

mcp = FastMCP("echo-server")


@mcp.tool
def echo(text: str) -> str:
    """Echo the given text back, wrapped in markers."""
    return f"echo:{text}"


if __name__ == "__main__":
    mcp.run()
