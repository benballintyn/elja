"""Tests for elja.mcp and MCP settings."""

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from elja.agent import build_agent
from elja.deps import EljaDeps
from elja.mcp import build_mcp_toolsets
from elja.settings import EljaSettings, MCPServerConfig, WorkspaceConfig

ECHO_SERVER = Path(__file__).parent / "mcp_echo_server.py"


class TestMCPSettings:
    def test_no_servers_by_default(self) -> None:
        assert EljaSettings().mcp.servers == {}

    def test_stdio_server_from_toml(self, tmp_path: Path) -> None:
        config = tmp_path / "elja.toml"
        config.write_text(
            """
[mcp.servers.mytools]
command = "python"
args = ["server.py"]
env = { API_KEY = "k" }

[mcp.servers.remote]
transport = "http"
url = "http://localhost:9000/mcp"
"""
        )
        from elja.settings import load_settings

        settings = load_settings(config)
        assert settings.mcp.servers["mytools"].transport == "stdio"
        assert settings.mcp.servers["mytools"].command == "python"
        assert settings.mcp.servers["mytools"].args == ["server.py"]
        assert settings.mcp.servers["mytools"].env == {"API_KEY": "k"}
        assert settings.mcp.servers["remote"].transport == "http"
        assert settings.mcp.servers["remote"].url == "http://localhost:9000/mcp"

    def test_stdio_requires_command(self) -> None:
        with pytest.raises(ValidationError, match="command"):
            MCPServerConfig(transport="stdio")

    def test_http_requires_url(self) -> None:
        with pytest.raises(ValidationError, match="url"):
            MCPServerConfig(transport="http")


class TestBuildMCPToolsets:
    def test_empty_settings_build_nothing(self) -> None:
        assert build_mcp_toolsets(EljaSettings()) == []

    def test_one_toolset_per_server(self) -> None:
        settings = EljaSettings(
            mcp={
                "servers": {
                    "a": {"command": "python", "args": ["x.py"]},
                    "b": {"transport": "http", "url": "http://localhost:9000/mcp"},
                }
            }  # type: ignore[arg-type]
        )
        toolsets = build_mcp_toolsets(settings)
        assert [t.id for t in toolsets] == ["a", "b"]


class TestMCPEndToEnd:
    async def test_agent_calls_mcp_tool(self, tmp_path: Path) -> None:
        """A real stdio MCP server's tool is discovered and callable by the agent."""
        settings = EljaSettings(
            workspace=WorkspaceConfig(root=tmp_path),
            mcp={"servers": {"echo": {"command": sys.executable, "args": [str(ECHO_SERVER)]}}},  # type: ignore[arg-type]
        )
        returned: list[str] = []

        def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) == 1:
                assert any(t.name == "echo" for t in info.function_tools)
                return ModelResponse(parts=[ToolCallPart(tool_name="echo", args={"text": "hi"})])
            for part in messages[-1].parts:
                content = getattr(part, "content", None)
                if content is not None:
                    returned.append(str(content))
            return ModelResponse(parts=[TextPart(content="done")])

        agent = build_agent(settings)
        deps = EljaDeps.from_settings(settings)
        with agent.override(model=FunctionModel(script)):
            async with agent:
                result = await agent.run("echo hi", deps=deps)
        assert result.output == "done"
        assert any("echo:hi" in r for r in returned)


class TestBuildAgentIncludesMCP:
    def test_agent_carries_mcp_toolsets(self) -> None:
        settings = EljaSettings(
            mcp={"servers": {"echo": {"command": "python", "args": ["x.py"]}}}  # type: ignore[arg-type]
        )
        agent = build_agent(settings)
        ids = [getattr(t, "id", None) for t in agent.toolsets]
        assert "echo" in ids
