"""Tests for elja.mcp and MCP settings."""

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pytest_mock import MockerFixture

from elja.agent import build_agent
from elja.deps import EljaDeps
from elja.mcp import build_mcp_toolsets, preflight_mcp_toolsets
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


class TestLifecycle:
    async def test_stdio_server_persists_across_runs(self, tmp_path: Path) -> None:
        """The CLI never enters the agent context; keep-alive must reuse the subprocess."""
        settings = EljaSettings(
            workspace=WorkspaceConfig(root=tmp_path),
            mcp={"servers": {"echo": {"command": sys.executable, "args": [str(ECHO_SERVER)]}}},  # type: ignore[arg-type]
        )
        pids: list[str] = []

        def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) == 1:
                return ModelResponse(parts=[ToolCallPart(tool_name="pid", args={})])
            pids.append(str(getattr(messages[-1].parts[0], "content", "")))
            return ModelResponse(parts=[TextPart(content="ok")])

        agent = build_agent(settings)
        deps = EljaDeps.from_settings(settings)
        with agent.override(model=FunctionModel(script)):
            for _ in range(2):  # no `async with agent` on purpose — mirrors run_turn
                result = await agent.run("pid?", deps=deps)
                assert result.output == "ok"
        assert len(pids) == 2
        assert pids[0] == pids[1]


class TestPreflight:
    async def test_drops_bad_servers_and_names_them(self, tmp_path: Path) -> None:
        settings = EljaSettings(
            mcp={
                "servers": {
                    "good": {"command": sys.executable, "args": [str(ECHO_SERVER)]},
                    "bad": {"command": "definitely-missing-binary-xyz", "args": []},
                }
            }  # type: ignore[arg-type]
        )
        errors: list[tuple[str, str]] = []
        ok = await preflight_mcp_toolsets(
            build_mcp_toolsets(settings), lambda n, e: errors.append((n, e))
        )
        assert [t.id for t in ok] == ["good"]
        assert len(errors) == 1
        assert errors[0][0] == "bad"
        assert errors[0][1]  # message is non-empty

    async def test_repl_warns_and_survives_bad_server(
        self, tmp_path: Path, mocker: object, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from collections.abc import AsyncIterator

        from pytest_mock import MockerFixture

        from elja.cli import repl

        assert isinstance(mocker, MockerFixture)
        settings = EljaSettings(
            workspace=WorkspaceConfig(root=tmp_path),
            mcp={"servers": {"bad": {"command": "definitely-missing-binary-xyz", "args": []}}},  # type: ignore[arg-type]
        )

        async def sf(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
            yield "fine"

        from pydantic_ai import Agent

        mocker.patch(
            "elja.cli.build_agent",
            return_value=Agent(FunctionModel(stream_function=sf), deps_type=EljaDeps),
        )
        await repl(settings, "s", once="hello")
        out = capsys.readouterr().out
        assert "warning: MCP server 'bad'" in out
        assert "fine" in out


class TestServerOptions:
    def test_headers_reach_http_transport_and_stay_secret(self) -> None:
        """Auth headers are passed to the transport but never leak from settings repr."""
        settings = EljaSettings(
            mcp={
                "servers": {
                    "r": {
                        "transport": "http",
                        "url": "http://localhost:9000/mcp",
                        "headers": {"Authorization": "Bearer sekrit-token"},
                    }
                }
            }  # type: ignore[arg-type]
        )
        assert "sekrit-token" not in repr(settings)
        (toolset,) = build_mcp_toolsets(settings)
        transport = toolset.client.transport  # type: ignore[attr-defined]
        assert transport.headers == {"Authorization": "Bearer sekrit-token"}

    def test_headers_rejected_for_stdio(self) -> None:
        with pytest.raises(ValidationError, match="headers"):
            MCPServerConfig(command="python", headers={"X": "y"})  # type: ignore[dict-item]

    def test_bad_prefix_rejected(self) -> None:
        with pytest.raises(ValidationError, match="tool_prefix"):
            MCPServerConfig(command="python", tool_prefix="has space")

    def test_init_timeout_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            MCPServerConfig(command="python", init_timeout=0)

    def test_prefixed_toolset_keeps_recoverable_name(self) -> None:
        from elja.mcp import toolset_name

        settings = EljaSettings(
            mcp={
                "servers": {
                    "helper": {"command": "python", "args": ["x.py"], "tool_prefix": "helper"}
                }
            }  # type: ignore[arg-type]
        )
        from pydantic_ai.toolsets import PrefixedToolset

        (toolset,) = build_mcp_toolsets(settings)
        assert isinstance(toolset, PrefixedToolset)
        assert toolset_name(toolset) == "helper"

    def test_init_timeout_forwarded(self, mocker: MockerFixture) -> None:
        from pydantic_ai.mcp import MCPToolset as RealToolset

        spy = mocker.patch("elja.mcp.MCPToolset", wraps=RealToolset)
        settings = EljaSettings(
            mcp={"servers": {"slow": {"command": "python", "args": ["x.py"], "init_timeout": 30}}}  # type: ignore[arg-type]
        )
        build_mcp_toolsets(settings)
        assert spy.call_args.kwargs["init_timeout"] == 30

    def test_init_timeout_forwarded_with_prefix(self, mocker: MockerFixture) -> None:
        from pydantic_ai.mcp import MCPToolset as RealToolset

        spy = mocker.patch("elja.mcp.MCPToolset", wraps=RealToolset)
        settings = EljaSettings(
            mcp={
                "servers": {
                    "slow": {
                        "command": "python",
                        "args": ["x.py"],
                        "init_timeout": 30,
                        "tool_prefix": "slow",
                    }
                }
            }  # type: ignore[arg-type]
        )
        build_mcp_toolsets(settings)
        assert spy.call_args.kwargs["init_timeout"] == 30

    def test_duplicate_prefixes_rejected(self) -> None:
        with pytest.raises(ValidationError, match="duplicate tool_prefix"):
            EljaSettings(
                mcp={
                    "servers": {
                        "a": {"command": "python", "args": [], "tool_prefix": "p"},
                        "b": {"command": "python", "args": [], "tool_prefix": "p"},
                    }
                }  # type: ignore[arg-type]
            )

    def test_toolset_name_raises_without_id(self) -> None:
        from pydantic_ai.toolsets import FunctionToolset

        from elja.mcp import toolset_name

        with pytest.raises(ValueError, match="no server name"):
            toolset_name(FunctionToolset())

    async def test_prefixed_preflight_failure_named_by_server(self) -> None:
        from elja.mcp import preflight_mcp_toolsets

        settings = EljaSettings(
            mcp={
                "servers": {
                    "bad": {
                        "command": "definitely-missing-binary-xyz",
                        "args": [],
                        "tool_prefix": "bad",
                    }
                }
            }  # type: ignore[arg-type]
        )
        errors: list[tuple[str, str]] = []
        ok = await preflight_mcp_toolsets(
            build_mcp_toolsets(settings), lambda n, e: errors.append((n, e))
        )
        assert ok == []
        assert errors[0][0] == "bad"

    async def test_prefixed_tools_callable_end_to_end(self, tmp_path: Path) -> None:
        """A prefixed server's tool is visible and callable as <prefix>_<name>."""
        settings = EljaSettings(
            workspace=WorkspaceConfig(root=tmp_path),
            mcp={
                "servers": {
                    "echo": {
                        "command": sys.executable,
                        "args": [str(ECHO_SERVER)],
                        "tool_prefix": "helper",
                    }
                }
            },  # type: ignore[arg-type]
        )
        returned: list[str] = []

        def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) == 1:
                names = [t.name for t in info.function_tools]
                assert "helper_echo" in names
                assert "echo" not in names
                return ModelResponse(
                    parts=[ToolCallPart(tool_name="helper_echo", args={"text": "hi"})]
                )
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


class TestPrefixedPermissions:
    async def test_gate_matches_prefixed_name(self, tmp_path: Path) -> None:
        """Permission entries must use the prefixed name — pin the hook ordering."""
        from elja.deps import EljaDeps as Deps
        from elja.settings import PermissionsConfig

        settings = EljaSettings(
            workspace=WorkspaceConfig(root=tmp_path),
            permissions=PermissionsConfig(tools={"helper_echo": "deny"}),
            mcp={
                "servers": {
                    "echo": {
                        "command": sys.executable,
                        "args": [str(ECHO_SERVER)],
                        "tool_prefix": "helper",
                    }
                }
            },  # type: ignore[arg-type]
        )
        seen: list[str] = []

        def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) == 1:
                return ModelResponse(
                    parts=[ToolCallPart(tool_name="helper_echo", args={"text": "hi"})]
                )
            seen.extend(str(p) for p in messages[-1].parts)
            return ModelResponse(parts=[TextPart(content="done")])

        agent = build_agent(settings)
        with agent.override(model=FunctionModel(script)):
            async with agent:
                result = await agent.run("go", deps=Deps.from_settings(settings))
        assert result.output == "done"
        assert any("denied" in s for s in seen)
        assert not any("echo:hi" in s for s in seen)
