"""Tests for elja.subagents."""

from pathlib import Path

import pytest
from pydantic import ValidationError
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pytest_mock import MockerFixture

from elja.deps import EljaDeps
from elja.settings import EljaSettings, SubagentConfig, WorkspaceConfig
from elja.subagents import build_subagent_toolset
from elja.tools import build_toolset


@pytest.fixture
def settings(tmp_path: Path) -> EljaSettings:
    return EljaSettings(
        workspace=WorkspaceConfig(root=tmp_path),
        subagents={
            "researcher": SubagentConfig(
                description="Researches a question and reports key facts.",
                instructions="Answer tersely.",
                tools=["read_file", "list_dir"],
            )
        },
    )


class TestConfig:
    def test_none_by_default(self) -> None:
        assert EljaSettings().subagents == {}

    def test_toml_roundtrip(self, tmp_path: Path) -> None:
        config = tmp_path / "elja.toml"
        config.write_text(
            """
[subagents.researcher]
description = "Looks things up."
instructions = "Be terse."
tools = ["read_file"]
"""
        )
        from elja.settings import load_settings

        settings = load_settings(config)
        assert settings.subagents["researcher"].tools == ["read_file"]

    def test_unknown_tool_rejected(self) -> None:
        with pytest.raises(ValidationError, match="unknown tool"):
            SubagentConfig(description="d", instructions="i", tools=["frobnicate"])


class TestBuildToolset:
    def test_no_subagents_builds_empty(self, tmp_path: Path) -> None:
        settings = EljaSettings(workspace=WorkspaceConfig(root=tmp_path))
        assert build_subagent_toolset(settings) is None

    def test_delegate_tool_per_subagent(self, settings: EljaSettings) -> None:
        toolset = build_subagent_toolset(settings)
        assert toolset is not None
        assert set(toolset.tools) == {"delegate_researcher"}

    def test_reserved_name_rejected(self, tmp_path: Path) -> None:
        settings = EljaSettings(
            workspace=WorkspaceConfig(root=tmp_path),
            subagents={"read_file": SubagentConfig(description="d", instructions="i")},
        )
        with pytest.raises(ValueError, match="read_file"):
            build_subagent_toolset(settings)

    def test_bad_name_rejected(self, tmp_path: Path) -> None:
        settings = EljaSettings(
            workspace=WorkspaceConfig(root=tmp_path),
            subagents={"bad name!": SubagentConfig(description="d", instructions="i")},
        )
        with pytest.raises(ValueError, match="bad name!"):
            build_subagent_toolset(settings)


class TestDelegation:
    async def test_delegation_isolated_context_and_result_only(
        self, settings: EljaSettings, mocker: MockerFixture
    ) -> None:
        """The child sees only the task (no parent history) and returns only its answer."""
        child_saw: list[list[ModelMessage]] = []

        def child_script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            child_saw.append(list(messages))
            assert sorted(t.name for t in info.function_tools) == ["list_dir", "read_file"]
            return ModelResponse(parts=[TextPart(content="child-answer-42")])

        mocker.patch("elja.subagents.build_model", return_value=FunctionModel(child_script))

        calls: list[int] = []

        def parent_script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append(1)
            if len(calls) == 1:
                assert any(t.name == "delegate_researcher" for t in info.function_tools)
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="delegate_researcher", args={"task": "find the answer"}
                        )
                    ]
                )
            last_parts = messages[-1].parts
            assert any("child-answer-42" in str(getattr(p, "content", "")) for p in last_parts)
            return ModelResponse(parts=[TextPart(content="parent-done")])

        toolset = build_subagent_toolset(settings)
        assert toolset is not None
        parent: Agent[EljaDeps, str] = Agent(
            FunctionModel(parent_script),
            deps_type=EljaDeps,
            toolsets=[build_toolset(settings), toolset],
        )
        result = await parent.run("please delegate", deps=EljaDeps.from_settings(settings))
        assert result.output == "parent-done"
        # Child context is isolated: exactly one request, containing only the task.
        assert len(child_saw) == 1
        child_request = child_saw[0][0]
        user_parts = [p for p in child_request.parts if isinstance(p, UserPromptPart)]
        assert len(user_parts) == 1
        assert "find the answer" in str(user_parts[0].content)
        assert "please delegate" not in str(user_parts[0].content)

    async def test_child_usage_rolls_up(
        self, settings: EljaSettings, mocker: MockerFixture
    ) -> None:
        def child_script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content="ok")])

        mocker.patch("elja.subagents.build_model", return_value=FunctionModel(child_script))
        calls: list[int] = []

        def parent_script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append(1)
            if len(calls) == 1:
                return ModelResponse(
                    parts=[ToolCallPart(tool_name="delegate_researcher", args={"task": "t"})]
                )
            return ModelResponse(parts=[TextPart(content="done")])

        toolset = build_subagent_toolset(settings)
        assert toolset is not None
        parent: Agent[EljaDeps, str] = Agent(
            FunctionModel(parent_script), deps_type=EljaDeps, toolsets=[toolset]
        )
        result = await parent.run("go", deps=EljaDeps.from_settings(settings))
        # 2 parent requests + 1 child request.
        assert result.usage.requests == 3

    async def test_child_failure_surfaces_as_retry(
        self, settings: EljaSettings, mocker: MockerFixture
    ) -> None:
        def child_script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise RuntimeError("child exploded")

        mocker.patch("elja.subagents.build_model", return_value=FunctionModel(child_script))
        calls: list[int] = []

        def parent_script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append(1)
            if len(calls) == 1:
                return ModelResponse(
                    parts=[ToolCallPart(tool_name="delegate_researcher", args={"task": "t"})]
                )
            last = " ".join(str(p) for p in messages[-1].parts)
            assert "subagent 'researcher' failed:" in last
            assert "child exploded" in last
            return ModelResponse(parts=[TextPart(content="carried on")])

        toolset = build_subagent_toolset(settings)
        assert toolset is not None
        parent: Agent[EljaDeps, str] = Agent(
            FunctionModel(parent_script), deps_type=EljaDeps, toolsets=[toolset]
        )
        result = await parent.run("go", deps=EljaDeps.from_settings(settings))
        assert result.output == "carried on"


class TestToolGating:
    def test_child_default_subset_respects_disabled_tools(self, tmp_path: Path) -> None:
        from elja.settings import ToolsConfig
        from elja.subagents import _child_toolset

        settings = EljaSettings(
            workspace=WorkspaceConfig(root=tmp_path),
            tools=ToolsConfig(run_shell=False, write_file=False),
        )
        cfg = SubagentConfig(description="d", instructions="i")
        toolset = _child_toolset(settings, "x", cfg)
        assert set(toolset.tools) == {"read_file", "list_dir", "web_search"}

    def test_explicit_grant_of_disabled_tool_rejected(self, tmp_path: Path) -> None:
        from elja.settings import ToolsConfig
        from elja.subagents import _child_toolset

        settings = EljaSettings(
            workspace=WorkspaceConfig(root=tmp_path),
            tools=ToolsConfig(run_shell=False),
        )
        cfg = SubagentConfig(description="d", instructions="i", tools=["run_shell"])
        with pytest.raises(ValueError, match="disabled in"):
            _child_toolset(settings, "x", cfg)


class TestBudget:
    async def test_request_limit_is_per_delegation(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        """cfg.request_limit budgets THIS delegation, offset by parent spend."""
        settings = EljaSettings(
            workspace=WorkspaceConfig(root=tmp_path),
            subagents={
                "helper": SubagentConfig(
                    description="d", instructions="i", tools=["list_dir"], request_limit=1
                )
            },
        )
        child_requests: list[int] = []

        def child_script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            child_requests.append(1)
            # Child tries a second request via a tool call: exceeds its budget of 1.
            return ModelResponse(parts=[ToolCallPart(tool_name="list_dir", args={"path": "."})])

        mocker.patch("elja.subagents.build_model", return_value=FunctionModel(child_script))
        calls: list[int] = []
        seen_result: list[str] = []

        def parent_script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append(1)
            if len(calls) == 1:
                return ModelResponse(
                    parts=[ToolCallPart(tool_name="delegate_helper", args={"task": "t"})]
                )
            seen_result.append(str(messages[-1]))
            return ModelResponse(parts=[TextPart(content="handled")])

        toolset = build_subagent_toolset(settings)
        assert toolset is not None
        parent: Agent[EljaDeps, str] = Agent(
            FunctionModel(parent_script), deps_type=EljaDeps, toolsets=[toolset]
        )
        result = await parent.run("go", deps=EljaDeps.from_settings(settings))
        # Budget exhaustion is terminal info, not a retry loop: run survives.
        assert result.output == "handled"
        assert any("budget exhausted" in s for s in seen_result)
        assert any("Do not delegate this again" in s for s in seen_result)


class TestAgentWiring:
    def test_build_agent_includes_delegates(self, settings: EljaSettings) -> None:
        from elja.agent import build_agent

        seen: list[str] = []

        def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen.extend(t.name for t in info.function_tools)
            return ModelResponse(parts=[TextPart(content="ok")])

        agent = build_agent(settings)
        with agent.override(model=FunctionModel(script)):
            agent.run_sync("hi", deps=EljaDeps.from_settings(settings))
        assert "delegate_researcher" in seen
