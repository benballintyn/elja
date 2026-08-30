"""Tests for elja.agent."""

from pathlib import Path

from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from elja.agent import DEFAULT_INSTRUCTIONS, build_agent, build_usage_limits
from elja.deps import EljaDeps
from elja.settings import (
    AgentConfig,
    EljaSettings,
    LimitsConfig,
    ToolsConfig,
    WorkspaceConfig,
)


def _capture_model(seen: dict[str, object]) -> FunctionModel:
    """A model that records what the agent sends it, then answers."""

    def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen["instructions"] = info.instructions
        seen["tool_names"] = sorted(t.name for t in info.function_tools)
        return ModelResponse(parts=[TextPart(content="ok")])

    return FunctionModel(script)


def _run(settings: EljaSettings) -> dict[str, object]:
    seen: dict[str, object] = {}
    agent = build_agent(settings)
    deps = EljaDeps.from_settings(settings)
    with agent.override(model=_capture_model(seen)):
        result = agent.run_sync("hi", deps=deps)
    assert result.output == "ok"
    return seen


def test_default_instructions_and_full_toolset(tmp_path: Path) -> None:
    settings = EljaSettings(workspace=WorkspaceConfig(root=tmp_path))
    seen = _run(settings)
    assert seen["instructions"] == DEFAULT_INSTRUCTIONS
    assert seen["tool_names"] == [
        "list_dir",
        "read_file",
        "run_shell",
        "web_search",
        "write_file",
    ]


def test_instructions_override(tmp_path: Path) -> None:
    settings = EljaSettings(
        workspace=WorkspaceConfig(root=tmp_path),
        agent=AgentConfig(instructions="Only speak French."),
    )
    seen = _run(settings)
    assert seen["instructions"] == "Only speak French."


def test_tool_toggles_flow_through(tmp_path: Path) -> None:
    settings = EljaSettings(
        workspace=WorkspaceConfig(root=tmp_path),
        tools=ToolsConfig(run_shell=False, write_file=False),
    )
    seen = _run(settings)
    assert seen["tool_names"] == ["list_dir", "read_file", "web_search"]


def test_build_agent_uses_configured_model(tmp_path: Path) -> None:
    settings = EljaSettings(workspace=WorkspaceConfig(root=tmp_path))
    agent = build_agent(settings)
    assert agent.model is not None
    assert getattr(agent.model, "model_name", None) == "qwen/qwen3.8-27b"


def test_build_usage_limits() -> None:
    settings = EljaSettings(limits=LimitsConfig(request_limit=7, total_tokens_limit=1000))
    limits = build_usage_limits(settings)
    assert limits.request_limit == 7
    assert limits.total_tokens_limit == 1000


def test_build_usage_limits_defaults() -> None:
    limits = build_usage_limits(EljaSettings())
    assert limits.request_limit == 25
    assert limits.total_tokens_limit is None
