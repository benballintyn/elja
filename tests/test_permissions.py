"""Tests for elja.permissions."""

from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pytest_mock import MockerFixture

from elja.deps import EljaDeps
from elja.permissions import build_permission_gate
from elja.settings import EljaSettings, PermissionsConfig, WorkspaceConfig
from elja.tools import build_toolset


def _shell_then_done(seen: list[str]) -> FunctionModel:
    calls: list[int] = []

    def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append(1)
        if len(calls) == 1:
            return ModelResponse(
                parts=[ToolCallPart(tool_name="run_shell", args={"command": "touch ran.txt"})]
            )
        seen.extend(str(p) for p in messages[-1].parts)
        return ModelResponse(parts=[TextPart(content="done")])

    return FunctionModel(script)


def _agent(settings: EljaSettings, model: FunctionModel) -> Agent[EljaDeps, str]:
    return Agent(
        model,
        deps_type=EljaDeps,
        toolsets=[build_toolset(settings)],
        capabilities=[build_permission_gate(settings)],
    )


class TestDescribe:
    def test_middle_elides_very_long_args_with_loud_marker(self) -> None:
        """Both ENDS of a long command stay visible — hiding the tail is an attack."""
        from elja.permissions import _describe

        call = ToolCallPart(tool_name="run_shell", args={"command": "HEAD" + "x" * 3000 + "TAIL"})
        text = _describe(call)
        assert len(text) < 2600
        assert "chars hidden" in text
        assert "HEAD" in text
        assert "TAIL" in text

    def test_unserializable_args_fall_back(self) -> None:
        from elja.permissions import _describe

        call = ToolCallPart(tool_name="t", args={"obj": object()})
        assert "t(" in _describe(call)


class TestConfig:
    def test_defaults_close_the_shell_hole(self) -> None:
        """run_shell asks by default; everything else allows."""
        cfg = EljaSettings().permissions
        assert cfg.default == "allow"
        assert cfg.tools == {"run_shell": "ask"}


class TestPolicies:
    async def test_deny_skips_and_informs_model(self, tmp_path: Path) -> None:
        settings = EljaSettings(
            workspace=WorkspaceConfig(root=tmp_path),
            permissions=PermissionsConfig(tools={"run_shell": "deny"}),
        )
        seen: list[str] = []
        result = await _agent(settings, _shell_then_done(seen)).run(
            "go", deps=EljaDeps.from_settings(settings)
        )
        assert result.output == "done"
        assert any("denied" in s for s in seen)
        assert not (tmp_path / "ran.txt").exists()

    async def test_ask_without_approver_fails_closed(self, tmp_path: Path) -> None:
        settings = EljaSettings(workspace=WorkspaceConfig(root=tmp_path))
        seen: list[str] = []
        result = await _agent(settings, _shell_then_done(seen)).run(
            "go",
            deps=EljaDeps.from_settings(settings),  # no confirm callback
        )
        assert result.output == "done"
        assert any("requires approval" in s for s in seen)
        assert not (tmp_path / "ran.txt").exists()

    async def test_ask_approved_executes(self, tmp_path: Path) -> None:
        settings = EljaSettings(workspace=WorkspaceConfig(root=tmp_path))
        prompts: list[str] = []

        def confirm(description: str) -> bool:
            prompts.append(description)
            return True

        seen: list[str] = []
        result = await _agent(settings, _shell_then_done(seen)).run(
            "go", deps=EljaDeps.from_settings(settings, confirm=confirm)
        )
        assert result.output == "done"
        assert (tmp_path / "ran.txt").exists()
        # The approval prompt names the tool and shows its arguments.
        assert len(prompts) == 1
        assert "run_shell" in prompts[0]
        assert "touch ran.txt" in prompts[0]

    async def test_ask_declined_skips_with_guidance(self, tmp_path: Path) -> None:
        settings = EljaSettings(workspace=WorkspaceConfig(root=tmp_path))
        seen: list[str] = []
        result = await _agent(settings, _shell_then_done(seen)).run(
            "go", deps=EljaDeps.from_settings(settings, confirm=lambda _: False)
        )
        assert result.output == "done"
        assert any("declined" in s for s in seen)
        assert not (tmp_path / "ran.txt").exists()

    async def test_default_policy_applies_to_unlisted_tools(self, tmp_path: Path) -> None:
        """default='deny' gates tools with no explicit entry."""
        settings = EljaSettings(
            workspace=WorkspaceConfig(root=tmp_path),
            permissions=PermissionsConfig(default="deny", tools={}),
        )
        calls: list[int] = []
        seen: list[str] = []

        def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append(1)
            if len(calls) == 1:
                return ModelResponse(
                    parts=[ToolCallPart(tool_name="list_dir", args={"path": "."})]
                )
            seen.extend(str(p) for p in messages[-1].parts)
            return ModelResponse(parts=[TextPart(content="done")])

        result = await _agent(settings, FunctionModel(script)).run(
            "go", deps=EljaDeps.from_settings(settings)
        )
        assert result.output == "done"
        assert any("denied" in s for s in seen)


class TestSubagentGating:
    async def test_child_tool_calls_prompt_the_same_approver(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        """A delegated child hitting an 'ask' tool goes through the user's approver."""
        from elja.settings import SubagentConfig
        from elja.subagents import build_subagent_toolset

        settings = EljaSettings(
            workspace=WorkspaceConfig(root=tmp_path),
            subagents={
                "worker": SubagentConfig(description="d", instructions="i", tools=["run_shell"])
            },
        )
        child_calls: list[int] = []

        def child_script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            child_calls.append(1)
            if len(child_calls) == 1:
                return ModelResponse(
                    parts=[ToolCallPart(tool_name="run_shell", args={"command": "touch kid.txt"})]
                )
            return ModelResponse(parts=[TextPart(content="child done")])

        mocker.patch("elja.subagents.build_model", return_value=FunctionModel(child_script))
        parent_calls: list[int] = []

        def parent_script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            parent_calls.append(1)
            if len(parent_calls) == 1:
                return ModelResponse(
                    parts=[ToolCallPart(tool_name="delegate_worker", args={"task": "t"})]
                )
            return ModelResponse(parts=[TextPart(content="parent done")])

        prompts: list[str] = []

        def confirm(description: str) -> bool:
            prompts.append(description)
            return True

        toolset = build_subagent_toolset(settings)
        assert toolset is not None
        parent: Agent[EljaDeps, str] = Agent(
            FunctionModel(parent_script), deps_type=EljaDeps, toolsets=[toolset]
        )
        result = await parent.run("go", deps=EljaDeps.from_settings(settings, confirm=confirm))
        assert result.output == "parent done"
        assert any("run_shell" in p for p in prompts)
        assert (tmp_path / "kid.txt").exists()


class TestReplApproval:
    async def test_interactive_yes_approves(self, tmp_path: Path, mocker: MockerFixture) -> None:
        from collections.abc import AsyncIterator

        from pydantic_ai.models.function import DeltaToolCall, DeltaToolCalls

        from elja.cli import repl
        from elja.session import Session

        settings = EljaSettings(workspace=WorkspaceConfig(root=tmp_path))
        stream_calls: list[int] = []

        async def sf(
            messages: list[ModelMessage], info: AgentInfo
        ) -> AsyncIterator[str | DeltaToolCalls]:
            stream_calls.append(1)
            if len(stream_calls) == 1:
                yield {1: DeltaToolCall(name="run_shell", json_args='{"command": "touch ok.txt"}')}
            else:
                yield "finished"

        agent: Agent[EljaDeps, str] = Agent(
            FunctionModel(stream_function=sf),
            deps_type=EljaDeps,
            toolsets=[build_toolset(settings)],
            capabilities=[build_permission_gate(settings)],
        )
        mocker.patch("elja.cli.build_agent", return_value=agent)
        prompts = iter(["run it", "y", "exit"])
        await repl(settings, "s", input_fn=lambda _: next(prompts))
        assert (tmp_path / "ok.txt").exists()
        assert len(Session.for_name(settings, "s").load()) == 4

    async def test_interactive_eof_declines(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """EOF at the approval prompt counts as a decline (fail closed)."""
        from collections.abc import AsyncIterator

        from pydantic_ai.models.function import DeltaToolCall, DeltaToolCalls

        from elja.cli import repl

        settings = EljaSettings(workspace=WorkspaceConfig(root=tmp_path))
        stream_calls: list[int] = []

        async def sf(
            messages: list[ModelMessage], info: AgentInfo
        ) -> AsyncIterator[str | DeltaToolCalls]:
            stream_calls.append(1)
            if len(stream_calls) == 1:
                yield {1: DeltaToolCall(name="run_shell", json_args='{"command": "touch no.txt"}')}
            else:
                yield "ok"

        agent: Agent[EljaDeps, str] = Agent(
            FunctionModel(stream_function=sf),
            deps_type=EljaDeps,
            toolsets=[build_toolset(settings)],
            capabilities=[build_permission_gate(settings)],
        )
        mocker.patch("elja.cli.build_agent", return_value=agent)
        answers = iter(["run it"])

        def input_fn(prompt: str) -> str:
            try:
                return next(answers)
            except StopIteration:
                raise EOFError from None

        await repl(settings, "s", input_fn=input_fn)
        assert not (tmp_path / "no.txt").exists()


class TestApprovalSerialization:
    async def test_parallel_asks_never_overlap(self, tmp_path: Path) -> None:
        """Two gated calls in one step must prompt one at a time (shared stdin)."""
        import threading
        import time

        settings = EljaSettings(workspace=WorkspaceConfig(root=tmp_path))
        active = 0
        max_active = 0
        lock = threading.Lock()

        def confirm(_: str) -> bool:
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.05)
            with lock:
                active -= 1
            return True

        calls: list[int] = []

        def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append(1)
            if len(calls) == 1:
                return ModelResponse(
                    parts=[
                        ToolCallPart(tool_name="run_shell", args={"command": "touch a.txt"}),
                        ToolCallPart(tool_name="run_shell", args={"command": "touch b.txt"}),
                    ]
                )
            return ModelResponse(parts=[TextPart(content="done")])

        result = await _agent(settings, FunctionModel(script)).run(
            "go", deps=EljaDeps.from_settings(settings, confirm=confirm)
        )
        assert result.output == "done"
        assert (tmp_path / "a.txt").exists()
        assert (tmp_path / "b.txt").exists()
        assert max_active == 1


class TestOrdering:
    def test_gate_pins_innermost(self) -> None:
        gate = build_permission_gate(EljaSettings())
        assert gate.get_ordering().position == "innermost"
