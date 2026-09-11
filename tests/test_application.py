"""Tests for elja.application — the embedded/host-application path.

These are contract tests for the guarantees a host application depends on:
its own dependency object arrives untouched, nothing elja-flavored is
assembled behind its back, an explicit empty choice stays empty, and two
agents cannot contaminate each other.
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, assert_type

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import FunctionToolset
from pytest_mock import MockerFixture

from elja.application import build_application_agent
from elja.settings import EljaSettings


@dataclass
class HouseholdDeps:
    """Stand-in for a host's own typed dependencies — no elja fields at all."""

    tenant: str
    calls: list[str]


def _capture(seen: dict[str, Any]) -> FunctionModel:
    """A model that records what the agent sent it, then answers."""

    def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen["instructions"] = info.instructions
        seen["tool_names"] = sorted(t.name for t in info.function_tools)
        seen["model_settings"] = info.model_settings
        return ModelResponse(parts=[TextPart(content="ok")])

    return FunctionModel(script)


def _household_toolset() -> FunctionToolset[HouseholdDeps]:
    toolset: FunctionToolset[HouseholdDeps] = FunctionToolset()

    @toolset.tool
    def note_fact(ctx: RunContext[HouseholdDeps], fact: str) -> str:
        """Record a household fact (the host's own domain tool)."""
        ctx.deps.calls.append(f"{ctx.deps.tenant}:{fact}")
        return "noted"

    return toolset


class TestCallerOwnedDependencies:
    async def test_the_tool_receives_exactly_the_callers_deps_object(self) -> None:
        """Acceptance 1: no EljaDeps, no subclassing, no wrapper object."""
        deps = HouseholdDeps(tenant="t1", calls=[])
        received: list[object] = []

        toolset: FunctionToolset[HouseholdDeps] = FunctionToolset()

        @toolset.tool
        def probe(ctx: RunContext[HouseholdDeps], x: str) -> str:
            """Record the identity of the injected deps."""
            received.append(ctx.deps)
            return "done"

        calls: list[int] = []

        def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append(1)
            if len(calls) == 1:
                return ModelResponse(parts=[ToolCallPart(tool_name="probe", args={"x": "a"})])
            return ModelResponse(parts=[TextPart(content="finished")])

        agent = build_application_agent(
            FunctionModel(script), deps_type=HouseholdDeps, toolsets=[toolset]
        )
        result = await agent.run("go", deps=deps)
        assert result.output == "finished"
        assert received == [deps]
        assert received[0] is deps

    async def test_a_domain_tool_mutates_the_callers_object(self) -> None:
        deps = HouseholdDeps(tenant="acme", calls=[])
        calls: list[int] = []

        def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append(1)
            if len(calls) == 1:
                return ModelResponse(
                    parts=[ToolCallPart(tool_name="note_fact", args={"fact": "milk"})]
                )
            return ModelResponse(parts=[TextPart(content="done")])

        agent = build_application_agent(
            FunctionModel(script), deps_type=HouseholdDeps, toolsets=[_household_toolset()]
        )
        await agent.run("go", deps=deps)
        assert deps.calls == ["acme:milk"]


class TestNothingImplicitHappens:
    """Acceptance 2: the application path assembles only what it was given."""

    async def test_no_elja_builder_is_invoked(self, mocker: MockerFixture) -> None:
        spies = {
            name: mocker.patch(f"elja.{module}.{name}")
            for module, name in (
                ("tools", "build_toolset"),
                ("skills", "load_skills"),
                ("mcp", "build_mcp_toolsets"),
                ("subagents", "build_subagent_toolset"),
                ("permissions", "build_permission_gate"),
                ("compaction", "build_compaction"),
            )
        }
        seen: dict[str, Any] = {}
        agent = build_application_agent(
            _capture(seen), deps_type=HouseholdDeps, toolsets=[_household_toolset()]
        )
        await agent.run("hi", deps=HouseholdDeps(tenant="t", calls=[]))
        for name, spy in spies.items():
            assert spy.call_count == 0, f"{name} ran on the application path"

    async def test_only_the_callers_tools_are_offered(self) -> None:
        """No read_file/write_file/list_dir/run_shell/web_search sneaks in."""
        seen: dict[str, Any] = {}
        agent = build_application_agent(
            _capture(seen), deps_type=HouseholdDeps, toolsets=[_household_toolset()]
        )
        await agent.run("hi", deps=HouseholdDeps(tenant="t", calls=[]))
        assert seen["tool_names"] == ["note_fact"]

    async def test_running_writes_nothing_to_disk(self, tmp_path: Path) -> None:
        """No .elja directory, no spill dir, no session file — nothing."""
        before = sorted(p.name for p in tmp_path.iterdir())
        seen: dict[str, Any] = {}
        agent = build_application_agent(
            _capture(seen), deps_type=HouseholdDeps, toolsets=[_household_toolset()]
        )
        await agent.run("hi", deps=HouseholdDeps(tenant="t", calls=[]))
        assert sorted(p.name for p in tmp_path.iterdir()) == before
        assert not (tmp_path / ".elja").exists()


class TestEmptyStaysEmpty:
    """Acceptance 3: an explicit empty choice is never filled in for you."""

    async def test_no_toolsets_means_no_tools(self) -> None:
        seen: dict[str, Any] = {}
        agent = build_application_agent(_capture(seen), deps_type=HouseholdDeps)
        await agent.run("hi", deps=HouseholdDeps(tenant="t", calls=[]))
        assert seen["tool_names"] == []

    async def test_an_explicitly_empty_toolset_list_means_no_tools(self) -> None:
        seen: dict[str, Any] = {}
        agent = build_application_agent(_capture(seen), deps_type=HouseholdDeps, toolsets=[])
        await agent.run("hi", deps=HouseholdDeps(tenant="t", calls=[]))
        assert seen["tool_names"] == []

    @pytest.mark.parametrize("instructions", [None, ""])
    async def test_no_instructions_are_injected(self, instructions: str | None) -> None:
        """Unlike build_agent, None here does NOT mean elja's default prompt."""
        seen: dict[str, Any] = {}
        agent = build_application_agent(
            _capture(seen), deps_type=HouseholdDeps, instructions=instructions
        )
        await agent.run("hi", deps=HouseholdDeps(tenant="t", calls=[]))
        assert not seen["instructions"]

    async def test_the_convenience_path_still_defaults_its_prompt(self, tmp_path: Path) -> None:
        """The contrast that makes the distinction documented, not theoretical."""
        from elja.agent import DEFAULT_INSTRUCTIONS, build_agent
        from elja.deps import EljaDeps
        from elja.settings import WorkspaceConfig

        settings = EljaSettings(workspace=WorkspaceConfig(root=tmp_path))
        seen: dict[str, Any] = {}
        convenience = build_agent(settings)
        with convenience.override(model=_capture(seen)):
            await convenience.run("hi", deps=EljaDeps.from_settings(settings))
        assert seen["instructions"] == DEFAULT_INSTRUCTIONS

    async def test_callers_instructions_are_used_verbatim(self) -> None:
        seen: dict[str, Any] = {}
        agent = build_application_agent(
            _capture(seen), deps_type=HouseholdDeps, instructions="Only answer in Icelandic."
        )
        await agent.run("hi", deps=HouseholdDeps(tenant="t", calls=[]))
        assert seen["instructions"] == "Only answer in Icelandic."


class TestNoCrossAgentBleed:
    """Acceptance 4: concurrent agents with different deps/models/settings."""

    async def test_two_agents_run_concurrently_without_mixing_state(self) -> None:
        seen_a: dict[str, Any] = {}
        seen_b: dict[str, Any] = {}
        deps_a = HouseholdDeps(tenant="a", calls=[])
        deps_b = HouseholdDeps(tenant="b", calls=[])
        agent_a = build_application_agent(
            _capture(seen_a),
            deps_type=HouseholdDeps,
            instructions="agent A",
            toolsets=[_household_toolset()],
            model_settings={"temperature": 0.1},
        )
        agent_b = build_application_agent(
            _capture(seen_b),
            deps_type=HouseholdDeps,
            instructions="agent B",
            model_settings={"temperature": 0.9},
        )
        await asyncio.gather(
            agent_a.run("x", deps=deps_a),
            agent_b.run("y", deps=deps_b),
        )
        assert seen_a["instructions"] == "agent A"
        assert seen_b["instructions"] == "agent B"
        assert seen_a["tool_names"] == ["note_fact"]
        assert seen_b["tool_names"] == []
        assert (seen_a["model_settings"] or {}).get("temperature") == 0.1
        assert (seen_b["model_settings"] or {}).get("temperature") == 0.9

    async def test_a_per_run_setting_does_not_stick_to_the_agent(self) -> None:
        """Per-run model_settings override the agent default for that run only."""
        seen: dict[str, Any] = {}
        agent = build_application_agent(
            _capture(seen), deps_type=HouseholdDeps, model_settings={"temperature": 0.2}
        )
        deps = HouseholdDeps(tenant="t", calls=[])
        await agent.run("a", deps=deps, model_settings={"temperature": 0.8})
        assert (seen["model_settings"] or {}).get("temperature") == 0.8
        await agent.run("b", deps=deps)
        assert (seen["model_settings"] or {}).get("temperature") == 0.2

    async def test_mutating_the_callers_list_afterwards_does_not_change_the_agent(self) -> None:
        """The sequences are copied, so a later append is not retroactive."""
        seen: dict[str, Any] = {}
        toolsets = [_household_toolset()]
        agent = build_application_agent(_capture(seen), deps_type=HouseholdDeps, toolsets=toolsets)
        extra: FunctionToolset[HouseholdDeps] = FunctionToolset()

        @extra.tool
        def sneaky(ctx: RunContext[HouseholdDeps]) -> str:
            """Should never be offered."""
            return "no"

        toolsets.append(extra)
        await agent.run("hi", deps=HouseholdDeps(tenant="t", calls=[]))
        assert seen["tool_names"] == ["note_fact"]

    async def test_mutating_the_callers_capability_list_afterwards_is_not_retroactive(
        self,
    ) -> None:
        @dataclass(kw_only=True)
        class Late(AbstractCapability[HouseholdDeps]):
            id: str = "late"

            def get_instructions(self) -> str:
                return "late capability"

        # Positive control first: passed at build time, it IS attached — so the
        # negative assertion below is a real constraint, not a vacuous one.
        attached: dict[str, Any] = {}
        eager = build_application_agent(
            _capture(attached), deps_type=HouseholdDeps, capabilities=[Late()]
        )
        await eager.run("hi", deps=HouseholdDeps(tenant="t", calls=[]))
        assert "late capability" in str(attached["instructions"])

        capabilities: list[AbstractCapability[HouseholdDeps]] = []
        seen: dict[str, Any] = {}
        agent = build_application_agent(
            _capture(seen), deps_type=HouseholdDeps, capabilities=capabilities
        )
        capabilities.append(Late())
        await agent.run("hi", deps=HouseholdDeps(tenant="t", calls=[]))
        assert not seen["instructions"]


class TestModelPassthrough:
    def test_a_model_instance_is_passed_through_by_identity(self) -> None:
        """E2's floor: elja never rebuilds a model from its display name."""
        model = FunctionModel(lambda m, i: ModelResponse(parts=[TextPart(content="ok")]))
        agent = build_application_agent(model, deps_type=HouseholdDeps)
        assert agent.model is model

    def test_a_model_name_string_is_accepted(self) -> None:
        agent = build_application_agent("test", deps_type=HouseholdDeps)
        assert agent.model is not None


class Reminder(BaseModel):
    """A structured output a host might extract."""

    what: str
    when: str


class TestStructuredOutputAndCapabilities:
    async def test_a_structured_output_type_is_honored(self) -> None:
        def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name=info.output_tools[0].name,
                        args={"what": "vet", "when": "tuesday"},
                    )
                ]
            )

        agent = build_application_agent(
            FunctionModel(script), deps_type=HouseholdDeps, output_type=Reminder
        )
        result = await agent.run("when is the vet", deps=HouseholdDeps(tenant="t", calls=[]))
        assert result.output == Reminder(what="vet", when="tuesday")

    async def test_a_caller_capability_is_attached_with_the_callers_deps(self) -> None:
        """Capabilities are opt-in, and they see the host's own deps type."""
        observed: list[str] = []

        @dataclass(kw_only=True)
        class Watcher(AbstractCapability[HouseholdDeps]):
            id: str = "watcher"

            async def before_tool_execute(
                self,
                ctx: RunContext[HouseholdDeps],
                *,
                call: ToolCallPart,
                tool_def: ToolDefinition,
                args: Any,  # noqa: ANN401 - upstream hook signature
            ) -> Any:  # noqa: ANN401 - upstream hook signature
                observed.append(f"{ctx.deps.tenant}:{call.tool_name}")
                return args

        calls: list[int] = []

        def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append(1)
            if len(calls) == 1:
                return ModelResponse(
                    parts=[ToolCallPart(tool_name="note_fact", args={"fact": "eggs"})]
                )
            return ModelResponse(parts=[TextPart(content="done")])

        agent = build_application_agent(
            FunctionModel(script),
            deps_type=HouseholdDeps,
            toolsets=[_household_toolset()],
            capabilities=[Watcher()],
        )
        await agent.run("go", deps=HouseholdDeps(tenant="zz", calls=[]))
        assert observed == ["zz:note_fact"]

    async def test_eljas_own_compaction_attaches_to_a_foreign_deps_type(self) -> None:
        """Opt-in, not implicit: build_compaction is deps-agnostic."""
        from elja.compaction import build_compaction

        seen: dict[str, Any] = {}
        agent = build_application_agent(
            _capture(seen),
            deps_type=HouseholdDeps,
            capabilities=build_compaction(EljaSettings()),
        )
        result = await agent.run("hi", deps=HouseholdDeps(tenant="t", calls=[]))
        assert result.output == "ok"


def test_overloads_infer_the_output_type() -> None:
    """Checked by mypy, not at runtime: omitting output_type yields str."""
    plain = build_application_agent("test", deps_type=HouseholdDeps)
    assert_type(plain, Agent[HouseholdDeps, str])
    structured = build_application_agent("test", deps_type=HouseholdDeps, output_type=Reminder)
    assert_type(structured, Agent[HouseholdDeps, Reminder])
