"""Tests for elja.application — the embedded/host-application path.

These are contract tests for the guarantees a host application depends on:
its own dependency object arrives untouched, nothing elja-flavored is
assembled behind its back, an explicit empty choice stays empty, and two
agents cannot contaminate each other.
"""

import ast
import asyncio
import contextlib
import sys
import sysconfig
from dataclasses import dataclass
from pathlib import Path
from typing import Any, assert_type

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, RunContext, UnexpectedModelBehavior
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.concurrency import ConcurrencyLimiter
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import FunctionToolset
from pytest_mock import MockerFixture

import elja.application
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


# Long enough that a loaded machine does not flake, short enough that a test
# which stops overlapping FAILS in seconds instead of hanging until the runner
# gives up. Used where overlap is EXPECTED, so the wait is patience never paid.
_OVERLAP_TIMEOUT = 10.0

# Where overlap must NOT happen, the timeout is the expected path and is paid on
# every run — twice, because asyncio.Barrier does not break on cancellation, so
# each party waits out its own deadline. Short on purpose.
_NO_OVERLAP_TIMEOUT = 0.5


def _capture_async(
    seen: dict[str, Any],
    gate: asyncio.Barrier,
    both: asyncio.Event,
    timeout: float = _OVERLAP_TIMEOUT,
) -> FunctionModel:
    """Like _capture, but each call waits for the other run to reach the model.

    A sync FunctionModel script runs to completion with no await point, so
    asyncio.gather would serialize the two runs and the test would observe no
    concurrency at all.

    The wait is BOUNDED: if the second run never arrives, the barrier times out,
    ``both`` stays unset, and the caller's assertion fails. A test that can only
    hang proves nothing about what it was supposed to catch.
    """

    async def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        with contextlib.suppress(TimeoutError, asyncio.BrokenBarrierError):
            async with asyncio.timeout(timeout):
                await gate.wait()
            both.set()
        seen["instructions"] = info.instructions
        seen["tool_names"] = sorted(t.name for t in info.function_tools)
        seen["model_settings"] = info.model_settings
        return ModelResponse(parts=[TextPart(content="ok")])

    return FunctionModel(function=script)


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

    def test_the_application_module_imports_nothing_from_elja(self) -> None:
        """A from-import is how the machinery would creep back in.

        Patching ``elja.<module>.<name>`` cannot detect it: elja's house style
        binds imported symbols into the importing module before any test runs.
        So pin it statically instead — this module must depend on pydantic-ai
        alone.
        """
        tree = ast.parse(Path(elja.application.__file__).read_text(encoding="utf-8"))
        offenders = [
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("elja")
        ] + [
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
            if alias.name.startswith("elja")
        ]
        assert offenders == []

    async def test_nothing_touches_the_filesystem_a_process_or_a_socket(
        self, tmp_path: Path
    ) -> None:
        """The scan, subprocess and network clauses, audited rather than patched.

        Patching named primitives only catches the routes you thought of:
        ``os.listdir``, ``open``, ``os.system``, a raw socket and an
        ``__import__``-ed scan all slip past them. An audit hook sees every route
        that raises one of the events below — which is not literally everything,
        so the set is written out rather than described as total.

        The hook goes up BEFORE construction, because construction is where an
        implicit side effect would live. Module loading also raises ``open``, so
        reads from the interpreter's own library directories are filtered by
        prefix; a read anywhere else, elja's own source included, counts.
        """
        # A live trigger for the scan assertions: the autouse _hermetic fixture
        # chdirs into this same tmp_path, and load_skills resolves its directory
        # relative to the cwd. Without it an injected skills scan finds nothing
        # and the mutation survives — measured, after removing it on the theory
        # that it was dead setup.
        (tmp_path / "skills").mkdir()
        (tmp_path / "skills" / "a.md").write_text("---\nid: a\ndescription: d\n---\nbody\n")
        watched = {
            "open",
            "os.listdir",
            "os.scandir",
            "os.mkdir",
            "os.rename",
            "os.remove",
            "os.chdir",
            "os.system",
            "os.spawn",
            "os.posix_spawn",
            "os.exec",
            "os.fork",
            "os.forkpty",
            "subprocess.Popen",
            "socket.__new__",
            "socket.connect",
            "socket.getaddrinfo",
            "socket.gethostbyname",
            "glob.glob",
        }
        # The real library prefixes, not substrings: "site-packages" anywhere in a
        # path used to hide a write to /tmp/anything.pyc.
        library_prefixes = tuple(
            str(Path(path).resolve())
            for path in {
                sysconfig.get_path("purelib"),
                sysconfig.get_path("platlib"),
                sysconfig.get_path("stdlib"),
            }
            if path
        )
        touches: list[str] = []
        recording = False

        def audit(event: str, args: tuple[object, ...]) -> None:
            if not recording or event not in watched:
                return
            if event == "open" and args:
                target = str(args[0])
                if target.startswith(library_prefixes):
                    return  # a module load, not a file this code chose to read
            touches.append(event)

        sys.addaudithook(audit)
        seen: dict[str, Any] = {}
        recording = True
        try:
            agent = build_application_agent(
                _capture(seen), deps_type=HouseholdDeps, toolsets=[_household_toolset()]
            )
            await agent.run("hi", deps=HouseholdDeps(tenant="t", calls=[]))
        finally:
            recording = False
        assert touches == []


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
        assert seen["instructions"] is None

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

    async def test_an_explicitly_empty_prompt_is_honored_on_the_convenience_path(
        self, tmp_path: Path
    ) -> None:
        """The other half of the documented asymmetry: "" means no prompt."""
        from elja.agent import build_agent
        from elja.deps import EljaDeps
        from elja.settings import AgentConfig, WorkspaceConfig

        settings = EljaSettings(
            workspace=WorkspaceConfig(root=tmp_path), agent=AgentConfig(instructions="")
        )
        seen: dict[str, Any] = {}
        convenience = build_agent(settings)
        with convenience.override(model=_capture(seen)):
            await convenience.run("hi", deps=EljaDeps.from_settings(settings))
        assert seen["instructions"] is None

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
        gate = asyncio.Barrier(2)  # bleed test: both runs must reach the model
        both_in_flight = asyncio.Event()
        agent_a = build_application_agent(
            _capture_async(seen_a, gate, both_in_flight),
            deps_type=HouseholdDeps,
            instructions="agent A",
            toolsets=[_household_toolset()],
            model_settings={"temperature": 0.1},
        )
        agent_b = build_application_agent(
            _capture_async(seen_b, gate, both_in_flight),
            deps_type=HouseholdDeps,
            instructions="agent B",
            model_settings={"temperature": 0.9},
        )
        await asyncio.gather(
            agent_a.run("x", deps=deps_a),
            agent_b.run("y", deps=deps_b),
        )
        assert both_in_flight.is_set(), "the two runs never actually overlapped"
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


class TestEveryAgentComesThroughThisDoor:
    def test_the_package_constructs_an_agent_in_exactly_one_place(self) -> None:
        """The docstring's claim is universal, so assert it as one.

        A call-count spy only says "this caller used the door at least once"; it
        cannot see a second, direct ``Agent(...)`` beside it. Walk the package
        instead: the only construction site is elja/application.py.
        """
        package = Path(elja.__file__).parent
        sites: list[str] = []
        for module in sorted(package.glob("*.py")):
            tree = ast.parse(module.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "Agent"
                ):
                    sites.append(f"{module.name}:{node.lineno}")
        assert [site.split(":")[0] for site in sites] == ["application.py"], sites

    def test_build_agent_constructs_through_the_factory(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        from elja.agent import build_agent
        from elja.settings import WorkspaceConfig

        spy = mocker.patch(
            "elja.agent.build_application_agent", wraps=elja.application.build_application_agent
        )
        build_agent(EljaSettings(workspace=WorkspaceConfig(root=tmp_path)))
        assert spy.call_count == 1

    def test_a_configured_delegate_constructs_through_the_factory(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        """Otherwise the docstring's claim is false for every sub-agent."""
        from elja.settings import SubagentConfig, WorkspaceConfig
        from elja.subagents import build_subagent_toolset

        spy = mocker.patch(
            "elja.subagents.build_application_agent",
            wraps=elja.application.build_application_agent,
        )
        settings = EljaSettings(
            workspace=WorkspaceConfig(root=tmp_path),
            subagents={"helper": SubagentConfig(description="d", instructions="i")},
        )
        assert build_subagent_toolset(settings) is not None
        assert spy.call_count == 1


class TestModelPassthrough:
    def test_a_model_instance_is_passed_through_by_identity(self) -> None:
        """E2's floor: elja never rebuilds a model from its display name."""
        model = FunctionModel(lambda m, i: ModelResponse(parts=[TextPart(content="ok")]))
        agent = build_application_agent(model, deps_type=HouseholdDeps)
        assert agent.model is model

    def test_a_model_name_string_is_accepted(self) -> None:
        agent = build_application_agent("test", deps_type=HouseholdDeps)
        assert isinstance(agent.model, TestModel)


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

    async def test_the_default_cleared_placeholder_is_wrong_for_a_hosts_own_tools(
        self,
    ) -> None:
        """Pins the caveat the module docstring carries, so it cannot go stale.

        elja's placeholder invites the model to RE-RUN a cleared tool, which is
        safe for the convenience path's idempotent reads and unsafe for a host
        with side-effecting tools. It also names .elja/spill/, which this path
        never creates. The docstring tells hosts to pass their own via
        build_compaction(cleared_placeholder=...), and this test fails the moment
        the default text stops matching that advice.
        """
        from elja.compaction import CLEARED_PLACEHOLDER, build_compaction
        from elja.settings import CompactionConfig

        # A target masking alone can reach, so the placeholder is what the
        # model sees; a tighter target escalates to summarization and replaces it.
        settings = EljaSettings(
            compaction=CompactionConfig(target_tokens=3000, keep_tool_pairs=1, keep_messages=2)
        )
        views: list[list[ModelMessage]] = []

        def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if "summarization assistant" in (info.instructions or ""):
                return ModelResponse(parts=[TextPart(content="## Intent\nwork")])
            views.append(list(messages))
            return ModelResponse(parts=[TextPart(content="done")])

        agent = build_application_agent(
            FunctionModel(script),
            deps_type=HouseholdDeps,
            toolsets=[_household_toolset()],
            capabilities=build_compaction(settings),
        )
        history: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content="keep the records")])
        ]
        for i in range(8):
            history.append(
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="note_fact", args={"fact": f"f{i}"}, tool_call_id=f"c{i}"
                        )
                    ]
                )
            )
            history.append(
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name="note_fact",
                            content=f"noted{i} " * 600,
                            tool_call_id=f"c{i}",
                        )
                    ]
                )
            )
        await agent.run(
            "carry on", message_history=history, deps=HouseholdDeps(tenant="t", calls=[])
        )
        assert views, "the agent never ran"
        rendered = str(views[0])
        assert CLEARED_PLACEHOLDER in rendered
        # The two claims the docstring warns about, pinned as present.
        assert "re-run the tool" in CLEARED_PLACEHOLDER
        assert ".elja/spill/" in CLEARED_PLACEHOLDER

    async def test_the_widened_options_reach_the_agent(self) -> None:
        """A host must not have to abandon this path for a standard option."""
        agent = build_application_agent(
            "test",
            deps_type=HouseholdDeps,
            name="household-agent",
            retries=0,
            end_strategy="exhaustive",
            tool_timeout=12.5,
            max_concurrency=3,
        )
        assert agent.name == "household-agent"
        assert agent.end_strategy == "exhaustive"

    async def test_retries_zero_means_one_attempt_not_upstreams_default(self) -> None:
        """E2 wants a host able to own every paid attempt.

        Silently reverting to upstream's 1 retry buys a second paid model call
        per tool failure, which is the opposite of explicit metered attempts.
        """
        toolset: FunctionToolset[HouseholdDeps] = FunctionToolset()

        @toolset.tool
        def always_retries(ctx: RunContext[HouseholdDeps]) -> str:
            """Always asks the model to try again."""
            raise ModelRetry("try again")

        turns: list[int] = []

        def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            turns.append(1)
            return ModelResponse(parts=[ToolCallPart(tool_name="always_retries", args={})])

        agent = build_application_agent(
            FunctionModel(script),
            deps_type=HouseholdDeps,
            toolsets=[toolset],
            retries=0,
        )
        with pytest.raises(UnexpectedModelBehavior):
            await agent.run("go", deps=HouseholdDeps(tenant="t", calls=[]))
        assert len(turns) == 1

    async def test_tool_timeout_covers_a_tool_registered_on_the_agent(self) -> None:
        """It reaches the agent's OWN function toolset, which @agent.tool fills."""
        calls: list[int] = []
        outcomes: list[str] = []

        def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append(1)
            if len(calls) == 1:
                return ModelResponse(parts=[ToolCallPart(tool_name="slow", args={})])
            outcomes.append(str(messages[-1]))
            return ModelResponse(parts=[TextPart(content="handled")])

        agent = build_application_agent(
            FunctionModel(script), deps_type=HouseholdDeps, tool_timeout=0.05
        )

        @agent.tool
        async def slow(ctx: RunContext[HouseholdDeps]) -> str:
            """Takes far longer than the configured cap."""
            await asyncio.sleep(3)
            return "finished"

        result = await agent.run("go", deps=HouseholdDeps(tenant="t", calls=[]))
        assert result.output == "handled"
        assert outcomes and "Timed out" in outcomes[0]
        assert "finished" not in outcomes[0]

    async def test_tool_timeout_does_not_reach_a_toolset_you_pass_in(self) -> None:
        """The other half, so the docstring's scoping is not taken on trust.

        Upstream applies tool_timeout only to the toolset it builds itself, so a
        host whose tools live in a passed-in toolset must put the cap there. Both
        directions are pinned because getting this wrong leaves a worker pinned by
        a hung tool while the config says otherwise.
        """
        toolset: FunctionToolset[HouseholdDeps] = FunctionToolset()

        @toolset.tool
        async def slow(ctx: RunContext[HouseholdDeps]) -> str:
            """Finishes despite the agent-level cap."""
            await asyncio.sleep(0.3)
            return "finished"

        outcomes: list[str] = []
        calls: list[int] = []

        def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append(1)
            if len(calls) == 1:
                return ModelResponse(parts=[ToolCallPart(tool_name="slow", args={})])
            outcomes.append(str(messages[-1]))
            return ModelResponse(parts=[TextPart(content="handled")])

        agent = build_application_agent(
            FunctionModel(script),
            deps_type=HouseholdDeps,
            toolsets=[toolset],
            tool_timeout=0.05,
        )
        await agent.run("go", deps=HouseholdDeps(tenant="t", calls=[]))
        assert outcomes and "finished" in outcomes[0]

    async def test_a_toolset_level_timeout_is_the_one_that_enforces_there(self) -> None:
        """Which is why the docstring points a host at the toolset."""
        toolset: FunctionToolset[HouseholdDeps] = FunctionToolset(timeout=0.05)

        @toolset.tool
        async def slow(ctx: RunContext[HouseholdDeps]) -> str:
            """Takes far longer than the configured cap."""
            await asyncio.sleep(3)
            return "finished"

        outcomes: list[str] = []
        calls: list[int] = []

        def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append(1)
            if len(calls) == 1:
                return ModelResponse(parts=[ToolCallPart(tool_name="slow", args={})])
            outcomes.append(str(messages[-1]))
            return ModelResponse(parts=[TextPart(content="handled")])

        agent = build_application_agent(
            FunctionModel(script), deps_type=HouseholdDeps, toolsets=[toolset]
        )
        result = await agent.run("go", deps=HouseholdDeps(tenant="t", calls=[]))
        assert result.output == "handled"
        assert outcomes
        assert "Timed out" in outcomes[0]
        assert "finished" not in outcomes[0]

    async def test_max_concurrency_of_one_prevents_the_overlap(self) -> None:
        """The same barrier machinery, used in the direction that proves a cap."""
        gate = asyncio.Barrier(2)  # cap test: two parties that must never meet
        both_in_flight = asyncio.Event()
        deps = HouseholdDeps(tenant="t", calls=[])
        seen_a: dict[str, Any] = {}
        seen_b: dict[str, Any] = {}
        limiter = ConcurrencyLimiter(1)
        agent_a = build_application_agent(
            _capture_async(seen_a, gate, both_in_flight, _NO_OVERLAP_TIMEOUT),
            deps_type=HouseholdDeps,
            max_concurrency=limiter,
        )
        agent_b = build_application_agent(
            _capture_async(seen_b, gate, both_in_flight, _NO_OVERLAP_TIMEOUT),
            deps_type=HouseholdDeps,
            max_concurrency=limiter,
        )
        await asyncio.gather(agent_a.run("x", deps=deps), agent_b.run("y", deps=deps))
        # One at a time, so the two runs can never meet at the barrier.
        assert not both_in_flight.is_set()


def test_overloads_infer_the_output_type() -> None:
    """Checked by mypy, not at runtime: omitting output_type yields str."""
    plain = build_application_agent("test", deps_type=HouseholdDeps)
    assert_type(plain, Agent[HouseholdDeps, str])
    structured = build_application_agent("test", deps_type=HouseholdDeps, output_type=Reminder)
    assert_type(structured, Agent[HouseholdDeps, Reminder])
