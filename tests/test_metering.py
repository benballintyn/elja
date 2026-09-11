"""Contract tests for a host application that meters and gates paid requests.

A host supplies a ``Model`` it has already wrapped, so that admission against
its own budget runs *before* any network dispatch. These tests pin what elja
guarantees about that object:

1. Admission runs before every request elja causes, including the summarizer's
   own private request during compaction.
2. A separately supplied guarded summarizer stays separate, which is how a host
   tells a main request apart from a compaction request.
3. A denial is terminal: no retry, no fallback, no conversion into a tool-retry
   suggestion, in plain and streamed runs alike.
4. The summary attempt is accounted once.

What elja does NOT provide is a spend ledger. See ``docs/EMBEDDING.md``.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.toolsets import FunctionToolset

from elja.compaction import build_compaction
from elja.deps import EljaDeps
from elja.settings import CompactionConfig, EljaSettings, WorkspaceConfig


class BudgetDeniedError(Exception):
    """What a host's admission check raises when the budget is spent."""


class GuardedModel(WrapperModel):
    """A host's metering wrapper: admit, then dispatch.

    Attribution lives on the instance, not in ``RunContext``: a ``Model`` never
    receives a run context, and the summarizer's private agent runs with
    ``deps=None``, so per-instance state is the mechanism that works for both.
    """

    def __init__(
        self, wrapped: Model, tag: str, admissions: list[str], *, deny: bool = False
    ) -> None:
        """Wrap a model with a host-style admission gate."""
        super().__init__(wrapped)
        self.tag = tag
        self.admissions = admissions
        self.deny = deny
        self.dispatches: list[str] = []
        self.stream_had_run_context: list[bool] = []

    def _admit(self) -> None:
        self.admissions.append(self.tag)
        if self.deny:
            raise BudgetDeniedError(f"{self.tag}: budget exhausted")

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Admit before dispatching, so a denial costs no provider call."""
        self._admit()
        self.dispatches.append(self.tag)
        return await super().request(messages, model_settings, model_request_parameters)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        """The streamed path needs its own gate; `request` does not cover it."""
        self._admit()
        self.dispatches.append(self.tag)
        # Unlike `request`, this hook DOES receive the run context — recorded so
        # the asymmetry is pinned rather than assumed (see docs/EMBEDDING.md).
        self.stream_had_run_context.append(run_context is not None)
        async with super().request_stream(
            messages, model_settings, model_request_parameters, run_context
        ) as stream:
            yield stream


def _bulky_history(pairs: int, result_size: int = 600) -> list[ModelMessage]:
    """A transcript too large for masking alone, so the summarizer must fire."""
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="original task: audit the files")])
    ]
    for i in range(pairs):
        messages.append(
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="read_file", args={"path": f"f{i}.txt"}, tool_call_id=f"c{i}"
                    )
                ]
            )
        )
        messages.append(
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name="read_file",
                        content=f"data{i} " * result_size,
                        tool_call_id=f"c{i}",
                    )
                ]
            )
        )
    return messages


def _script(roles: list[str]) -> FunctionModel:
    """Answers as either the summarizer or the agent, recording which."""

    def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if "summarization assistant" in (info.instructions or ""):
            roles.append("summarizer")
            return ModelResponse(parts=[TextPart(content="## Intent\naudit the files")])
        roles.append("agent")
        return ModelResponse(parts=[TextPart(content="done")])

    return FunctionModel(script)


def _streamable() -> FunctionModel:
    """A FunctionModel that can serve a streamed request."""

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        yield "done"

    return FunctionModel(stream_function=stream)


def _compacting_settings(tmp_path: Path) -> EljaSettings:
    return EljaSettings(
        workspace=WorkspaceConfig(root=tmp_path),
        compaction=CompactionConfig(target_tokens=1000, keep_tool_pairs=1, keep_messages=2),
    )


class TestAdmissionRunsBeforeEveryRequest:
    async def test_the_main_guard_also_governs_the_summarizer(self, tmp_path: Path) -> None:
        """With no summarizer override, compaction reuses the guarded object."""
        settings = _compacting_settings(tmp_path)
        admissions: list[str] = []
        roles: list[str] = []
        guarded = GuardedModel(_script(roles), "main", admissions)
        agent: Agent[EljaDeps, str] = Agent(
            guarded, deps_type=EljaDeps, capabilities=build_compaction(settings)
        )
        result = await agent.run(
            "continue",
            message_history=_bulky_history(8),
            deps=EljaDeps.from_settings(settings),
        )
        assert result.output == "done"
        # The summarizer really ran, and it ran through the guard.
        assert roles == ["summarizer", "agent"]
        assert admissions == ["main", "main"]

    async def test_a_supplied_summarizer_guard_stays_separate(self, tmp_path: Path) -> None:
        """Separate instances are how a host tells main from compaction."""
        settings = _compacting_settings(tmp_path)
        admissions: list[str] = []
        roles: list[str] = []
        main = GuardedModel(_script(roles), "main", admissions)
        summarizer = GuardedModel(_script(roles), "summarizer", admissions)
        agent: Agent[EljaDeps, str] = Agent(
            main,
            deps_type=EljaDeps,
            capabilities=build_compaction(settings, summarizer_model=summarizer),
        )
        result = await agent.run(
            "continue",
            message_history=_bulky_history(8),
            deps=EljaDeps.from_settings(settings),
        )
        assert result.output == "done"
        assert admissions == ["summarizer", "main"]
        assert summarizer.dispatches == ["summarizer"]
        assert main.dispatches == ["main"]

    def test_a_supplied_summarizer_model_is_not_rebuilt(self, tmp_path: Path) -> None:
        """The instance reaches the strategy itself, wrapper and client intact."""
        settings = _compacting_settings(tmp_path)
        summarizer = GuardedModel(_script([]), "summarizer", [])
        (cap,) = build_compaction(settings, summarizer_model=summarizer)
        tiers = cap.tiers  # type: ignore[attr-defined]
        assert tiers[1].model is summarizer

    def test_no_summarizer_override_leaves_the_strategy_inheriting(self, tmp_path: Path) -> None:
        """Backward compatibility: the default is still "inherit the run's model"."""
        (cap,) = build_compaction(_compacting_settings(tmp_path))
        tiers = cap.tiers  # type: ignore[attr-defined]
        assert tiers[1].model is None


class TestDenialIsTerminal:
    async def test_a_denied_main_request_reaches_the_caller_and_dispatches_nothing(
        self, tmp_path: Path
    ) -> None:
        settings = _compacting_settings(tmp_path)
        admissions: list[str] = []
        guarded = GuardedModel(_script([]), "main", admissions, deny=True)
        agent: Agent[EljaDeps, str] = Agent(guarded, deps_type=EljaDeps)
        with pytest.raises(BudgetDeniedError):
            await agent.run("go", deps=EljaDeps.from_settings(settings))
        # Admitted once, refused, and never dispatched — no retry, no fallback.
        assert admissions == ["main"]
        assert guarded.dispatches == []

    async def test_a_denied_summarizer_request_is_not_retried(self, tmp_path: Path) -> None:
        """A compaction denial aborts the run instead of looping the summarizer."""
        settings = _compacting_settings(tmp_path)
        admissions: list[str] = []
        main = GuardedModel(_script([]), "main", admissions)
        summarizer = GuardedModel(_script([]), "summarizer", admissions, deny=True)
        agent: Agent[EljaDeps, str] = Agent(
            main,
            deps_type=EljaDeps,
            capabilities=build_compaction(settings, summarizer_model=summarizer),
        )
        with pytest.raises(BudgetDeniedError):
            await agent.run(
                "continue",
                message_history=_bulky_history(8),
                deps=EljaDeps.from_settings(settings),
            )
        assert admissions == ["summarizer"]
        assert summarizer.dispatches == []
        assert main.dispatches == []

    async def test_a_denial_during_streaming_is_also_terminal(self, tmp_path: Path) -> None:
        settings = _compacting_settings(tmp_path)
        admissions: list[str] = []
        guarded = GuardedModel(_streamable(), "main", admissions, deny=True)
        agent: Agent[EljaDeps, str] = Agent(guarded, deps_type=EljaDeps)
        with pytest.raises(BudgetDeniedError):
            async with agent.run_stream_events(
                "go", deps=EljaDeps.from_settings(settings)
            ) as events:
                async for _event in events:
                    pass
        assert admissions == ["main"]
        assert guarded.dispatches == []

    async def test_an_admitted_streamed_request_does_reach_the_provider(
        self, tmp_path: Path
    ) -> None:
        """The positive control: without it, the denial test above proves nothing."""
        settings = _compacting_settings(tmp_path)
        admissions: list[str] = []
        guarded = GuardedModel(_streamable(), "main", admissions)
        agent: Agent[EljaDeps, str] = Agent(guarded, deps_type=EljaDeps)
        async with agent.run_stream_events("go", deps=EljaDeps.from_settings(settings)) as events:
            async for _event in events:
                pass
        assert admissions == ["main"]
        assert guarded.dispatches == ["main"]
        # A streamed wrapper hook receives the run context; `request` does not.
        assert guarded.stream_had_run_context == [True]

    async def test_a_denial_after_a_tool_raised_is_not_turned_into_a_retry(
        self, tmp_path: Path
    ) -> None:
        """A tool's ModelRetry must not outlive a budget denial on the next turn."""
        settings = _compacting_settings(tmp_path)
        admissions: list[str] = []
        toolset: FunctionToolset[EljaDeps] = FunctionToolset()

        @toolset.tool
        def flaky(ctx: RunContext[EljaDeps]) -> str:
            """Always asks the model to try again."""
            raise ModelRetry("try something else")

        turns: list[int] = []

        def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            turns.append(1)
            return ModelResponse(parts=[ToolCallPart(tool_name="flaky", args={})])

        guard = GuardedModel(FunctionModel(script), "main", admissions)
        agent: Agent[EljaDeps, str] = Agent(guard, deps_type=EljaDeps, toolsets=[toolset])
        # Deny from the second request onward: the first turn's ModelRetry is
        # pending, and the denial must still end the run.
        original_admit = guard._admit

        def admit_then_deny() -> None:
            original_admit()
            if len(admissions) >= 2:
                raise BudgetDeniedError("main: budget exhausted")

        guard._admit = admit_then_deny  # type: ignore[method-assign]
        with pytest.raises(BudgetDeniedError):
            await agent.run("go", deps=EljaDeps.from_settings(settings))
        assert len(turns) == 1
        assert guard.dispatches == ["main"]


class TestUsageAttribution:
    async def test_the_summary_attempt_is_counted_once(self, tmp_path: Path) -> None:
        """Parent totals roll the summary up; they do not post it twice."""
        settings = _compacting_settings(tmp_path)
        roles: list[str] = []
        admissions: list[str] = []
        guarded = GuardedModel(_script(roles), "main", admissions)
        agent: Agent[EljaDeps, str] = Agent(
            guarded, deps_type=EljaDeps, capabilities=build_compaction(settings)
        )
        result = await agent.run(
            "continue",
            message_history=_bulky_history(8),
            deps=EljaDeps.from_settings(settings),
        )
        assert roles == ["summarizer", "agent"]
        # One summarizer request + one agent request, each counted exactly once.
        assert result.usage.requests == 2
        assert len(admissions) == 2
