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
from pydantic_ai.usage import RequestUsage, UsageLimits

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

    def _admit(self, hook: str = "request") -> None:
        """Record WHICH hook asked, so a denial test says what it claims."""
        self.admissions.append(f"{self.tag}:{hook}")
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

    async def count_tokens(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> RequestUsage:
        """A third provider-reaching hook, and a real call where it is implemented.

        pydantic-ai invokes this before EVERY request when
        ``UsageLimits.count_tokens_before_request`` is set, and routes it through
        ``check_allow_model_requests()`` like any other model request. A guard on
        ``request``/``request_stream`` alone never sees it.

        Not every provider implements it: ``Model.count_tokens`` raises
        ``NotImplementedError`` and ``OpenAIChatModel`` — elja's default — does not
        override it, which is why this class supplies its own rather than relying on
        a wrapped model to have one.
        """
        self._admit("count_tokens")
        self.dispatches.append(f"{self.tag}:count_tokens")
        return await super().count_tokens(messages, model_settings, model_request_parameters)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        """The streamed path needs its own gate; `request` does not cover it."""
        self._admit("request_stream")
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


def _counting_model() -> Model:
    """A model that implements count_tokens, which FunctionModel does not."""

    class Counting(WrapperModel):
        async def count_tokens(
            self,
            messages: list[ModelMessage],
            model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters,
        ) -> RequestUsage:
            """Answer without a network call, as a provider would with one."""
            return RequestUsage(input_tokens=11)

    return Counting(_script([]))


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
        assert admissions == ["main:request", "main:request"]

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
        assert admissions == ["summarizer:request", "main:request"]
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


class TestCountTokensIsGuardedToo:
    """The hook a two-method guard misses entirely."""

    async def test_admission_fires_for_the_count_tokens_dispatch(self, tmp_path: Path) -> None:
        settings = _compacting_settings(tmp_path)
        admissions: list[str] = []
        guarded = GuardedModel(_counting_model(), "main", admissions)
        agent: Agent[EljaDeps, str] = Agent(guarded, deps_type=EljaDeps)
        await agent.run(
            "go",
            deps=EljaDeps.from_settings(settings),
            usage_limits=UsageLimits(request_limit=5, count_tokens_before_request=True),
        )
        # Counted first, then the request itself — both through the guard.
        assert guarded.dispatches == ["main:count_tokens", "main"]
        assert admissions == ["main:count_tokens", "main:request"]

    async def test_a_denial_stops_the_count_tokens_dispatch(self, tmp_path: Path) -> None:
        settings = _compacting_settings(tmp_path)
        admissions: list[str] = []
        guarded = GuardedModel(_counting_model(), "main", admissions, deny=True)
        agent: Agent[EljaDeps, str] = Agent(guarded, deps_type=EljaDeps)
        with pytest.raises(BudgetDeniedError):
            await agent.run(
                "go",
                deps=EljaDeps.from_settings(settings),
                usage_limits=UsageLimits(request_limit=5, count_tokens_before_request=True),
            )
        # Naming the hook is the point: without it this test passes unchanged
        # when count_tokens is never consulted and `request` denies instead.
        assert admissions == ["main:count_tokens"]
        assert guarded.dispatches == []

    async def test_without_the_flag_count_tokens_is_never_reached(self, tmp_path: Path) -> None:
        """So the assertions above are about the flag, not about every run."""
        settings = _compacting_settings(tmp_path)
        admissions: list[str] = []
        guarded = GuardedModel(_counting_model(), "main", admissions)
        agent: Agent[EljaDeps, str] = Agent(guarded, deps_type=EljaDeps)
        await agent.run("go", deps=EljaDeps.from_settings(settings))
        assert guarded.dispatches == ["main"]


class TestSummarizerAttribution:
    async def test_model_settings_distinguish_compaction_from_a_main_request(
        self, tmp_path: Path
    ) -> None:
        """One guard instance, two phases — no upstream patch needed."""
        settings = _compacting_settings(tmp_path)
        phases: list[str | None] = []

        class PhaseReadingGuard(WrapperModel):
            async def request(
                self,
                messages: list[ModelMessage],
                model_settings: ModelSettings | None,
                model_request_parameters: ModelRequestParameters,
            ) -> ModelResponse:
                """Record the phase tag the settings carried."""
                headers = (model_settings or {}).get("extra_headers") or {}
                phases.append(headers.get("x-phase"))
                return await super().request(messages, model_settings, model_request_parameters)

        agent: Agent[EljaDeps, str] = Agent(
            PhaseReadingGuard(_script([])),
            deps_type=EljaDeps,
            capabilities=build_compaction(
                settings,
                summarizer_model_settings={"extra_headers": {"x-phase": "compaction"}},
            ),
        )
        await agent.run(
            "continue",
            message_history=_bulky_history(8),
            deps=EljaDeps.from_settings(settings),
        )
        assert phases == ["compaction", None]

    async def test_agent_level_settings_never_reach_the_compaction_request(
        self, tmp_path: Path
    ) -> None:
        """Because `_summarize` builds a SEPARATE agent, not because of merging.

        The doc said the loss was `merge_model_settings`' shallowness — i.e. a
        consequence of *having passed* `summarizer_model_settings`. It is not: the
        parent agent's `model_settings` are never handed to the summarizer's private
        agent at all, so with nothing passed the compaction request goes out with
        `model_settings=None`. For a host carrying gateway auth at the agent level
        that is an unauthenticated and unbounded paid request, deep in a long
        conversation, after a paid main turn.

        Three rows, measured in one place so the three channels cannot drift apart.
        """
        settings = _compacting_settings(tmp_path)
        carried: ModelSettings = {
            "temperature": 0.1,
            "max_tokens": 77,
            "extra_headers": {"authorization": "Bearer gw"},
        }

        async def phases_for(
            *,
            agent_level: ModelSettings | None,
            on_model: ModelSettings | None,
            summarizer: ModelSettings | None,
        ) -> list[ModelSettings | None]:
            seen: list[ModelSettings | None] = []

            class Recorder(WrapperModel):
                async def request(
                    self,
                    messages: list[ModelMessage],
                    model_settings: ModelSettings | None,
                    model_request_parameters: ModelRequestParameters,
                ) -> ModelResponse:
                    seen.append(model_settings)
                    return await super().request(
                        messages, model_settings, model_request_parameters
                    )

            def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
                summarizing = "summarization assistant" in (info.instructions or "")
                return ModelResponse(
                    parts=[TextPart(content="## Intent\nx" if summarizing else "done")]
                )

            agent: Agent[EljaDeps, str] = Agent(
                Recorder(FunctionModel(script, settings=on_model)),
                deps_type=EljaDeps,
                capabilities=build_compaction(settings, summarizer_model_settings=summarizer),
                model_settings=agent_level,
            )
            await agent.run(
                "continue",
                message_history=_bulky_history(8),
                deps=EljaDeps.from_settings(settings),
            )
            assert len(seen) == 2, "the summarizer did not run; the case is the wrong one"
            return seen

        # 1. Agent level, nothing restated: the compaction request gets NOTHING.
        compaction, main = await phases_for(agent_level=carried, on_model=None, summarizer=None)
        assert compaction is None
        assert main == carried

        # 2. On the model object instead: it survives, which is the remedy.
        compaction, main = await phases_for(agent_level=None, on_model=carried, summarizer=None)
        assert compaction == carried
        assert main == carried

        # 3. Restated per key, and the restatement REPLACES rather than merges:
        # `authorization` is gone from the compaction request while the model's
        # other two settings survive untouched.
        compaction, main = await phases_for(
            agent_level=None,
            on_model=carried,
            summarizer={"extra_headers": {"x-phase": "compaction"}},
        )
        assert compaction is not None
        assert compaction["extra_headers"] == {"x-phase": "compaction"}
        assert compaction["max_tokens"] == 77
        assert (main or {}).get("extra_headers") == {"authorization": "Bearer gw"}


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
        assert admissions == ["main:request"]
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
        assert admissions == ["summarizer:request"]
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
        assert admissions == ["main:request_stream"]
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
        assert admissions == ["main:request_stream"]
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

        guard._admit = admit_then_deny  # type: ignore[method-assign,assignment]
        with pytest.raises(BudgetDeniedError):
            await agent.run("go", deps=EljaDeps.from_settings(settings))
        assert len(turns) == 1
        assert guard.dispatches == ["main"]


class TestUsageAttribution:
    async def test_the_summary_attempt_is_billed_exactly_once(self, tmp_path: Path) -> None:
        """Parent totals roll the summary up; they do not post it twice.

        ``usage.requests`` counts committed request STEPS, not provider
        dispatches, so a double-posted attempt would be invisible in it. The
        claim has to be made against the tokens: give the two phases distinct
        usage and assert the parent total is the sum, once.
        """
        settings = _compacting_settings(tmp_path)
        roles: list[str] = []
        admissions: list[str] = []

        def billed(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if "summarization assistant" in (info.instructions or ""):
                roles.append("summarizer")
                return ModelResponse(
                    parts=[TextPart(content="## Intent\naudit the files")],
                    usage=RequestUsage(input_tokens=1000, output_tokens=7),
                )
            roles.append("agent")
            return ModelResponse(
                parts=[TextPart(content="done")],
                usage=RequestUsage(input_tokens=30, output_tokens=3),
            )

        guarded = GuardedModel(FunctionModel(billed), "main", admissions)
        agent: Agent[EljaDeps, str] = Agent(
            guarded, deps_type=EljaDeps, capabilities=build_compaction(settings)
        )
        result = await agent.run(
            "continue",
            message_history=_bulky_history(8),
            deps=EljaDeps.from_settings(settings),
        )
        assert roles == ["summarizer", "agent"]
        assert result.usage.input_tokens == 1030
        assert result.usage.output_tokens == 10
        # Companion, not the claim: two committed steps, two admissions.
        assert result.usage.requests == 2
        assert len(admissions) == 2


class TestCountTokensBeforeRequestIsNotSafeToTurnOn:
    """The doc claim this guards used to read as benign-where-unsupported.

    It is the opposite. `Model.count_tokens` raises `NotImplementedError`, and the
    model class elja builds for its own default provider does not override it — so
    a host that reads "it only does something on providers that implement it",
    turns the flag on over an LM Studio endpoint, and ships, fails every request.
    """

    def test_eljas_default_model_does_not_implement_it(self) -> None:
        from elja.model import build_model

        model = build_model(EljaSettings())
        assert type(model).__name__ == "OpenAIChatModel"
        assert type(model).count_tokens is Model.count_tokens

    async def test_the_flag_raises_rather_than_no_opping(self) -> None:
        """Driven through elja's own default model, which is the claim's subject.

        A `WrapperModel` stand-in would pass for a neighbouring reason: `WrapperModel`
        *does* define `count_tokens` in order to delegate it, so the
        `NotImplementedError` would come from whatever it wrapped. If upstream gave
        `WrapperModel` a real local estimate — the shape `OpenAIEmbeddingModel`
        already uses — the stand-in would go green while `OpenAIChatModel` kept
        raising. `build_model` needs no API key and `count_tokens` raises before any
        dispatch, so the real boundary is free to drive, and this also pins the exact
        message `docs/EMBEDDING.md` quotes.
        """
        from elja.model import build_model

        agent: Agent[None, str] = Agent(build_model(EljaSettings()))
        with pytest.raises(
            NotImplementedError,
            match="Token counting ahead of the request is not supported by OpenAIChatModel",
        ):
            await agent.run("go", usage_limits=UsageLimits(count_tokens_before_request=True))

    def test_the_two_providers_that_do_implement_it(self) -> None:
        """Named in the doc, so a provider dropping its implementation shows up here."""
        from elja.model import build_model

        for provider in ("anthropic", "google"):
            model = build_model(EljaSettings(model={"provider": provider, "api_key": "x"}))  # type: ignore[arg-type]
            assert type(model).count_tokens is not Model.count_tokens, provider
