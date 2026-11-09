"""E4: the summarizer is bounded, and an unreachable target is a typed outcome.

``tests/test_compaction_composition.py`` holds the issue's own three shapes over a
six-step turn. This file pins the mechanism underneath, one acceptance criterion at a
time: the budget counts requests (hidden retries included), unchanged unfit context is
not re-summarized, the latch resets exactly where the docs say, protected content and
tool pairing survive both the fitting and the cannot-fit path, the host's guard keeps
its semantics, and reporting placed after compaction reads the request that was sent.

**Scripted agent responses carry ``RequestUsage(output_tokens=1)``.** Left alone,
``FunctionModel`` invents a word-count usage for every response, and the estimator
anchors on it as if a provider had reported it — so a history of long words looks a
third of its size and the target stops binding after the first request. A usage with
no input tokens is not an anchor, which keeps the estimate on the heuristic the target
was written against. Where anchoring itself is the point, the test says so.
"""

import asyncio
from collections.abc import AsyncIterator, Callable
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai import Agent, CancellationToken, RunContext
from pydantic_ai.exceptions import RunCancelled, UnexpectedModelBehavior
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextContent,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.usage import RequestUsage, RunUsage
from pydantic_ai_harness.compaction import (
    ClearToolResults,
    ContextUsage,
    ReportContextUsage,
    SummarizingCompaction,
    TieredCompaction,
    compact_now,
    is_pinned,
    pin,
)
from pydantic_ai_harness.compaction._receipts import is_receipt_part, make_receipt_part
from pytest_mock import MockerFixture

from elja.compaction import (
    BoundedTieredCompaction,
    OnUnfit,
    TargetUnreachable,
    TargetUnreachableError,
    UnfitReason,
    _drop_compaction_copies,
    _first_user_message,
    _is_harness_marker,
    _leading_system_parts,
    _MeteredSummarizer,
    _protected_floor,
    _RunState,
    _SummarizerBudgetSpentError,
    build_compaction,
    proceed_over_target,
)
from elja.deps import EljaDeps
from elja.settings import CompactionConfig, EljaSettings, SubagentConfig, WorkspaceConfig

NO_ANCHOR = RequestUsage(output_tokens=1)
HOST_PLACEHOLDER = "[result cleared; retrieve the saved evidence by its receipt id]"


class BudgetDeniedError(Exception):
    """A host's admission refusal."""


def _settings(tmp_path: Path, *, target: int = 1000, budget: int = 3) -> EljaSettings:
    return EljaSettings(
        workspace=WorkspaceConfig(root=tmp_path),
        compaction=CompactionConfig(
            target_tokens=target,
            keep_tool_pairs=1,
            keep_messages=2,
            max_summarizer_requests=budget,
        ),
    )


def _history(
    pairs: int, *, pinned: str | None = None, first: str = "original task: reconcile the ledger"
) -> list[ModelMessage]:
    """Tool pairs with bulky results, too big for the target."""
    messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content=first)])]
    if pinned is not None:
        messages.append(ModelRequest(parts=[pin(pinned)]))
    for i in range(pairs):
        messages.append(
            ModelResponse(
                parts=[ToolCallPart(tool_name="read_row", args={"row": i}, tool_call_id=f"c{i}")]
            )
        )
        messages.append(
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name="read_row", content=f"row{i} " * 600, tool_call_id=f"c{i}"
                    )
                ]
            )
        )
    return messages


def _recording(reports: list[TargetUnreachable]) -> OnUnfit[Any]:
    """A gate that records and lets the request go."""

    def gate(ctx: RunContext[Any], report: TargetUnreachable) -> None:
        reports.append(report)

    return gate


def _toolset() -> FunctionToolset[EljaDeps]:
    toolset: FunctionToolset[EljaDeps] = FunctionToolset()

    @toolset.tool
    async def ping(ctx: RunContext[EljaDeps]) -> str:
        """A cheap tool, so one turn takes several requests."""
        return "pong"

    return toolset


def _is_summarizer(info: AgentInfo) -> bool:
    return "summarization assistant" in (info.instructions or "")


def _script(
    roles: list[str],
    *,
    steps: int = 6,
    narration: int = 0,
    summaries: Callable[[int], str] = lambda n: "## Intent\nledger",
) -> FunctionModel:
    """Answer as summarizer or agent, recording which; the agent pings ``steps - 1`` times.

    ``narration`` words of assistant text per step make the history grow in a way
    masking cannot reclaim — the shape of a long run that legitimately needs more than
    one summary.
    """

    def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if _is_summarizer(info):
            roles.append("summarizer")
            return ModelResponse(parts=[TextPart(content=summaries(roles.count("summarizer")))])
        roles.append("agent")
        if roles.count("agent") < steps:
            text = [TextPart(content="narration " * narration)] if narration else []
            return ModelResponse(
                parts=[*text, ToolCallPart(tool_name="ping", args={})], usage=NO_ANCHOR
            )
        return ModelResponse(parts=[TextPart(content="ok")], usage=NO_ANCHOR)

    return FunctionModel(script)


def _agent(
    model: Model,
    settings: EljaSettings,
    *,
    on_unfit: OnUnfit[Any] | None = None,
    extra: list[Any] | None = None,
    summarizer_model: Model | None = None,
    cleared_placeholder: str | None = None,
    receipts: bool = False,
) -> Agent[EljaDeps, str]:
    compaction = build_compaction(
        settings,
        on_unfit=on_unfit,
        summarizer_model=summarizer_model,
        cleared_placeholder=cleared_placeholder,
        receipts=receipts,
    )
    return Agent(
        model,
        deps_type=EljaDeps,
        toolsets=[_toolset()],
        capabilities=[*compaction, *(extra or [])],
    )


class GuardedModel(WrapperModel):
    """A host's metering wrapper: admit, then dispatch."""

    def __init__(self, wrapped: Model, *, deny: bool = False) -> None:
        """Wrap a model with a host-style admission gate."""
        super().__init__(wrapped)
        self.deny = deny
        self.admissions = 0
        self.dispatches = 0

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Admit before dispatching, so a denial costs no provider call."""
        self.admissions += 1
        if self.deny:
            raise BudgetDeniedError("budget exhausted")
        self.dispatches += 1
        return await super().request(messages, model_settings, model_request_parameters)


class TestTheBudgetCountsRequests:
    """An explicit bound, counted where the money is: at the summarizer's model."""

    @pytest.mark.parametrize("budget", [1, 2, 3])
    async def test_a_run_that_keeps_growing_gets_exactly_its_budget(
        self, tmp_path: Path, budget: int
    ) -> None:
        """The positive control and the bound in one: each summary fits, then one more is needed.

        Assistant narration is not maskable, so this run legitimately needs a summary
        every few steps. It gets precisely ``budget`` of them, and the next request that
        would need one gets a typed ``budget_exhausted`` instead of a paid call.
        """
        roles: list[str] = []
        settings = _settings(tmp_path, budget=budget)
        agent = _agent(_script(roles, steps=16, narration=300), settings)
        with pytest.raises(TargetUnreachableError) as raised:
            await agent.run("go", deps=EljaDeps.from_settings(settings))
        assert roles.count("summarizer") == budget
        report = raised.value.report
        assert report.reason == "budget_exhausted"
        assert report.summarizer_requests == report.max_summarizer_requests == budget

    async def test_a_zero_budget_is_masking_only(self, tmp_path: Path) -> None:
        roles: list[str] = []
        reports: list[TargetUnreachable] = []
        settings = _settings(tmp_path, budget=0)
        agent = _agent(_script(roles), settings, on_unfit=_recording(reports))
        await agent.run(
            "carry on", message_history=_history(8), deps=EljaDeps.from_settings(settings)
        )
        assert "summarizer" not in roles
        assert reports, "nothing was over target; the case is the wrong one"
        assert {report.reason for report in reports} == {"budget_exhausted"}

    @pytest.mark.parametrize(
        ("budget", "dispatched"),
        [(1, 1), (3, 2)],
        ids=["retry-refused-at-the-budget", "retry-counted-within-it"],
    )
    async def test_a_hidden_retry_inside_the_summarizer_cannot_evade_the_bound(
        self, tmp_path: Path, budget: int, dispatched: int
    ) -> None:
        """A reply with no text makes pydantic-ai's summary agent retry: two paid requests.

        Counting *attempts* would call that one. The bound counts requests at the
        model, so with a budget of one the retry is refused before it reaches the
        host's guard; with room for it, it goes through the guard and is counted.
        """
        summarizer = GuardedModel(
            FunctionModel(lambda messages, info: ModelResponse(parts=[]))  # no text at all
        )
        settings = _settings(tmp_path, budget=budget)
        agent = _agent(_script([]), settings, summarizer_model=summarizer)
        if budget == 1:
            with pytest.raises(TargetUnreachableError) as raised:
                await agent.run(
                    "carry on", message_history=_history(8), deps=EljaDeps.from_settings(settings)
                )
            assert raised.value.report.reason == "budget_exhausted"
            assert raised.value.report.summarizer_requests == 1
        else:
            with pytest.raises(UnexpectedModelBehavior):
                await agent.run(
                    "carry on", message_history=_history(8), deps=EljaDeps.from_settings(settings)
                )
        assert summarizer.dispatches == dispatched
        assert summarizer.admissions == dispatched


class TestUnchangedUnfitContextIsNotResummarized:
    """The latch, and exactly where it resets."""

    # Under the target alone, so the prediction lets the summary run; over it once a
    # 300-word summary is added, so the summary cannot make the request fit.
    PIN = "KEEP " * 560
    BIG_SUMMARY = "## Intent\n" + "detail " * 300

    async def test_a_summary_that_cannot_fit_is_paid_for_once(self, tmp_path: Path) -> None:
        """Raise mode: one summary, then the typed outcome. No second attempt."""
        roles: list[str] = []
        settings = _settings(tmp_path)
        agent = _agent(_script(roles, summaries=lambda n: self.BIG_SUMMARY), settings)
        with pytest.raises(TargetUnreachableError) as raised:
            await agent.run(
                "carry on",
                message_history=_history(8, pinned=self.PIN),
                deps=EljaDeps.from_settings(settings),
            )
        assert roles == ["summarizer"]
        report = raised.value.report
        assert report.reason == "no_progress"
        assert report.protected_tokens <= report.target_tokens < report.estimated_tokens
        assert report.summarizer_requests == 1

    async def test_later_requests_are_reported_without_another_summary(
        self, tmp_path: Path
    ) -> None:
        """Proceed mode: every request is put to the gate; the summarizer is not re-entered."""
        roles: list[str] = []
        reports: list[TargetUnreachable] = []
        settings = _settings(tmp_path)
        agent = _agent(
            _script(roles, summaries=lambda n: self.BIG_SUMMARY),
            settings,
            on_unfit=_recording(reports),
        )
        await agent.run(
            "carry on",
            message_history=_history(8, pinned=self.PIN),
            deps=EljaDeps.from_settings(settings),
        )
        assert roles.count("summarizer") == 1
        assert roles.count("agent") == 6
        assert [r.occurrence for r in reports] == [1, 2, 3, 4, 5, 6]
        assert {r.reason for r in reports} == {"no_progress"}
        assert {r.summarizer_requests for r in reports} == {1}

    async def test_a_new_run_starts_with_a_fresh_budget_and_latch(self, tmp_path: Path) -> None:
        """Reset semantics: per run. The second run may try once more, and says so from 1."""
        roles: list[str] = []
        this_run: list[str] = []
        reports: list[TargetUnreachable] = []

        def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if _is_summarizer(info):
                roles.append("summarizer")
                return ModelResponse(parts=[TextPart(content=self.BIG_SUMMARY)])
            this_run.append("agent")
            if len(this_run) < 2:
                return ModelResponse(
                    parts=[ToolCallPart(tool_name="ping", args={})], usage=NO_ANCHOR
                )
            return ModelResponse(parts=[TextPart(content="ok")], usage=NO_ANCHOR)

        settings = _settings(tmp_path)
        agent = _agent(FunctionModel(script), settings, on_unfit=_recording(reports))
        deps = EljaDeps.from_settings(settings)
        for _ in range(2):
            this_run.clear()
            await agent.run("carry on", message_history=_history(8, pinned=self.PIN), deps=deps)
        assert roles.count("summarizer") == 2
        assert [(r.occurrence, r.summarizer_requests) for r in reports] == [
            (1, 1),
            (2, 1),
            (1, 1),
            (2, 1),
        ]

    async def test_overflow_mode_resummarizes_only_on_real_progress_then_leaves(
        self, tmp_path: Path
    ) -> None:
        """Recovery within a run: a later summary that fits ends overflow mode.

        The first summary is too big, so the run enters overflow mode. Assistant
        narration then grows the summarizable remainder past a whole target — real
        progress — so a second summary is allowed; it is small, the request fits, and
        overflow mode ends: the requests after it fit and report nothing.
        """
        roles: list[str] = []
        reports: list[TargetUnreachable] = []
        # Budget to spare, so what follows recovery is not cut short by it.
        settings = _settings(tmp_path, budget=10)
        agent = _agent(
            _script(
                roles,
                steps=8,
                narration=150,
                summaries=lambda n: self.BIG_SUMMARY if n == 1 else "## Intent\nledger",
            ),
            settings,
            on_unfit=_recording(reports),
        )
        await agent.run(
            "carry on",
            message_history=_history(8, pinned="KEEP " * 400),
            deps=EljaDeps.from_settings(settings),
        )
        after_first = roles.index("summarizer") + 1
        second = roles.index("summarizer", after_first)
        # More than one agent request between the first two summaries: overflow mode
        # waited for real progress instead of re-summarizing on every request.
        assert roles[after_first:second].count("agent") > 1
        # Every report was made in overflow mode, before the summary that fit. After
        # it the run is back in normal mode: it summarizes as soon as it is over
        # target again (narration this size needs that), and each of those fits.
        assert reports and {r.reason for r in reports} == {"no_progress"}
        assert len(reports) == roles[:second].count("agent")
        assert roles.count("summarizer") > 2

    async def test_a_large_current_request_is_protected_only_while_current(
        self, tmp_path: Path
    ) -> None:
        """After the context changes, the floor is measured afresh.

        A tool returns something bigger than the target. While it is the current
        request it is protected — compaction may not drop it — so that one request is
        reported. One step later it is history, masking clears it, and the run fits
        again without a summary having been paid for.
        """
        roles: list[str] = []
        reports: list[TargetUnreachable] = []
        settings = _settings(tmp_path)
        toolset: FunctionToolset[EljaDeps] = FunctionToolset()

        @toolset.tool
        async def dump(ctx: RunContext[EljaDeps]) -> str:
            """Return far more than the target."""
            return "BLOB " * 1500

        def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if _is_summarizer(info):
                roles.append("summarizer")
                return ModelResponse(parts=[TextPart(content="## Intent\nx")])
            roles.append("agent")
            calls = {1: "dump", 2: "ping", 3: "ping"}
            name = calls.get(roles.count("agent"))
            if name is not None:
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=name, args={})], usage=NO_ANCHOR
                )
            return ModelResponse(parts=[TextPart(content="ok")], usage=NO_ANCHOR)

        agent: Agent[EljaDeps, str] = Agent(
            FunctionModel(script),
            deps_type=EljaDeps,
            toolsets=[toolset, _toolset()],
            capabilities=build_compaction(settings, on_unfit=_recording(reports)),
        )
        await agent.run("go", deps=EljaDeps.from_settings(settings))
        assert "summarizer" not in roles
        assert [r.reason for r in reports] == ["protected_floor"]

    async def test_concurrent_runs_do_not_share_a_budget(self, tmp_path: Path) -> None:
        """Per-run state is per run, not per agent: two at once each get their own."""
        roles: list[str] = []
        settings = _settings(tmp_path, budget=1)
        agent = _agent(_script(roles, steps=2), settings)
        deps = EljaDeps.from_settings(settings)
        await asyncio.gather(
            *(agent.run("carry on", message_history=_history(8), deps=deps) for _ in range(2))
        )
        assert roles.count("summarizer") == 2


class TestThePredictionPaysNothing:
    async def test_the_summarizer_guard_is_never_asked(self, tmp_path: Path) -> None:
        """No dispatch and no admission: a host's budget is not even consulted."""
        summarizer = GuardedModel(_script([]))
        settings = _settings(tmp_path)
        agent = _agent(_script([]), settings, summarizer_model=summarizer)
        with pytest.raises(TargetUnreachableError):
            await agent.run(
                "carry on",
                message_history=_history(8, pinned="IRREDUCIBLE CONSTRAINT " * 500),
                deps=EljaDeps.from_settings(settings),
            )
        assert (summarizer.admissions, summarizer.dispatches) == (0, 0)

    async def test_instructions_count_toward_the_floor(self, tmp_path: Path) -> None:
        """They are sent on every request and no tier can remove them, so they are protected."""
        roles: list[str] = []
        settings = _settings(tmp_path)
        agent: Agent[EljaDeps, str] = Agent(
            _script(roles),
            deps_type=EljaDeps,
            instructions="Follow the house style. " * 200,
            capabilities=build_compaction(settings),
        )
        with pytest.raises(TargetUnreachableError) as raised:
            await agent.run(
                "carry on", message_history=_history(8), deps=EljaDeps.from_settings(settings)
            )
        assert raised.value.report.reason == "protected_floor"
        assert roles == []


class TestTheHostsGuardKeepsItsSemantics:
    async def test_a_denial_is_terminal_and_is_not_turned_into_a_capacity_outcome(
        self, tmp_path: Path
    ) -> None:
        """The metering wrapper delegates to the guard; the guard's exception wins."""
        summarizer = GuardedModel(_script([]), deny=True)
        main = GuardedModel(_script([]))
        settings = _settings(tmp_path)
        agent = _agent(main, settings, summarizer_model=summarizer)
        with pytest.raises(BudgetDeniedError):
            await agent.run(
                "carry on", message_history=_history(8), deps=EljaDeps.from_settings(settings)
            )
        assert (summarizer.admissions, summarizer.dispatches, main.dispatches) == (1, 0, 0)

    async def test_cancellation_stops_further_attempts(self, tmp_path: Path) -> None:
        """Cancelled while the summary is in flight: no retry, no next attempt, no main call."""
        token = CancellationToken()
        roles: list[str] = []

        async def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if _is_summarizer(info):
                roles.append("summarizer")
                token.cancel()
                await asyncio.sleep(5)
                raise AssertionError(
                    "cancellation did not reach the summarizer"
                )  # pragma: no cover
            roles.append("agent")  # pragma: no cover
            return ModelResponse(parts=[TextPart(content="ok")])  # pragma: no cover

        settings = _settings(tmp_path)
        agent = _agent(FunctionModel(script), settings, on_unfit=_recording([]))
        with pytest.raises(RunCancelled):
            await agent.run(
                "carry on",
                message_history=_history(8),
                deps=EljaDeps.from_settings(settings),
                cancellation_token=token,
            )
        assert roles == ["summarizer"]

    async def test_usage_is_attributed_once_on_the_cannot_fit_path(self, tmp_path: Path) -> None:
        """The summary's tokens roll up into the parent once, as on the fitting path."""

        def billed(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if _is_summarizer(info):
                return ModelResponse(
                    parts=[TextPart(content="## Intent\n" + "detail " * 300)],
                    usage=RequestUsage(input_tokens=1000, output_tokens=7),
                )
            return ModelResponse(
                parts=[TextPart(content="ok")],
                usage=RequestUsage(input_tokens=30, output_tokens=3),
            )

        settings = _settings(tmp_path)
        agent = _agent(FunctionModel(billed), settings, on_unfit=_recording([]))
        result = await agent.run(
            "carry on",
            message_history=_history(8, pinned="KEEP " * 560),
            deps=EljaDeps.from_settings(settings),
        )
        assert (result.usage.input_tokens, result.usage.output_tokens) == (1030, 10)


def _pairs(view: list[ModelMessage]) -> tuple[set[str], set[str]]:
    calls = {p.tool_call_id for m in view for p in m.parts if isinstance(p, ToolCallPart)}
    returns = {p.tool_call_id for m in view for p in m.parts if isinstance(p, ToolReturnPart)}
    return calls, returns


class TestWhatSurvivesBothPaths:
    """Pins, the current request, pairing, opaque metadata, placeholder and receipts.

    Three paths, because no single one exercises every property: masking alone fits
    (no summary, so no receipt), a summary fits (the tail is emptied, so no masked
    placeholder), and the cannot-fit path the host chose to send — which has both, and
    is the one where something "protected" is most tempting to drop.
    """

    SOURCE = {"source": "ticket-42", "correction": "v2"}

    @pytest.mark.parametrize(
        ("target", "pairs", "pinned", "fits", "summarizes"),
        [
            (3000, 8, "NEVER DROP: tenant=acme", True, False),
            # Twelve pairs: the host's short placeholder lets masking alone fit eight.
            (1000, 12, "NEVER DROP: tenant=acme", True, True),
            (3000, 8, "NEVER DROP: tenant=acme " + "IRREDUCIBLE CONSTRAINT " * 500, False, True),
        ],
        ids=["fits-by-masking", "fits-by-summary", "cannot-fit"],
    )
    async def test_everything_protected_reaches_the_model(
        self,
        tmp_path: Path,
        target: int,
        pairs: int,
        pinned: str,
        fits: bool,
        summarizes: bool,
    ) -> None:
        views: list[list[ModelMessage]] = []
        roles: list[str] = []
        reports: list[TargetUnreachable] = []

        def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if _is_summarizer(info):
                roles.append("summarizer")
                return ModelResponse(parts=[TextPart(content="## Intent\nledger")])
            views.append(list(messages))
            return ModelResponse(parts=[TextPart(content="ok")])

        settings = _settings(tmp_path, target=target)
        agent = _agent(
            FunctionModel(script),
            settings,
            on_unfit=_recording(reports),
            cleared_placeholder=HOST_PLACEHOLDER,
            receipts=True,
        )
        history = _history(pairs, pinned=pinned)
        history[0] = ModelRequest(
            parts=[UserPromptPart(content=[TextContent("the task", metadata=self.SOURCE)])]
        )
        await agent.run(
            [TextContent("carry on", metadata=self.SOURCE)],
            message_history=history,
            deps=EljaDeps.from_settings(settings),
        )
        assert (reports == []) is fits
        assert ("summarizer" in roles) is summarizes
        (view,) = views
        parts = [part for message in view for part in message.parts]
        # The pin, once, verbatim.
        pins = [p.content for p in parts if isinstance(p, UserPromptPart) and is_pinned(p)]
        assert pins == [pin(pinned).content]
        # The current request is last, carrying its metadata.
        current = view[-1].parts[-1]
        assert isinstance(current, UserPromptPart)
        assert current.content == [TextContent("carry on", metadata=self.SOURCE)]
        # The first user message's opaque metadata, preserved with it.
        assert any(
            isinstance(p, UserPromptPart)
            and not isinstance(p.content, str)
            and any(isinstance(i, TextContent) and i.metadata == self.SOURCE for i in p.content)
            and "the task" in str(p.content)
            for p in parts
        )
        # Pairing is valid on the request that was actually sent.
        calls, returns = _pairs(view)
        assert calls == returns
        # The host's placeholder wherever a result was masked, never elja's default.
        assert "re-run the tool" not in str(view)
        assert (HOST_PLACEHOLDER in str(view)) is (target == 3000)
        # A receipt wherever history was summarized away.
        assert any(is_receipt_part(p) for p in parts) is summarizes

    async def test_on_the_raise_path_nothing_is_sent_and_the_history_is_untouched(
        self, tmp_path: Path
    ) -> None:
        sent: list[int] = []

        def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            sent.append(1)  # pragma: no cover - the point is that this never runs
            return ModelResponse(parts=[TextPart(content="ok")])  # pragma: no cover

        settings = _settings(tmp_path)
        agent = _agent(FunctionModel(script), settings)
        history = _history(8, pinned="IRREDUCIBLE CONSTRAINT " * 500)
        snapshot = deepcopy(history)
        with pytest.raises(TargetUnreachableError):
            await agent.run(
                "carry on", message_history=history, deps=EljaDeps.from_settings(settings)
            )
        assert sent == []
        assert history == snapshot


class TestReportingAfterCompaction:
    async def test_the_reading_is_the_request_that_was_sent(self, tmp_path: Path) -> None:
        """Over target on the cannot-fit path, and at most what the report predicted."""
        readings: list[ContextUsage] = []
        reports: list[TargetUnreachable] = []
        settings = _settings(tmp_path)
        agent = _agent(
            _script([], steps=1),
            settings,
            on_unfit=_recording(reports),
            extra=[ReportContextUsage(on_usage=readings.append)],
        )
        await agent.run(
            "carry on",
            message_history=_history(8, pinned="IRREDUCIBLE CONSTRAINT " * 500),
            deps=EljaDeps.from_settings(settings),
        )
        (reading,) = readings
        (report,) = reports
        assert report.target_tokens < reading.used_tokens <= report.estimated_tokens
        # The reliability metadata travels with both.
        assert reading.resolved is False
        assert report.window_tokens is None
        assert report.anchored is False

    async def test_no_reading_is_taken_for_a_request_that_was_never_sent(
        self, tmp_path: Path
    ) -> None:
        readings: list[ContextUsage] = []
        settings = _settings(tmp_path)
        agent = _agent(_script([]), settings, extra=[ReportContextUsage(on_usage=readings.append)])
        with pytest.raises(TargetUnreachableError):
            await agent.run(
                "carry on",
                message_history=_history(8, pinned="IRREDUCIBLE CONSTRAINT " * 500),
                deps=EljaDeps.from_settings(settings),
            )
        assert readings == []

    async def test_a_resolvable_window_and_an_anchor_are_reported(self, tmp_path: Path) -> None:
        """The hard limit, where pydantic-ai knows it, and whether usage anchored the estimate."""

        class Priced(FunctionModel):
            @property
            def model_id(self) -> str:
                return "openai:gpt-4o"

        anchored = _history(8, pinned="IRREDUCIBLE CONSTRAINT " * 500)
        response = anchored[2]
        assert isinstance(response, ModelResponse)
        anchored[2] = replace(response, usage=RequestUsage(input_tokens=5000, output_tokens=5))
        settings = _settings(tmp_path)
        agent = _agent(Priced(lambda messages, info: ModelResponse(parts=[])), settings)
        with pytest.raises(TargetUnreachableError) as raised:
            await agent.run(
                "carry on", message_history=anchored, deps=EljaDeps.from_settings(settings)
            )
        report = raised.value.report
        assert report.anchored is True
        assert report.window_tokens is not None and report.window_tokens > 100_000
        assert f"the model's window is {report.window_tokens}" in str(raised.value)


class TestTheGate:
    async def test_an_async_gate_is_awaited(self, tmp_path: Path) -> None:
        seen: list[str] = []

        async def gate(ctx: RunContext[Any], report: TargetUnreachable) -> None:
            await asyncio.sleep(0)
            seen.append(report.reason)

        settings = _settings(tmp_path)
        agent = _agent(_script([], steps=1), settings, on_unfit=gate)
        await agent.run(
            "carry on",
            message_history=_history(8, pinned="IRREDUCIBLE CONSTRAINT " * 500),
            deps=EljaDeps.from_settings(settings),
        )
        assert seen == ["protected_floor"]

    async def test_a_gate_that_raises_stops_the_run_with_its_own_error(
        self, tmp_path: Path
    ) -> None:
        class TooBigForThisTenantError(Exception):
            pass

        def gate(ctx: RunContext[EljaDeps], report: TargetUnreachable) -> None:
            raise TooBigForThisTenantError(report.reason)

        settings = _settings(tmp_path)
        agent = _agent(_script([]), settings, on_unfit=gate)
        with pytest.raises(TooBigForThisTenantError, match="protected_floor"):
            await agent.run(
                "carry on",
                message_history=_history(8, pinned="IRREDUCIBLE CONSTRAINT " * 500),
                deps=EljaDeps.from_settings(settings),
            )

    @pytest.mark.parametrize(
        ("reason", "phrase"),
        [
            ("protected_floor", "protected content alone is ~1500 tokens"),
            ("no_progress", "a summary this run could not bring it under"),
            ("budget_exhausted", "spent its 3 summarizer requests"),
        ],
    )
    def test_the_description_names_the_cause(self, reason: UnfitReason, phrase: str) -> None:
        report = TargetUnreachable(
            reason=reason,
            target_tokens=1000,
            estimated_tokens=2000,
            protected_tokens=1500,
            summarizer_requests=0,
            max_summarizer_requests=3,
            occurrence=1,
            anchored=False,
            window_tokens=None,
        )
        assert phrase in report.describe()
        assert "over the compaction target of 1000" in report.describe()
        assert "window" not in report.describe()
        assert str(TargetUnreachableError(report)) == report.describe()


class TestTheConveniencePathProceeds:
    async def test_it_shows_one_status_line_per_run_and_sends(self, tmp_path: Path) -> None:
        statuses: list[str] = []
        roles: list[str] = []
        settings = _settings(tmp_path)
        agent = _agent(_script(roles), settings, on_unfit=proceed_over_target)
        deps = EljaDeps.from_settings(settings, on_status=statuses.append)
        await agent.run(
            "carry on",
            message_history=_history(8, pinned="IRREDUCIBLE CONSTRAINT " * 500),
            deps=deps,
        )
        assert roles.count("agent") == 6
        assert len(statuses) == 1
        assert "over the compaction target" in statuses[0]

    async def test_deps_without_a_status_sink_are_fine(self) -> None:
        report = TargetUnreachable(
            reason="no_progress",
            target_tokens=1000,
            estimated_tokens=1200,
            protected_tokens=10,
            summarizer_requests=1,
            max_summarizer_requests=3,
            occurrence=1,
            anchored=False,
            window_tokens=None,
        )
        model = FunctionModel(lambda messages, info: ModelResponse(parts=[]))
        ctx = RunContext[object](deps=object(), model=model, usage=RunUsage())
        await proceed_over_target(ctx, report)  # no status sink to look up; no error

    def test_elja_chat_uses_it(self, tmp_path: Path) -> None:
        from elja.agent import build_agent

        agent = build_agent(_settings(tmp_path))
        (cap,) = [
            c for c in agent.root_capability.capabilities if isinstance(c, BoundedTieredCompaction)
        ]
        assert cap.on_unfit is proceed_over_target

    def test_a_configured_sub_agent_uses_it(self, tmp_path: Path, mocker: MockerFixture) -> None:
        import elja.subagents

        spy = mocker.spy(elja.subagents, "build_compaction")
        elja.subagents._make_delegate(
            _settings(tmp_path), "ops", SubagentConfig(description="d", instructions="i")
        )
        assert spy.call_args.kwargs["on_unfit"] is proceed_over_target


class TestConstruction:
    def test_a_negative_budget_is_refused(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            BoundedTieredCompaction(
                tiers=[ClearToolResults(max_tokens=1)],
                target_tokens=1000,
                max_summarizer_requests=-1,
            )

    def test_a_summarizer_model_name_is_refused(self) -> None:
        """A name builds its own client, outside the guard and outside the bound."""
        with pytest.raises(ValueError, match="not the name 'openai:gpt-4o'"):
            BoundedTieredCompaction(
                tiers=[SummarizingCompaction(model="openai:gpt-4o", max_messages=1)],
                target_tokens=1000,
            )

    def test_focus_keeps_the_bound(self, tmp_path: Path) -> None:
        (cap,) = build_compaction(_settings(tmp_path, budget=2), on_unfit=proceed_over_target)
        assert isinstance(cap, BoundedTieredCompaction)
        focused = cap.with_focus("the ledger")
        assert isinstance(focused, BoundedTieredCompaction)
        assert (focused.max_summarizer_requests, focused.on_unfit) == (2, proceed_over_target)
        assert isinstance(focused, TieredCompaction)


class TestTheManualPath:
    async def test_each_compact_now_call_is_its_own_run(self, tmp_path: Path) -> None:
        roles: list[str] = []
        (cap,) = build_compaction(_settings(tmp_path, budget=1))
        assert isinstance(cap, BoundedTieredCompaction)
        for _ in range(2):
            compacted = await compact_now(cap, _history(8), model=_script(roles))
            assert len(str(compacted)) < len(str(_history(8)))
        assert roles == ["summarizer", "summarizer"]

    async def test_a_composition_with_no_summarizing_tier_reports_no_progress(self) -> None:
        cap = BoundedTieredCompaction(tiers=[ClearToolResults(max_tokens=1)], target_tokens=1000)
        with pytest.raises(TargetUnreachableError) as raised:
            await compact_now(cap, _history(8), model=_script([]))
        report = raised.value.report
        assert report.reason == "no_progress"
        assert report.max_summarizer_requests == 3
        assert 0 < report.protected_tokens < report.estimated_tokens

    async def test_a_summarizing_tier_that_summarizes_nothing_spends_nothing(self) -> None:
        """Its cutoff keeps everything, so no request is made and none is counted."""
        roles: list[str] = []
        cap = BoundedTieredCompaction(
            tiers=[SummarizingCompaction(max_messages=1, keep_messages=100)],
            target_tokens=1000,
        )
        with pytest.raises(TargetUnreachableError) as raised:
            await compact_now(cap, _history(8), model=_script(roles))
        assert roles == []
        assert (raised.value.report.reason, raised.value.report.summarizer_requests) == (
            "no_progress",
            0,
        )

    def test_a_response_last_is_part_of_the_floor(self) -> None:
        history = [*_history(1), ModelResponse(parts=[TextPart(content="the answer")])]
        floor = _protected_floor(history, preserve_first_user_message=False)
        assert floor[-1] is history[-1]


class TestTheHarnessMarkersAreTheHarnesss:
    """elja spells these out rather than importing private modules; drift fails here."""

    def test_the_summary_prefix(self) -> None:
        from pydantic_ai_harness.compaction import _summarizing_compaction

        from elja.compaction import _SUMMARY_PREFIX

        assert _summarizing_compaction._SUMMARY_PREFIX == _SUMMARY_PREFIX

    def test_receipts_and_pins_are_markers_and_user_turns_are_not(self) -> None:
        assert _is_harness_marker(make_receipt_part("[History before this point ...]"))
        assert _is_harness_marker(pin("x"))
        assert not _is_harness_marker(UserPromptPart(content="hello"))
        assert not _is_harness_marker(UserPromptPart(content=[TextContent("hello")]))

    CORPUS: list[list[ModelMessage]] = [
        [ModelRequest(parts=[SystemPromptPart(content="P"), UserPromptPart(content="u")])],
        [
            ModelRequest(parts=[pin("p"), SystemPromptPart(content="P")]),
            ModelRequest(parts=[SystemPromptPart(content="Q")]),
            ModelResponse(parts=[TextPart(content="r")]),
            ModelRequest(parts=[UserPromptPart(content="late")]),
        ],
        [
            ModelRequest(
                parts=[
                    make_receipt_part("[History before this point]"),
                    SystemPromptPart(content="Summary of previous conversation:\n\nold"),
                    SystemPromptPart(content="after"),
                ]
            ),
            ModelRequest(parts=[UserPromptPart(content="u")]),
        ],
        [ModelResponse(parts=[TextPart(content="r")])],
        [ModelRequest(parts=[SystemPromptPart(content="only")])],
    ]

    @pytest.mark.parametrize("messages", CORPUS, ids=[str(i) for i in range(len(CORPUS))])
    def test_leading_system_parts_and_first_user_message_agree(
        self, messages: list[ModelMessage]
    ) -> None:
        from pydantic_ai_harness.compaction._shared import find_first_user_message
        from pydantic_ai_harness.compaction._summarizing_compaction import (
            _extract_system_prompts,
        )

        assert _leading_system_parts(messages) == _extract_system_prompts(messages)
        assert _first_user_message(messages) is find_first_user_message(messages)


def _summary(body: str) -> SystemPromptPart:
    return SystemPromptPart(content=f"Summary of previous conversation:\n\n{body}")


class TestCopyNormalization:
    def test_two_system_parts_that_merely_agree_are_both_kept(self) -> None:
        """Exact copies only: a second part with the same text is not a copy."""
        policy = SystemPromptPart(content="POLICY")
        twin = SystemPromptPart(content="POLICY")  # its own timestamp
        assert policy != twin
        before: list[ModelMessage] = [ModelRequest(parts=[policy, twin])]
        after: list[ModelMessage] = [
            ModelRequest(parts=[policy, _summary("new")]),
            ModelRequest(parts=[policy, twin, UserPromptPart(content="u")]),
        ]
        result = _drop_compaction_copies(before, after)
        kept = [p for m in result for p in m.parts if isinstance(p, SystemPromptPart)]
        assert kept == [policy, after[0].parts[1], twin]

    def test_a_message_of_nothing_but_copies_is_dropped(self) -> None:
        policy = SystemPromptPart(content="POLICY")
        old = _summary("old")
        before: list[ModelMessage] = [ModelRequest(parts=[policy, old])]
        response = ModelResponse(parts=[TextPart(content="r")])
        after: list[ModelMessage] = [
            ModelRequest(parts=[policy, _summary("new")]),
            ModelRequest(parts=[policy, old]),
            response,
        ]
        result = _drop_compaction_copies(before, after)
        assert result == [after[0], response]

    def test_nothing_changes_when_no_summary_was_written(self) -> None:
        policy = SystemPromptPart(content="POLICY")
        after: list[ModelMessage] = [ModelRequest(parts=[policy]), ModelRequest(parts=[policy])]
        assert _drop_compaction_copies(after, after) is after

    async def test_several_older_summaries_are_folded_into_one_anchor(
        self, tmp_path: Path
    ) -> None:
        """A history written before the fix carries one summary per compacting turn.

        The harness would anchor on the oldest and skip the rest, so their content
        would be lost the moment copies are dropped. Folded, the summarizer is handed
        all of them, and what comes back has one summary.
        """
        anchors: list[str] = []

        def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if _is_summarizer(info):
                anchors.append(str(messages).split("<previous-summary>", 1)[1])
                return ModelResponse(parts=[TextPart(content="## Intent\nfolded")])
            return ModelResponse(parts=[TextPart(content="ok")])

        history = _history(8)
        history[0] = ModelRequest(
            parts=[
                SystemPromptPart(content="POLICY"),
                _summary("NEWER facts"),
                SystemPromptPart(content="POLICY"),
                _summary("OLDER facts"),
                UserPromptPart(content="the task"),
            ]
        )
        settings = _settings(tmp_path)
        agent = _agent(FunctionModel(script), settings)
        result = await agent.run(
            "carry on", message_history=history, deps=EljaDeps.from_settings(settings)
        )
        (anchor,) = anchors
        assert "NEWER facts" in anchor and "OLDER facts" in anchor
        assert anchor.index("NEWER") < anchor.index("OLDER")
        summaries = [
            p
            for m in result.all_messages()
            for p in m.parts
            if isinstance(p, SystemPromptPart) and p.content.startswith("Summary of")
        ]
        assert len(summaries) == 1
        assert "folded" in summaries[0].content


class TestTheMeteringWrapper:
    """The two dispatch paths the summary agent does not use today, pinned directly."""

    async def test_a_streamed_request_spends_from_the_same_budget(self) -> None:
        async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
            yield "summary"

        state = _RunState()
        metered = _MeteredSummarizer(FunctionModel(stream_function=stream), state, limit=1)
        parameters = ModelRequestParameters()
        messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content="x")])]
        async with metered.request_stream(messages, None, parameters) as response:
            async for _ in response:
                pass
        assert state.spent == 1
        with pytest.raises(_SummarizerBudgetSpentError):
            async with metered.request_stream(messages, None, parameters):
                pass  # pragma: no cover
        assert state.spent == 1

    async def test_a_count_is_refused_once_spent_and_never_spends(self) -> None:
        class Counting(FunctionModel):
            async def count_tokens(
                self,
                messages: list[ModelMessage],
                model_settings: ModelSettings | None,
                model_request_parameters: ModelRequestParameters,
            ) -> RequestUsage:
                return RequestUsage(input_tokens=11)

        state = _RunState()
        metered = _MeteredSummarizer(
            Counting(lambda messages, info: ModelResponse(parts=[])), state, limit=1
        )
        parameters = ModelRequestParameters()
        assert (await metered.count_tokens([], None, parameters)).input_tokens == 11
        assert state.spent == 0
        state.spent = 1
        with pytest.raises(_SummarizerBudgetSpentError):
            await metered.count_tokens([], None, parameters)

    def test_settings_and_identity_are_the_wrapped_models(self) -> None:
        inner = FunctionModel(
            lambda messages, info: ModelResponse(parts=[]), settings={"max_tokens": 77}
        )
        metered = _MeteredSummarizer(inner, _RunState(), limit=1)
        assert metered.settings == {"max_tokens": 77}
        assert metered.model_name == inner.model_name
        assert metered.system == inner.system
