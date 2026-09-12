"""E5/E6 contract tests: the server example, and what a host may rely on.

These drive ``examples/server_agent.py`` rather than a parallel fixture, so the
documented integration cannot rot. No new elja API was needed for any of it —
the request permits exactly that, and says to ship the contract tests instead.
"""

import asyncio
from collections.abc import AsyncIterator, Callable, Sequence
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import BaseModel
from pydantic_ai import (
    Agent,
    CancellationToken,
    DeferredToolRequests,
    DeferredToolResults,
    ModelRetry,
    RunContext,
)
from pydantic_ai.exceptions import (
    CallDeferred,
    RunCancelled,
    UsageLimitExceeded,
    UserError,
)
from pydantic_ai.messages import (
    INTERRUPTED_TOOL_RETURN_CONTENT,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import (
    AgentInfo,
    DeltaThinkingCalls,
    DeltaThinkingPart,
    DeltaToolCall,
    DeltaToolCalls,
    FunctionModel,
)
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.usage import RunUsage, UsageLimits
from pytest_mock import MockerFixture

from elja.application import build_application_agent
from elja.settings import CompactionConfig, EljaSettings, WorkspaceConfig
from examples.server_agent import (
    CLEARED,
    HostDeps,
    TurnRecorder,
    _checkpoint,
    build_host_agent,
    deserialize,
    foreign_thinking_parts,
    household_toolset,
    run_turn,
    run_with_deadline,
    serialize,
)

StreamItem = str | DeltaToolCalls | DeltaThinkingCalls


def _deps() -> HostDeps:
    return HostDeps(tenant="acme", store={})


def _saving_then_answering() -> FunctionModel:
    """Calls one tool, narrates, then answers — one tool-bearing turn, one final."""
    turns: list[int] = []

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[StreamItem]:
        turns.append(1)
        if len(turns) == 1:
            yield "let me write that down. "
            yield {
                1: DeltaToolCall(name="save_fact", json_args='{"key": "vet", "value": "tuesday"}')
            }
        else:
            yield "saved the vet appointment"

    return FunctionModel(stream_function=stream)


def _thinks_chunks_and_reads_back() -> FunctionModel:
    """Thinks, narrates in two chunks, writes, reads back, then answers in two chunks.

    Every text part arrives as more than one chunk on purpose. A script that
    yields whole strings only ever fires ``PartStartEvent``, which left
    ``run_turn``'s delta branch — and its thinking branch, and the example's
    ``read_fact`` tool — unexecuted by the whole suite.
    """
    turns: list[int] = []

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[StreamItem]:
        turns.append(1)
        if len(turns) == 1:
            yield {0: DeltaThinkingPart(content="the vet first")}
            yield "let me "
            yield "write that down. "
            yield {
                1: DeltaToolCall(name="save_fact", json_args='{"key": "vet", "value": "tuesday"}')
            }
        elif len(turns) == 2:
            yield {1: DeltaToolCall(name="read_fact", json_args='{"key": "vet"}')}
        else:
            yield "the vet is "
            yield "tuesday"

    return FunctionModel(stream_function=stream)


def _writes_only_once() -> FunctionModel:
    """Saves unless the history already shows a ``save_fact`` result.

    Stands in for the behaviour that makes the checkpoint matter: a model reading
    a history with no trace of a completed write does the write again.
    """

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[StreamItem]:
        already_saved = any(
            isinstance(part, ToolReturnPart) and part.tool_name == "save_fact"
            for message in messages
            for part in message.parts
        )
        if already_saved:
            yield "already saved"
        else:
            yield {
                1: DeltaToolCall(name="save_fact", json_args='{"key": "vet", "value": "tuesday"}')
            }

    return FunctionModel(stream_function=stream)


class _PricedModel(FunctionModel):
    """A FunctionModel that reports a real model id, so its window resolves.

    ``FunctionModel`` reports ``function:<name>``, which no pricing entry matches,
    so every reading taken through one is unresolved. Overriding the id is the
    only way to reach the other direction without a provider.
    """

    @property
    def model_id(self) -> str:
        return "openai:gpt-4o"


class TestAnEventConsumerCanTellTheTurnApart:
    """E5 acceptance, clause by clause."""

    async def test_intermediate_narration_is_not_folded_into_the_answer(self) -> None:
        shown: list[str] = []
        recorder = TurnRecorder()
        agent = build_host_agent(_saving_then_answering())
        output, messages = await run_turn(
            agent, _deps(), "remember the vet", recorder=recorder, display=shown.append
        )
        # The narration was displayed...
        assert "let me write that down. " in "".join(shown)
        # ...and is not part of the final answer.
        assert output == "saved the vet appointment"
        assert "let me write that down" not in str(output)

    async def test_the_answer_comes_only_from_the_run_result_event(self) -> None:
        """FinalResultEvent is not the discriminator, which is easy to get wrong.

        It fires whenever a part that COULD be the final output starts, so a turn
        that narrates and then calls a tool emits it too. Measured: twice in this
        one turn. A host that treats the first one as "the answer has begun" and
        accumulates from there ships the narration as the answer.
        """
        recorder = TurnRecorder()
        agent = build_host_agent(_saving_then_answering())
        output, _ = await run_turn(agent, _deps(), "remember the vet", recorder=recorder)
        kinds = recorder.kinds()
        assert kinds.count("output_part_started") == 2
        # The first one precedes the tool call, i.e. it was narration.
        assert kinds.index("output_part_started") < kinds.index("tool_call")
        # And the answer, taken from the run result, carries none of it.
        assert output == "saved the vet appointment"

    async def test_a_tool_bearing_turn_is_distinguishable_from_the_final_one(
        self,
    ) -> None:
        """What a host CAN rely on: the tool events, in order."""
        recorder = TurnRecorder()
        agent = build_host_agent(_saving_then_answering())
        await run_turn(agent, _deps(), "remember the vet", recorder=recorder)
        kinds = recorder.kinds()
        assert "tool_call" in kinds
        assert "tool_result" in kinds
        # The tool work completes before the last output part begins.
        assert kinds.index("tool_result") < len(kinds) - kinds[::-1].index("output_part_started")

    async def test_the_recorded_detail_names_the_tool(self) -> None:
        """The log is only useful if it says WHICH tool, so pin kind and detail.

        Asserting kinds alone left the ``detail`` argument unconstrained at every
        call site: three separate mutants that stopped passing a tool name
        survived the whole suite.
        """
        recorder = TurnRecorder()
        agent = build_host_agent(_saving_then_answering())
        await run_turn(agent, _deps(), "remember the vet", recorder=recorder)
        assert [(event.kind, event.detail) for event in recorder.events] == [
            ("output_part_started", ""),
            ("tool_call_started", "save_fact"),
            ("tool_call", "save_fact"),
            ("tool_result", "save_fact"),
            ("output_part_started", ""),
            ("turn_finished", ""),
        ]

    async def test_chunked_text_and_thinking_both_reach_the_host(self) -> None:
        """Deltas are the normal case for a real provider, not an edge one.

        A part that arrives in one piece fires only ``PartStartEvent``, so a suite
        built entirely from whole-string scripts never runs the delta branch at
        all — and a host wiring a UI to it would ship a renderer nothing had
        exercised. This script chunks every text part and opens with a thinking
        block.
        """
        shown: list[str] = []
        recorder = TurnRecorder()
        deps = _deps()
        agent = build_host_agent(_thinks_chunks_and_reads_back())
        output, _ = await run_turn(
            agent, deps, "remember the vet", recorder=recorder, display=shown.append
        )
        assert output == "the vet is tuesday"
        # First chunk of each part comes from PartStartEvent, the rest from
        # PartDeltaEvent: both branches ran, and in order.
        assert shown == ["let me ", "write that down. ", "the vet is ", "tuesday"]
        pairs = [(event.kind, event.detail) for event in recorder.events]
        assert ("thinking", "") in pairs
        # And the example's read tool ran, against what the write left behind.
        assert ("tool_result", "read_fact") in pairs
        assert deps.store == {"acme:vet": "tuesday"}

    async def test_the_display_sees_the_answer_as_well_as_the_narration(self) -> None:
        """Not "narration only" — nothing in the stream can tell them apart yet.

        The answer's text part starts and streams exactly like mid-run narration,
        so a host rendering every chunk renders both. That is precisely why the
        answer is ALSO returned: the return value is the copy to trust, and a host
        that reconstructs it from the chunks it displayed gets the narration too.
        """
        shown: list[str] = []
        agent = build_host_agent(_saving_then_answering())
        output, _ = await run_turn(
            agent, _deps(), "remember the vet", recorder=TurnRecorder(), display=shown.append
        )
        assert shown == ["let me write that down. ", "saved the vet appointment"]
        assert output == "saved the vet appointment"
        assert shown[-1] == output

    async def test_an_empty_chunk_never_reaches_the_display(self) -> None:
        """A zero-length delta is real, and forwarding it makes a UI emit blanks.

        pydantic-ai's parts manager yields a ``PartDeltaEvent`` for an empty
        content delta rather than dropping it, so the guard in the example is the
        only thing standing between a provider's padding chunk and the host's UI.
        """

        async def stream(
            messages: list[ModelMessage], info: AgentInfo
        ) -> AsyncIterator[StreamItem]:
            yield "a"
            yield ""
            yield "b"

        agent: Agent[HostDeps, str | DeferredToolRequests] = build_application_agent(
            FunctionModel(stream_function=stream),
            deps_type=HostDeps,
            output_type=[str, DeferredToolRequests],
        )
        shown: list[str] = []
        await run_turn(agent, _deps(), "hi", recorder=TurnRecorder(), display=shown.append)
        assert shown == ["a", "b"]

    async def test_the_host_owns_contiguous_sequence_numbers(self) -> None:
        """Nothing in the framework promises a durable id, so the host assigns one."""
        recorder = TurnRecorder()
        agent = build_host_agent(_saving_then_answering())
        await run_turn(agent, _deps(), "remember the vet", recorder=recorder)
        assert [event.seq for event in recorder.events] == list(range(1, len(recorder.events) + 1))

    async def test_usage_is_recorded(self) -> None:
        recorder = TurnRecorder()
        agent = build_host_agent(_saving_then_answering())
        await run_turn(agent, _deps(), "remember the vet", recorder=recorder)
        assert recorder.usage is not None
        assert recorder.usage.requests == 2

    async def test_a_tool_error_is_recorded_as_one(self) -> None:
        """There is no separate error event; the retry prompt is the signal."""
        toolset: FunctionToolset[HostDeps] = FunctionToolset()

        @toolset.tool
        def explode(ctx: RunContext[HostDeps]) -> str:
            """Always asks the model to try something else."""
            raise ModelRetry("no such record")

        turns: list[int] = []

        async def stream(
            messages: list[ModelMessage], info: AgentInfo
        ) -> AsyncIterator[StreamItem]:
            turns.append(1)
            if len(turns) == 1:
                yield {1: DeltaToolCall(name="explode", json_args="{}")}
            else:
                yield "gave up on that"

        agent: Agent[HostDeps, str | DeferredToolRequests] = build_application_agent(
            FunctionModel(stream_function=stream),
            deps_type=HostDeps,
            output_type=[str, DeferredToolRequests],
            toolsets=[toolset],
        )
        recorder = TurnRecorder()
        output, _ = await run_turn(agent, _deps(), "look it up", recorder=recorder)
        assert output == "gave up on that"
        assert "tool_error" in recorder.kinds()
        assert "tool_result" not in recorder.kinds()

    async def test_a_terminal_error_reaches_the_caller(self) -> None:
        recorder = TurnRecorder()
        agent = build_host_agent(_saving_then_answering())
        with pytest.raises(UsageLimitExceeded):
            await run_turn(
                agent,
                _deps(),
                "remember the vet",
                recorder=recorder,
                usage_limits=UsageLimits(request_limit=1),
            )

    async def test_a_broken_display_does_not_abort_the_turn(self) -> None:
        """But the recorder is NOT isolated: a host that cannot record must stop."""
        recorder = TurnRecorder()

        def broken(_text: str) -> None:
            raise RuntimeError("subscriber went away")

        agent = build_host_agent(_saving_then_answering())
        output, _ = await run_turn(
            agent, _deps(), "remember the vet", recorder=recorder, display=broken
        )
        assert output == "saved the vet appointment"
        assert "turn_finished" in recorder.kinds()

    @pytest.mark.parametrize(
        "failing_kind", ["tool_call", "tool_result", "output_part_started", "turn_finished"]
    )
    async def test_a_failing_recorder_is_not_swallowed(self, failing_kind: str) -> None:
        """Accounting is not display; its failure must surface.

        Parametrized per event kind, because suppressing one branch while the
        others still raise would otherwise look identical.
        """

        class Hostile(TurnRecorder):
            def record(self, kind: str, detail: str = "") -> None:
                if kind == failing_kind:
                    raise RuntimeError(f"cannot persist {kind}")
                super().record(kind, detail)

        agent = build_host_agent(_saving_then_answering())
        with pytest.raises(RuntimeError, match=f"cannot persist {failing_kind}"):
            await run_turn(agent, _deps(), "remember the vet", recorder=Hostile())

    async def test_a_display_can_stop_and_reconnect_without_corrupting_the_agent(
        self,
    ) -> None:
        """Two turns on one agent, the subscriber dropping in between."""
        agent = build_host_agent(_saving_then_answering())
        deps = _deps()
        first = TurnRecorder()
        await run_turn(agent, deps, "one", recorder=first, display=None)
        second: list[str] = []
        recorder = TurnRecorder()
        output, _ = await run_turn(agent, deps, "two", recorder=recorder, display=second.append)
        assert output == "saved the vet appointment"
        assert second, "the reconnected subscriber received nothing"


class TestContextReporting:
    async def test_context_is_reported_per_request_not_per_turn(self) -> None:
        readings: list[Any] = []
        agent = build_host_agent(_saving_then_answering(), on_context=readings.append)
        recorder = TurnRecorder()
        await run_turn(agent, _deps(), "remember the vet", recorder=recorder)
        # Two model requests in this turn, so two readings — independent of the
        # turn completing.
        assert len(readings) == 2

    def test_the_context_reporter_is_attached_last(self) -> None:
        """The documented rule, on the example's own wiring.

        Placed earlier it measures a request that was never sent — 7239 tokens
        versus 1177 for the same turn, measured in
        tests/test_compaction_composition.py. Counting readings does not catch a
        reporter in the wrong position, so assert the position.
        """
        settings = EljaSettings(compaction=CompactionConfig(target_tokens=2000))
        agent = build_host_agent(
            _saving_then_answering(), settings=settings, on_context=lambda _usage: None
        )
        capabilities = [
            capability
            for capability in agent.root_capability.capabilities
            if type(capability).__name__ in {"TieredCompaction", "ReportContextUsage"}
        ]
        assert [type(c).__name__ for c in capabilities] == [
            "TieredCompaction",
            "ReportContextUsage",
        ]

    async def test_the_reading_carries_its_reliability_metadata(self) -> None:
        """It is an estimate against a possibly-fallback window, and says so."""
        readings: list[Any] = []
        agent = build_host_agent(_saving_then_answering(), on_context=readings.append)
        await run_turn(agent, _deps(), "remember the vet", recorder=TurnRecorder())
        usage = readings[0]
        assert usage.used_tokens > 0
        assert usage.window_tokens > 0
        # `resolved` is False when the window came from the fallback rather than
        # from the provider, which is exactly when the fraction is least exact.
        assert usage.resolved is False

    async def test_a_resolved_window_says_so(self) -> None:
        """The other direction, or `resolved` could be hardcoded False.

        A reading whose window came from the pricing registry is exact, and the
        flag is how a host knows which of the two it has.
        """

        async def stream(
            messages: list[ModelMessage], info: AgentInfo
        ) -> AsyncIterator[StreamItem]:
            yield "noted"

        readings: list[Any] = []
        agent = build_host_agent(_PricedModel(stream_function=stream), on_context=readings.append)
        await run_turn(agent, _deps(), "remember the vet", recorder=TurnRecorder())
        assert readings[0].resolved is True
        assert readings[0].window_tokens > 0

    async def test_a_reading_is_taken_before_the_limit_is_checked(self) -> None:
        """Readings outnumber requests on a limit trip, so they are not a meter.

        `ReportContextUsage` measures in `before_model_request`, which runs
        upstream of the usage-limit check — so the request that trips the limit is
        measured and then never sent. Measured: 2 readings against 1 request. A
        host billing or rate-limiting on reading count over-charges by one for
        every turn that ends this way.
        """
        readings: list[Any] = []
        recorder = TurnRecorder()
        agent = build_host_agent(_saving_then_answering(), on_context=readings.append)
        with pytest.raises(UsageLimitExceeded):
            await run_turn(
                agent,
                _deps(),
                "remember the vet",
                recorder=recorder,
                usage_limits=UsageLimits(request_limit=1),
            )
        assert len(readings) == 2
        assert recorder.usage is not None
        assert recorder.usage.requests == 1


class TestAskingTheHumanIsATypedOutcome:
    """E6: a typed ask-the-user outcome, with no extra model call."""

    def _agent_that_asks(self) -> tuple[Agent[HostDeps, Any], list[int]]:
        """The example's own agent: ask_user is part of its toolset."""
        turns: list[int] = []

        def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            turns.append(1)
            if len(turns) == 1:
                return ModelResponse(
                    parts=[ToolCallPart(tool_name="ask_user", args={"question": "which vet?"})]
                )
            return ModelResponse(parts=[TextPart(content="booked with Dr. Meow")])

        return build_host_agent(FunctionModel(script)), turns

    async def test_the_run_ends_with_a_request_not_an_error(self) -> None:
        agent, turns = self._agent_that_asks()
        result = await agent.run("book the vet", deps=_deps())
        assert isinstance(result.output, DeferredToolRequests)
        assert [call.tool_name for call in result.output.calls] == ["ask_user"]
        # One model call: asking cost no extra round trip, and was not a retry.
        assert len(turns) == 1

    async def test_the_answer_resumes_the_same_conversation(self) -> None:
        agent, turns = self._agent_that_asks()
        asked = await agent.run("book the vet", deps=_deps())
        assert isinstance(asked.output, DeferredToolRequests)
        resumed = await agent.run(
            message_history=asked.all_messages(),
            deferred_tool_results=DeferredToolResults(
                calls={asked.output.calls[0].tool_call_id: "Dr. Meow"}
            ),
            deps=_deps(),
        )
        assert resumed.output == "booked with Dr. Meow"
        assert len(turns) == 2

    async def test_the_loop_closes_through_the_hosts_own_turn_function(self) -> None:
        """The ask and the answer both go through `run_turn`, not around it.

        Without `deferred_tool_results` on the turn function, a host could open the
        ask with its own streaming path but had to close it with a bare
        `agent.run` — losing the recorder, the display and the checkpoint for
        exactly the turn that resumes real work.
        """
        turns: list[int] = []

        async def stream(
            messages: list[ModelMessage], info: AgentInfo
        ) -> AsyncIterator[StreamItem]:
            turns.append(1)
            if len(turns) == 1:
                yield {1: DeltaToolCall(name="ask_user", json_args='{"question": "which vet?"}')}
            else:
                yield "booked with Dr. Meow"

        agent = build_host_agent(FunctionModel(stream_function=stream))
        deps = _deps()
        asked_recorder = TurnRecorder()
        asked, history = await run_turn(agent, deps, "book the vet", recorder=asked_recorder)
        assert isinstance(asked, DeferredToolRequests)

        answered = TurnRecorder()
        output, _ = await run_turn(
            agent,
            deps,
            history=history,
            recorder=answered,
            deferred_tool_results=DeferredToolResults(
                calls={asked.calls[0].tool_call_id: "Dr. Meow"}
            ),
        )
        assert output == "booked with Dr. Meow"
        assert len(turns) == 2
        # And the resuming turn was recorded like any other.
        assert "turn_finished" in answered.kinds()


class TestHostOwnedPersistence:
    async def test_history_round_trips_through_the_native_adapter(self, tmp_path: Path) -> None:
        recorder = TurnRecorder()
        agent = build_host_agent(_saving_then_answering())
        deps = _deps()
        _, messages = await run_turn(agent, deps, "remember the vet", recorder=recorder)
        path = tmp_path / "turn.json"
        path.write_bytes(serialize(messages))
        loaded = deserialize(path.read_bytes())
        assert [type(m).__name__ for m in loaded] == [type(m).__name__ for m in messages]
        # And it is usable as history for the next turn.
        output, resumed = await run_turn(
            agent, deps, "and the dentist", history=loaded, recorder=TurnRecorder()
        )
        assert output == "saved the vet appointment"
        # What comes back is the WHOLE conversation, not just this turn's share of
        # it: the first turn's prompt is still in there. A host that persists
        # `new_messages()` instead drops every earlier turn on each save, and the
        # assertion above cannot see the difference.
        assert any(
            isinstance(part, UserPromptPart) and part.content == "remember the vet"
            for message in resumed
            for part in message.parts
        )

    def test_opaque_provider_state_survives_the_round_trip(self) -> None:
        """A thinking signature is provider-specific and must not be dropped."""
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content="think about it")]),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content="weighing options",
                        signature="opaque-provider-blob",
                        provider_name="anthropic",
                    ),
                    TextPart(content="done"),
                ]
            ),
        ]
        restored = deserialize(serialize(messages))
        thinking = [
            part
            for message in restored
            for part in getattr(message, "parts", [])
            if isinstance(part, ThinkingPart)
        ]
        assert len(thinking) == 1
        assert thinking[0].signature == "opaque-provider-blob"
        assert thinking[0].provider_name == "anthropic"

    def test_foreign_reasoning_is_listed_whether_or_not_it_has_a_signature(self) -> None:
        """E6: handle a provider switch explicitly rather than replaying blindly.

        The signature is not the part that crosses. pydantic-ai refuses to replay a
        foreign *signature*, but OpenAI's Responses mapping sends a foreign
        thinking part's *content* out as an ordinary assistant message wrapped in
        the profile's thinking tags — so the reasoning arrives at the new provider
        either way. A filter keyed on the signature hides exactly the parts that
        get replayed, which is the inversion this test exists to prevent.
        """
        model = FunctionModel(lambda m, i: None)  # type: ignore[arg-type,return-value]
        messages: list[ModelMessage] = [
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content="a", signature="anthropic-blob", provider_name="anthropic"
                    ),
                    ThinkingPart(content="b", signature="native-blob", provider_name=model.system),
                    # No signature — and replayed as tagged text all the same.
                    ThinkingPart(content="c", provider_name="google"),
                    TextPart(content="done"),
                ]
            )
        ]
        assert [part.content for part in foreign_thinking_parts(messages, model)] == ["a", "c"]
        # And an empty history, or one with nothing foreign, reports nothing.
        assert foreign_thinking_parts([], model) == []

    def test_a_part_with_no_provider_inherits_its_messages(self) -> None:
        """Which is what the provider itself does when it decides ownership.

        OpenAI's Responses mapping treats a part carrying no ``provider_name`` as
        its own when the enclosing message is its own, so comparing the part alone
        calls a native part foreign and sends the host chasing a switch that never
        happened.
        """
        model = FunctionModel(lambda m, i: None)  # type: ignore[arg-type,return-value]
        native = ModelResponse(parts=[ThinkingPart(content="mine")], provider_name=model.system)
        other = ModelResponse(parts=[ThinkingPart(content="theirs")], provider_name="anthropic")
        assert foreign_thinking_parts([native], model) == []
        assert [part.content for part in foreign_thinking_parts([other], model)] == ["theirs"]

    def test_a_part_that_names_its_provider_keeps_it(self) -> None:
        """The precedence is one-way, and reversing it survived the suite.

        Every other case here has the part silent and the message named, or the other
        way round, so both orders agree. They disagree only when the two name
        *different* providers — and then the part is what pydantic-ai consults first.
        Reversing them calls a native part foreign whenever the enclosing message came
        from somewhere else, which is exactly the mid-history provider switch this
        helper exists to describe.
        """
        model = FunctionModel(lambda m, i: None)  # type: ignore[arg-type,return-value]
        mixed = ModelResponse(
            parts=[
                ThinkingPart(content="mine", provider_name=model.system),
                ThinkingPart(content="theirs", provider_name="anthropic"),
            ],
            provider_name="anthropic",
        )
        assert [part.content for part in foreign_thinking_parts([mixed], model)] == ["theirs"]

    def test_a_request_carries_no_reasoning_to_weigh(self) -> None:
        """Requests have parts too, and `ThinkingPart` is not among the types they can hold.

        So the `ModelResponse` filter is a type narrowing for `message.parts`, not a
        guard against anything reachable — removing it changes no behaviour. Asserted
        here as documentation of that, rather than left to look like a pinned guard.
        """
        model = FunctionModel(lambda m, i: None)  # type: ignore[arg-type,return-value]
        request = ModelRequest(parts=[UserPromptPart(content="think about it")])
        assert foreign_thinking_parts([request], model) == []
        assert ThinkingPart not in getattr(ModelRequestPart, "__args__", ())

    async def test_nothing_is_written_unless_the_host_writes_it(self, tmp_path: Path) -> None:
        before = sorted(p.name for p in tmp_path.iterdir())
        agent = build_host_agent(_saving_then_answering())
        await run_turn(agent, _deps(), "remember the vet", recorder=TurnRecorder())
        assert sorted(p.name for p in tmp_path.iterdir()) == before
        assert not (tmp_path / ".elja").exists()


class Reminder(BaseModel):
    """An extraction-style structured output."""

    what: str
    when: str


class TestStructuredExtraction:
    async def test_a_structured_output_type_is_honored_on_this_path(self) -> None:
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
            FunctionModel(script), deps_type=HostDeps, output_type=Reminder
        )
        result = await agent.run("when is the vet", deps=_deps())
        assert result.output == Reminder(what="vet", when="tuesday")


def _writes_then_stalls() -> FunctionModel:
    """Writes on the first turn, then hangs — a turn that dies mid-flight."""
    turns: list[int] = []

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[StreamItem]:
        turns.append(1)
        if len(turns) == 1:
            yield {
                1: DeltaToolCall(name="save_fact", json_args='{"key": "vet", "value": "tuesday"}')
            }
        else:
            await asyncio.sleep(_STALL)
            yield "never reached"  # pragma: no cover - the deadline always wins

    return FunctionModel(stream_function=stream)


# Long enough that a stall cannot finish on its own, so a test that claims the
# deadline did the stopping cannot pass because the model happened to return.
_STALL = 30.0
# The wall-clock ceiling for a cancellation that is supposed to be immediate. A
# test for a kill has to bound its own time, or a kill that never happened reads
# as a slow pass.
_CANCEL_CEILING = 5.0


class TestTheCheckpointOutlivesAFailedTurn:
    """E6: completed work is available even when later work fails.

    The request asks for "completed messages/tool outcomes at host-selected
    checkpoints, so the host can durably save input and effects even if later work
    fails". The effect is the whole problem: a tool that already wrote has changed
    the host's world, and a history with no trace of it makes the next turn write
    again.
    """

    async def test_a_tool_that_already_wrote_is_in_the_checkpoint(self) -> None:
        recorder = TurnRecorder()
        deps = _deps()
        agent = build_host_agent(_saving_then_answering())
        with pytest.raises(UsageLimitExceeded):
            await run_turn(
                agent,
                deps,
                "remember the vet",
                recorder=recorder,
                usage_limits=UsageLimits(request_limit=1),
            )
        # The write landed: there is an effect in the host's world to reconcile.
        assert deps.store == {"acme:vet": "tuesday"}
        returns = [
            part
            for message in recorder.partial
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        assert [part.tool_name for part in returns] == ["save_fact"]
        # And the usage spent getting there, which the host is billed for.
        assert recorder.usage is not None
        assert recorder.usage.requests == 1

    async def test_the_checkpoint_resumes_without_repeating_the_write(self) -> None:
        """The consequence, in both directions.

        Resuming from the checkpoint, the model sees its own completed write and
        does not repeat it. Resuming from nothing — which is what a host got before
        the checkpoint existed — it writes a second time.
        """
        deps = _deps()
        agent = build_host_agent(_writes_only_once())
        died = TurnRecorder()
        with pytest.raises(UsageLimitExceeded):
            await run_turn(
                agent,
                deps,
                "remember the vet",
                recorder=died,
                usage_limits=UsageLimits(request_limit=1),
            )
        assert deps.store == {"acme:vet": "tuesday"}

        resumed = TurnRecorder()
        output, _ = await run_turn(agent, deps, history=died.partial, recorder=resumed)
        assert output == "already saved"
        assert ("tool_call", "save_fact") not in [
            (event.kind, event.detail) for event in resumed.events
        ]

        # Without the checkpoint there is no history to resume from, and the write
        # happens twice. This is the defect the checkpoint closes.
        duplicate = TurnRecorder()
        await run_turn(agent, deps, "remember the vet", recorder=duplicate)
        assert ("tool_call", "save_fact") in [
            (event.kind, event.detail) for event in duplicate.events
        ]

    async def test_a_deadline_leaves_the_same_checkpoint(self) -> None:
        """A stop is just another way for a turn to die; the checkpoint is one place."""
        deps = _deps()
        recorder = TurnRecorder()
        agent = build_host_agent(_writes_then_stalls())
        async with asyncio.timeout(_CANCEL_CEILING):
            with pytest.raises(RunCancelled):
                await run_with_deadline(
                    agent, deps, "remember the vet", recorder=recorder, seconds=0.05
                )
        assert deps.store == {"acme:vet": "tuesday"}
        returns = [
            part
            for message in recorder.partial
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        assert [part.tool_name for part in returns] == ["save_fact"]

    async def test_a_turn_stopped_before_any_work_leaves_an_empty_checkpoint(self) -> None:
        """Nothing completed, so there is nothing to reconcile — and no spend either.

        An already-cancelled token still binds the run before refusing it, so the
        checkpoint is reachable and simply empty. A host can persist it blindly.
        """
        token = CancellationToken()
        token.cancel()
        recorder = TurnRecorder()
        deps = _deps()
        agent = build_host_agent(_saving_then_answering())
        with pytest.raises(RunCancelled):
            await run_turn(
                agent, deps, "remember the vet", recorder=recorder, cancellation_token=token
            )
        assert recorder.partial == []
        assert deps.store == {}
        assert recorder.usage is not None
        assert recorder.usage.requests == 0

    def test_the_checkpoint_never_replaces_the_hosts_own_failure(self) -> None:
        """Both accessors raise ``UserError`` until the first iteration binds the run.

        That window is real rather than theoretical: `_ensure_started` creates the
        background task but does not await it, so for the few event-loop steps before
        the binding lands, an external cancellation reaches the checkpoint unbound.
        Nothing completed in that window, so the checkpoint is correctly empty — but
        the guard is what stops a framework complaint about iteration order replacing
        the cancellation the host actually has to see.

        Driven directly because the window is a handful of loop steps wide and pinning
        it through the public surface would be timing-dependent; the *consequence* of
        losing the guard is what this asserts.
        """
        recorder = TurnRecorder()

        class _Unbound:
            def all_messages(self) -> list[ModelMessage]:
                raise UserError("The run has not started; iterate the events first.")

            @property
            def usage(self) -> RunUsage:
                raise UserError("The run has not started; iterate the events first.")

        _checkpoint(recorder, cast(Any, _Unbound()))
        assert recorder.partial == []
        assert recorder.usage is None


class TestCancellation:
    async def test_a_deadline_arrives_as_a_catchable_outcome(self) -> None:
        """Not a ``TimeoutError`` the host has to re-raise and unpack.

        A first-party cancellation is an ordinary application outcome: the host
        catches it, keeps the checkpoint, and answers the request. An external
        asyncio cancellation would have to keep propagating for the enclosing scope
        to unwind, leaving the state reachable only through the exception chain.
        """

        async def stream(
            messages: list[ModelMessage], info: AgentInfo
        ) -> AsyncIterator[StreamItem]:
            await asyncio.sleep(_STALL)
            yield "never reached"  # pragma: no cover - the deadline always wins

        agent: Agent[HostDeps, str | DeferredToolRequests] = build_application_agent(
            FunctionModel(stream_function=stream),
            deps_type=HostDeps,
            output_type=[str, DeferredToolRequests],
        )
        async with asyncio.timeout(_CANCEL_CEILING):
            with pytest.raises(RunCancelled):
                await run_with_deadline(
                    agent, _deps(), "take your time", recorder=TurnRecorder(), seconds=0.05
                )

    async def test_the_cancelled_run_carries_its_own_resumable_history(self) -> None:
        """``RunCancelled`` is not only a signal; it carries the run's history.

        What it does NOT carry is a *resumable* history — see
        `TestACancelledRunsOwnHistoryIsNotResumable`, which measures a tool stopped
        inside its own body leaving its call dangling in this very snapshot. This
        fixture stalls the model stream on a later turn, so `save_fact` completed and
        its return is a real success; all this asserts is that the completed work is
        present on the exception, which is the claim the assertion can support.
        """
        deps = _deps()
        agent = build_host_agent(_writes_then_stalls())
        async with asyncio.timeout(_CANCEL_CEILING):
            with pytest.raises(RunCancelled) as caught:
                await run_with_deadline(
                    agent, deps, "remember the vet", recorder=TurnRecorder(), seconds=0.05
                )
        assert any(
            isinstance(part, ToolReturnPart) and part.tool_name == "save_fact"
            for message in caught.value.all_messages()
            for part in message.parts
        )

    async def test_a_deadline_that_does_not_fire_leaves_the_turn_alone(self) -> None:
        deps = _deps()
        agent = build_host_agent(_saving_then_answering())
        output, _ = await run_with_deadline(
            agent, deps, "remember the vet", recorder=TurnRecorder(), seconds=_CANCEL_CEILING
        )
        assert output == "saved the vet appointment"
        assert deps.store == {"acme:vet": "tuesday"}

    async def test_the_timer_is_cancelled_rather_than_left_to_fire(
        self, mocker: MockerFixture
    ) -> None:
        """The `finally` is retention, not correctness — and was unasserted.

        A turn that finishes before its deadline leaves the timer armed without this:
        one live `TimerHandle` and one `CancellationToken` retained for the whole
        budget. Harmless per turn, and on a server with a long deadline it accumulates.
        The previous version of this test claimed to pin it and could not — the timer
        simply fired later with nothing registered, which is invisible.
        """
        loop = asyncio.get_running_loop()
        real_call_later = loop.call_later
        handles: list[asyncio.TimerHandle] = []

        def spy(
            delay: float,
            callback: Callable[..., object],
            *args: object,
        ) -> asyncio.TimerHandle:
            handle = real_call_later(delay, callback, *args)
            handles.append(handle)
            return handle

        mocker.patch.object(loop, "call_later", spy)
        agent = build_host_agent(_saving_then_answering())
        await run_with_deadline(
            agent, _deps(), "remember the vet", recorder=TurnRecorder(), seconds=_CANCEL_CEILING
        )
        assert handles, "no deadline timer was armed"
        assert all(handle.cancelled() for handle in handles)

    async def test_a_deadline_run_forwards_the_hosts_own_arguments(self) -> None:
        """History, display and limits all reach the inner turn.

        Dropping any of them silently changed the run: a deadline wrapper that
        ignores `usage_limits` spends past the host's ceiling.
        """
        shown: list[str] = []
        history: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content="remember the vet")]),
            ModelResponse(parts=[TextPart(content="noted")]),
        ]
        agent = build_host_agent(_saving_then_answering())
        recorder = TurnRecorder()
        with pytest.raises(UsageLimitExceeded):
            await run_with_deadline(
                agent,
                _deps(),
                "and the dentist",
                recorder=recorder,
                seconds=_CANCEL_CEILING,
                history=history,
                display=shown.append,
                usage_limits=UsageLimits(request_limit=1),
            )
        # The limit was forwarded, so the turn stopped at one request...
        assert recorder.usage is not None
        assert recorder.usage.requests == 1
        # ...the display was forwarded, so the narration arrived...
        assert shown == ["let me write that down. "]
        # ...and the history was forwarded, so it is in the checkpoint.
        assert any(
            isinstance(part, TextPart) and part.content == "noted"
            for message in recorder.partial
            for part in message.parts
        )


class TestTheHostsPlaceholderReachesCompaction:
    async def test_the_example_supplies_its_own_cleared_text(self, tmp_path: Path) -> None:
        """The example's whole point: a host's tools have side effects."""
        settings = EljaSettings(
            workspace=WorkspaceConfig(root=tmp_path),
            compaction=CompactionConfig(target_tokens=3000, keep_tool_pairs=1, keep_messages=2),
        )
        views: list[list[ModelMessage]] = []

        async def stream(
            messages: list[ModelMessage], info: AgentInfo
        ) -> AsyncIterator[StreamItem]:
            views.append(list(messages))
            yield "ok"

        agent = build_host_agent(FunctionModel(stream_function=stream), settings=settings)
        history: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content="keep the records")])
        ]
        for i in range(8):
            history.append(
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="save_fact",
                            args={"key": f"k{i}", "value": "v"},
                            tool_call_id=f"c{i}",
                        )
                    ]
                )
            )
            history.append(
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name="save_fact",
                            content=f"receipt{i} " * 600,
                            tool_call_id=f"c{i}",
                        )
                    ]
                )
            )
        await run_turn(agent, _deps(), "carry on", history=history, recorder=TurnRecorder())
        rendered = str(views[0])
        assert CLEARED in rendered
        assert "re-run the tool" not in rendered
        assert ".elja/spill/" not in rendered


def _writes_then_raises() -> tuple[FunctionToolset[HostDeps], FunctionModel, list[int]]:
    """A tool that commits a side effect and then fails. No timing needed.

    The example's own tools have no await point in them, so no script in this file
    can interrupt one mid-body — which is why the regime where the dying work IS the
    tool went untested. A tool that raises after writing reaches it deterministically,
    and is the ordinary shape of the problem: a payment gateway that errors after the
    debit posted.
    """
    toolset: FunctionToolset[HostDeps] = FunctionToolset()
    invocations: list[int] = []

    @toolset.tool
    async def charge(ctx: RunContext[HostDeps], amount: str) -> str:
        """Commit, then fail."""
        invocations.append(1)
        ctx.deps.store["charged"] = ctx.deps.store.get("charged", "") + amount + ";"
        raise RuntimeError("payment gateway 500 after the debit posted")

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[StreamItem]:
        yield {1: DeltaToolCall(name="charge", json_args='{"amount": "100"}')}

    return toolset, FunctionModel(stream_function=stream), invocations


class TestACheckpointIsReplayable:
    """E6's checkpoint is worth nothing if the history it hands back is refused.

    A provider rejects a history whose response has a tool call with no result, and so
    does pydantic-ai. Its own repair pass deliberately skips the LAST response —
    those calls are the live frontier that resumption may still answer — and a turn
    that dies inside a tool leaves its dangling call exactly there. So the host closes
    the frontier itself.
    """

    def _agent(
        self, toolset: FunctionToolset[HostDeps], model: FunctionModel
    ) -> Agent[HostDeps, str | DeferredToolRequests]:
        return build_application_agent(
            model,
            deps_type=HostDeps,
            output_type=[str, DeferredToolRequests],
            toolsets=[toolset],
        )

    async def test_a_tool_that_dies_mid_body_leaves_its_call_answered(self) -> None:
        """Without this the checkpoint carries a dangling call and cannot be replayed."""
        toolset, model, invocations = _writes_then_raises()
        deps = _deps()
        recorder = TurnRecorder()
        with pytest.raises(RuntimeError, match="payment gateway 500"):
            await run_turn(self._agent(toolset, model), deps, "charge the card", recorder=recorder)
        # The effect landed, once.
        assert deps.store == {"charged": "100;"}
        assert len(invocations) == 1
        # And the call it came from is answered, marked as interrupted rather than as
        # a result the model should believe.
        closed = [
            part
            for message in recorder.partial
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        assert [(part.tool_name, part.outcome) for part in closed] == [("charge", "interrupted")]

    async def test_the_checkpoint_replays_with_a_new_prompt_and_does_not_retry(self) -> None:
        """The two failures a dangling call causes, both closed.

        Replaying a dangling call with a new prompt raises `UserError`; replaying it
        *without* one re-executes the tool, which is the duplicate side effect the
        checkpoint exists to prevent, one level down. Neither happens here.
        """
        toolset, model, invocations = _writes_then_raises()
        deps = _deps()
        died = TurnRecorder()
        with pytest.raises(RuntimeError):
            await run_turn(self._agent(toolset, model), deps, "charge the card", recorder=died)

        async def answer(
            messages: list[ModelMessage], info: AgentInfo
        ) -> AsyncIterator[StreamItem]:
            yield "the charge was interrupted and I did not retry it"

        output, _ = await run_turn(
            self._agent(toolset, FunctionModel(stream_function=answer)),
            deps,
            "what happened?",
            history=died.partial,
            recorder=TurnRecorder(),
        )
        assert output == "the charge was interrupted and I did not retry it"
        assert len(invocations) == 1, "the interrupted tool ran a second time"
        assert deps.store == {"charged": "100;"}

    async def test_a_finished_turn_is_handed_back_unchanged(self) -> None:
        """The repair must not invent a return for a call that already has one."""
        from examples.server_agent import close_interrupted_calls

        recorder = TurnRecorder()
        agent = build_host_agent(_saving_then_answering())
        _, messages = await run_turn(agent, _deps(), "remember the vet", recorder=recorder)
        assert close_interrupted_calls(messages) == list(messages)
        # And a finished turn leaves no checkpoint to reconcile at all.
        assert recorder.partial == []

    def test_a_tool_bound_retry_answers_its_call_but_a_plain_one_does_not(self) -> None:
        """The distinction upstream draws, and getting it wrong breaks both directions.

        A `RetryPromptPart` carrying a `tool_name` is how a tool's `ModelRetry` reaches
        the model: it answers the call, so closing it out again would double-answer it.
        One *without* a `tool_name` is validation feedback rendered as plain user text —
        it answers nothing, so treating it as an answer leaves a genuinely dangling call
        open and the history unreplayable.
        """
        from examples.server_agent import close_interrupted_calls

        answered: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content="look it up")]),
            ModelResponse(parts=[ToolCallPart(tool_name="t", args={}, tool_call_id="c1")]),
            ModelRequest(
                parts=[RetryPromptPart(content="no such record", tool_name="t", tool_call_id="c1")]
            ),
        ]
        assert close_interrupted_calls(answered) == answered

        unanswered: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content="look it up")]),
            ModelResponse(parts=[ToolCallPart(tool_name="t", args={}, tool_call_id="c2")]),
            ModelRequest(parts=[RetryPromptPart(content="bad json")]),
        ]
        repaired = close_interrupted_calls(unanswered)
        closed = [
            part
            for message in repaired
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        assert [(part.tool_call_id, part.outcome) for part in closed] == [("c2", "interrupted")]
        # And the synthesized result sits BEFORE the user-facing retry text, where
        # providers expect tool results.
        tail = repaired[-1]
        assert isinstance(tail, ModelRequest)
        assert [type(part).__name__ for part in tail.parts] == [
            "ToolReturnPart",
            "RetryPromptPart",
        ]


class TestTheCheckpointSurvivesAnExternalCancellation:
    """`except BaseException`, and the breadth is the whole point.

    Narrowing it to `except Exception` — a plausible later lint concession, ruff's
    `BLE001`/`B036` both suggest it — survives the suite while silently losing the
    checkpoint for every SIGTERM- or timeout-driven shutdown, which is the path
    `run_with_deadline`'s docstring singles out as the one a host has to handle
    differently.
    """

    async def test_cancelling_the_consumer_task_still_leaves_the_checkpoint(self) -> None:
        deps = _deps()
        recorder = TurnRecorder()
        agent = build_host_agent(_writes_then_stalls())
        task = asyncio.create_task(run_turn(agent, deps, "remember the vet", recorder=recorder))
        # Wait for the tool to have committed, bounded so a turn that never gets
        # there fails fast instead of hanging.
        async with asyncio.timeout(_CANCEL_CEILING):
            while not deps.store:
                await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert deps.store == {"acme:vet": "tuesday"}
        returns = [
            part
            for message in recorder.partial
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        assert [part.tool_name for part in returns] == ["save_fact"]


def _stalls_inside_the_tool() -> tuple[FunctionToolset[HostDeps], FunctionModel]:
    """Writes, then blocks INSIDE the tool body — so the tool is what gets interrupted.

    `_writes_then_stalls` stalls the *model stream* on a later turn, which lets the
    tool complete: no call is ever interrupted, so it cannot distinguish a history
    that closes out interrupted calls from one that never had any. This fixture is the
    regime the claim is about.
    """
    toolset: FunctionToolset[HostDeps] = FunctionToolset()

    @toolset.tool
    async def charge(ctx: RunContext[HostDeps], amount: str) -> str:
        """Commit, then block until something stops us."""
        ctx.deps.store["charged"] = amount
        await asyncio.sleep(_STALL)
        return "ok"  # pragma: no cover - the deadline always wins

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[StreamItem]:
        yield {1: DeltaToolCall(name="charge", json_args='{"amount": "100"}')}

    return toolset, FunctionModel(stream_function=stream)


def _dangling_tool_names(messages: Sequence[ModelMessage]) -> list[str]:
    answered = {
        part.tool_call_id
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    }
    return [
        part.tool_name
        for message in messages
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, ToolCallPart) and part.tool_call_id not in answered
    ]


class TestTheRepairMatchesUpstreamsOrderedWalk:
    """A flat set of answered ids is not the same test, and the gap reopens the wedge.

    pydantic-ai's own pass walks the history in order: a result answers only a call
    that is *open* at that point. A set says "answered" for a call whose id was reused
    later and for a result that precedes its own call — and then the frontier stays
    dangling, which is the `UserError` this whole function exists to prevent. My first
    version used a set; these are the shapes that caught it.
    """

    @staticmethod
    def _call(call_id: str, name: str = "t") -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart(tool_name=name, args={}, tool_call_id=call_id)])

    @staticmethod
    def _return(call_id: str, name: str = "t") -> ModelRequest:
        return ModelRequest(
            parts=[ToolReturnPart(tool_name=name, content="ok", tool_call_id=call_id)]
        )

    def test_a_reused_call_id_does_not_mask_the_dangling_call(self) -> None:
        """Providers assign the id; `OpenAIChatModel` takes it verbatim with no guard."""
        from examples.server_agent import close_interrupted_calls

        history: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content="charge it twice")]),
            self._call("call_0"),
            self._return("call_0"),
            self._call("call_0"),
        ]
        assert _dangling_tool_names(history) == [], "the set-based view saw nothing dangling"
        repaired = close_interrupted_calls(history)
        assert _dangling_tool_names(repaired) == []
        outcomes = [
            part.outcome
            for message in repaired
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        assert outcomes == ["success", "interrupted"]

    def test_a_call_shadowing_an_open_call_leaves_both_to_close(self) -> None:
        """The shape upstream's comment is actually about, and the one a set cannot see.

        Two calls sharing an id with no result in between: the first can never be
        answered, because any later result answers the second. Both have to be closed
        out, and they belong to different responses — which is also the only way a
        synthesized request lands between two responses rather than at the end.
        """
        from examples.server_agent import close_interrupted_calls

        history: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content="go")]),
            self._call("dup"),
            self._call("dup"),
            ModelResponse(parts=[TextPart(content="never mind")]),
        ]
        repaired = close_interrupted_calls(history)
        closed = [
            part
            for message in repaired
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        assert [part.outcome for part in closed] == ["interrupted", "interrupted"]
        # One synthesized request per response that had a dangling call, in order,
        # rather than both piled at the end.
        assert [type(message).__name__ for message in repaired] == [
            "ModelRequest",
            "ModelResponse",
            "ModelRequest",
            "ModelResponse",
            "ModelRequest",
            "ModelResponse",
        ]
        assert _dangling_tool_names(repaired) == []

    def test_a_result_preceding_its_call_answers_nothing(self) -> None:
        """A shape a context-eviction pass or a hand-built history can produce."""
        from examples.server_agent import close_interrupted_calls

        history: list[ModelMessage] = [
            ModelRequest(
                parts=[
                    UserPromptPart(content="go"),
                    ToolReturnPart(tool_name="t", content="ok", tool_call_id="c1"),
                ]
            ),
            self._call("c1"),
        ]
        repaired = close_interrupted_calls(history)
        assert _dangling_tool_names(repaired) == []
        assert len(repaired) == len(history) + 1

    def test_the_repair_is_idempotent(self) -> None:
        """Upstream promises this of its own pass, and a host may repair on every save."""
        from examples.server_agent import close_interrupted_calls

        history: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content="go")]),
            self._call("c2"),
        ]
        once = close_interrupted_calls(history)
        assert close_interrupted_calls(once) == once

    def test_a_synthesized_return_matches_what_upstream_would_have_written(self) -> None:
        """Content, marker and timestamp, each asserted rather than described.

        The timestamp comes from the response being repaired, not the wall clock, which
        is what makes a second pass produce the same bytes — and is why `serialize` of
        two repairs of the same history compares equal.
        """
        from examples.server_agent import close_interrupted_calls

        response = self._call("c3")
        history: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content="go")]),
            response,
        ]
        (synthesized,) = [
            part
            for message in close_interrupted_calls(history)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        assert synthesized.content == INTERRUPTED_TOOL_RETURN_CONTENT
        assert synthesized.outcome == "interrupted"
        assert synthesized.metadata == {"pydantic_ai_synthesized_tool_return": True}
        assert synthesized.timestamp == response.timestamp
        # Deterministic: the same history repaired twice serializes identically.
        assert serialize(close_interrupted_calls(history)) == serialize(
            close_interrupted_calls(history)
        )

    def test_the_caller_never_gets_the_runs_own_list(self) -> None:
        """Even when there is nothing to repair, which is the case that used to alias."""
        from examples.server_agent import close_interrupted_calls

        history: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content="go")])]
        assert close_interrupted_calls(history) is not history


class TestADeferredCallIsAPendingQuestionNotInterruptedWork:
    """Closing out a deferred call destroys the host's question, permanently.

    `ask_user` raises `CallDeferred`, and this example ships it in the same toolset as
    its writing tools — so one response holding both is on-path. A deferred call is a
    question already put to someone; answering it later is the entire point of the
    ask-resume loop. Closed out, the answer is rejected with `UserError('Tool call …
    was already executed and its result cannot be overridden.')` and the model is told
    the question was interrupted. The history cannot tell the two apart (`tool_kind` is
    `None` on both), so the host names its own deferring tools.
    """

    @staticmethod
    def _mixed() -> list[ModelMessage]:
        return [
            ModelRequest(parts=[UserPromptPart(content="book the vet and charge it")]),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="ask_user", args={"question": "which vet?"}, tool_call_id="q1"
                    ),
                    ToolCallPart(tool_name="charge", args={"amount": "100"}, tool_call_id="p1"),
                ]
            ),
        ]

    def test_the_question_stays_open_while_the_failing_sibling_is_closed(self) -> None:
        from examples.server_agent import close_interrupted_calls

        repaired = close_interrupted_calls(self._mixed(), leave_open={"ask_user"})
        closed = [
            (part.tool_name, part.outcome)
            for message in repaired
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        assert closed == [("charge", "interrupted")]
        # The question is still unanswered, which is what lets the host answer it.
        assert _dangling_tool_names(repaired) == ["ask_user"]

    def test_without_leave_open_the_question_is_destroyed(self) -> None:
        """The other direction, so the argument cannot quietly stop being passed."""
        from examples.server_agent import close_interrupted_calls

        repaired = close_interrupted_calls(self._mixed())
        closed = [
            part.tool_name
            for message in repaired
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        assert closed == ["ask_user", "charge"]

    async def test_the_hosts_own_checkpoint_leaves_its_deferred_tool_open(self) -> None:
        """Pinned through `run_turn`, not just the helper, so the wiring is covered."""
        toolset: FunctionToolset[HostDeps] = FunctionToolset()

        @toolset.tool
        async def ask_user(ctx: RunContext[HostDeps], question: str) -> str:
            """Defer, exactly as the example's own tool does."""
            raise CallDeferred

        @toolset.tool
        async def charge(ctx: RunContext[HostDeps], amount: str) -> str:
            ctx.deps.store["charged"] = amount
            raise RuntimeError("gateway 500")

        async def stream(
            messages: list[ModelMessage], info: AgentInfo
        ) -> AsyncIterator[StreamItem]:
            yield {
                1: DeltaToolCall(name="ask_user", json_args='{"question": "which vet?"}'),
                2: DeltaToolCall(name="charge", json_args='{"amount": "100"}'),
            }

        agent: Agent[HostDeps, str | DeferredToolRequests] = build_application_agent(
            FunctionModel(stream_function=stream),
            deps_type=HostDeps,
            output_type=[str, DeferredToolRequests],
            toolsets=[toolset],
        )
        deps = _deps()
        recorder = TurnRecorder()
        with pytest.raises(RuntimeError, match="gateway 500"):
            await run_turn(agent, deps, "book and charge", recorder=recorder)
        assert deps.store == {"charged": "100"}
        assert _dangling_tool_names(recorder.partial) == ["ask_user"]


class TestACancelledRunsOwnHistoryIsNotResumable:
    """The claim the docstring used to make, measured and corrected.

    `RunCancelled.all_messages()` closes out an interrupted call only where a sibling
    in the same response already returned. A tool stopped inside its own body leaves
    its call dangling there — so a host on the deadline path must persist the
    checkpoint it repaired, not the exception's history.
    """

    async def test_the_exceptions_history_dangles_where_the_checkpoint_does_not(
        self,
    ) -> None:
        toolset, model = _stalls_inside_the_tool()
        agent: Agent[HostDeps, str | DeferredToolRequests] = build_application_agent(
            model,
            deps_type=HostDeps,
            output_type=[str, DeferredToolRequests],
            toolsets=[toolset],
        )
        deps = _deps()
        recorder = TurnRecorder()
        async with asyncio.timeout(_CANCEL_CEILING):
            with pytest.raises(RunCancelled) as caught:
                await run_with_deadline(agent, deps, "charge it", recorder=recorder, seconds=0.05)
        assert deps.store == {"charged": "100"}
        # The exception's own snapshot is NOT resumable...
        assert _dangling_tool_names(caught.value.all_messages()) == ["charge"]
        # ...and the checkpoint the host persists is.
        assert _dangling_tool_names(recorder.partial) == []


class TestLeaveOpenIsASetNotJustAContainer:
    """`Container[str]` accepted a bare `str`, and `in` on a `str` is substring matching.

    So `leave_open="ask_user"` type-checked clean under strict mypy and silently kept
    every tool whose name is a substring of it — `ask` and `user` — open as well,
    leaving a history the next replay refuses. `AbstractSet[str]` makes that a type
    error, because a `str` is not a `Set`.
    """

    @staticmethod
    def _three_dangling() -> list[ModelMessage]:
        return [
            ModelRequest(parts=[UserPromptPart(content="go")]),
            ModelResponse(
                parts=[
                    ToolCallPart(tool_name="ask", args={}, tool_call_id="a"),
                    ToolCallPart(tool_name="user", args={}, tool_call_id="u"),
                    ToolCallPart(tool_name="charge", args={}, tool_call_id="c"),
                ]
            ),
        ]

    def test_only_the_named_tool_is_left_open(self) -> None:
        from examples.server_agent import close_interrupted_calls

        repaired = close_interrupted_calls(self._three_dangling(), leave_open={"ask"})
        closed = [
            part.tool_name
            for message in repaired
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        # Exactly the two NOT named, rather than everything whose name happens to be a
        # substring of the argument.
        assert sorted(closed) == ["charge", "user"]

    def test_the_default_closes_everything(self) -> None:
        from examples.server_agent import close_interrupted_calls

        repaired = close_interrupted_calls(self._three_dangling())
        closed = [
            part.tool_name
            for message in repaired
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        assert sorted(closed) == ["ask", "charge", "user"]


class TestASynthesizedResultGoesAfterTheExistingOnes:
    """The position rule, on the only shape that has a non-zero insert point.

    Both docstrings single out "a response with two calls where one returned" as the
    load-bearing case, and it is the one a real run produces when parallel calls
    resolve differently. Providers expect tool results ahead of user-facing parts, so
    a synthesized result has to land after the existing results and before the rest —
    and two surviving mutants (scanning forward, or inserting at the match instead of
    after it) show that was asserted nowhere.
    """

    def test_the_trailing_request_keeps_results_first_and_in_order(self) -> None:
        from examples.server_agent import close_interrupted_calls

        history: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content="do both")]),
            ModelResponse(
                parts=[
                    ToolCallPart(tool_name="quick", args={}, tool_call_id="q"),
                    ToolCallPart(tool_name="also", args={}, tool_call_id="z"),
                    ToolCallPart(tool_name="slow", args={}, tool_call_id="s"),
                ]
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(tool_name="quick", content="ok", tool_call_id="q"),
                    ToolReturnPart(tool_name="also", content="ok", tool_call_id="z"),
                    UserPromptPart(content="and while you are there"),
                ]
            ),
        ]
        repaired = close_interrupted_calls(history)
        tail = repaired[-1]
        assert isinstance(tail, ModelRequest)
        # Two existing results, then the synthesized one, then the user text. Two
        # existing results rather than one so the order separates "insert after the
        # last match" from "insert at the first".
        assert [
            (type(part).__name__, getattr(part, "tool_call_id", None)) for part in tail.parts
        ] == [
            ("ToolReturnPart", "q"),
            ("ToolReturnPart", "z"),
            ("ToolReturnPart", "s"),
            ("UserPromptPart", None),
        ]


class TestAKeptOpenCheckpointResumesThroughDeferredResults:
    """What `leave_open` actually buys, asserted rather than inferred.

    The existing test only shows the call is still dangling and comments that this is
    "what lets the host answer it" — an inference. A kept-open checkpoint is **not**
    replayable with a new user prompt (same `UserError`); it is replayable with
    `deferred_tool_results=` keyed on the pending call's `tool_call_id`, which the host
    reads off the dangling `ToolCallPart` because the run raised instead of handing
    back a `DeferredToolRequests`. Both directions below.
    """

    @staticmethod
    def _asks_then_fails() -> tuple[FunctionToolset[HostDeps], FunctionModel]:
        toolset: FunctionToolset[HostDeps] = FunctionToolset()

        @toolset.tool
        async def ask_user(ctx: RunContext[HostDeps], question: str) -> str:
            """Defer, as the example's own tool does."""
            raise CallDeferred

        @toolset.tool
        async def charge(ctx: RunContext[HostDeps], amount: str) -> str:
            ctx.deps.store["charged"] = amount
            raise RuntimeError("gateway 500")

        async def stream(
            messages: list[ModelMessage], info: AgentInfo
        ) -> AsyncIterator[StreamItem]:
            yield {
                1: DeltaToolCall(name="ask_user", json_args='{"question": "which vet?"}'),
                2: DeltaToolCall(name="charge", json_args='{"amount": "100"}'),
            }

        return toolset, FunctionModel(stream_function=stream)

    async def _dying_turn(self) -> tuple[Agent[HostDeps, Any], HostDeps, TurnRecorder, str]:
        toolset, model = self._asks_then_fails()
        agent: Agent[HostDeps, str | DeferredToolRequests] = build_application_agent(
            model,
            deps_type=HostDeps,
            output_type=[str, DeferredToolRequests],
            toolsets=[toolset],
        )
        deps = _deps()
        recorder = TurnRecorder()
        with pytest.raises(RuntimeError, match="gateway 500"):
            await run_turn(agent, deps, "book and charge", recorder=recorder)
        pending = [
            part.tool_call_id
            for message in recorder.partial
            for part in message.parts
            if isinstance(part, ToolCallPart) and part.tool_name == "ask_user"
        ]
        assert len(pending) == 1
        return agent, deps, recorder, pending[0]

    async def test_answering_the_question_resumes_the_conversation(self) -> None:
        agent, deps, recorder, call_id = await self._dying_turn()

        async def answer(
            messages: list[ModelMessage], info: AgentInfo
        ) -> AsyncIterator[StreamItem]:
            yield "booked with Dr. Meow"

        resumed: Agent[HostDeps, str | DeferredToolRequests] = build_application_agent(
            FunctionModel(stream_function=answer),
            deps_type=HostDeps,
            output_type=[str, DeferredToolRequests],
            toolsets=[household_toolset()],
        )
        output, _ = await run_turn(
            resumed,
            deps,
            history=recorder.partial,
            recorder=TurnRecorder(),
            deferred_tool_results=DeferredToolResults(calls={call_id: "Dr. Meow"}),
        )
        assert output == "booked with Dr. Meow"

    async def test_a_new_prompt_silently_closes_the_question_instead(self) -> None:
        """The cost of keeping it open, and it is not an error — which is the trap.

        Replaying a kept-open checkpoint with a new user prompt *succeeds*. pydantic-ai
        repairs the history at send time, so the pending question reaches the model as
        `ToolReturnPart(ask_user, outcome='interrupted')` and the host's answer can
        never be supplied afterwards. Nothing warns. So `leave_open` buys the
        `deferred_tool_results` path and nothing else, and a host that takes the other
        one has quietly discarded the question it already asked.
        """
        _, deps, recorder, _ = await self._dying_turn()
        seen: list[tuple[str, object, object]] = []

        async def spy(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[StreamItem]:
            seen.extend(
                (
                    type(part).__name__,
                    getattr(part, "outcome", None),
                    getattr(part, "tool_name", None),
                )
                for message in messages
                for part in message.parts
            )
            yield "never mind then"

        agent: Agent[HostDeps, str | DeferredToolRequests] = build_application_agent(
            FunctionModel(stream_function=spy),
            deps_type=HostDeps,
            output_type=[str, DeferredToolRequests],
            toolsets=[household_toolset()],
        )
        output, _ = await run_turn(
            agent, deps, "never mind", history=recorder.partial, recorder=TurnRecorder()
        )
        assert output == "never mind then"
        assert ("ToolReturnPart", "interrupted", "ask_user") in seen, (
            "the question was not closed out, so this trap no longer exists and the "
            "docstring should say so"
        )
