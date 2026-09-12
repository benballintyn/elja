"""A server-shaped elja host: caller owns tools, deps, history and events.

This is the embedded path end to end, with nothing the CLI brings: no Rich, no
terminal, no workspace, no session file. Every piece a host is expected to own is
owned here, and the contract each part relies on is pinned in
``tests/test_server_contracts.py``.

What the host owns, and why elja does not:

- **Event identity and storage.** :class:`TurnRecorder` assigns its own
  monotonic sequence numbers. pydantic-ai's events carry no durable id, and
  inventing one inside elja would be a parallel protocol competing with the
  native one.
- **Persistence.** History goes in and out as ``message_history``, serialized
  with pydantic-ai's own adapter. Nothing is written unless the host writes it,
  and no turn is implicitly saved.
- **Display.** A display observer is optional and isolated: its exceptions are
  swallowed, because a broken UI must not abort a paid run. The *recorder* is not
  isolated — a host that cannot record must not keep spending.
- **Checkpoints.** A turn that dies after a tool already wrote leaves an effect
  the host has to reconcile, so :func:`run_turn` hands back everything that
  completed before re-raising (:attr:`TurnRecorder.partial`). A host that
  persisted nothing would replay a history with no trace of the write, and the
  model would write again.
- **Cancellation.** pydantic-ai's own cancellation channel stops a run; elja adds
  none of its own. See :func:`run_with_deadline` for why that beats wrapping the
  call in ``asyncio.timeout``.
- **Compaction is silent.** Nothing notifies the host that history was rewritten:
  the only callback the harness package offers is
  :class:`ReportContextUsage.on_usage`, which carries a reading and no messages.
  A host that wants to show "history compacted" infers it from a reading that
  drops, or from its own copy of the transcript.

What elja provides is the construction door
(:func:`elja.build_application_agent`), the compaction policy
(:func:`elja.compaction.build_compaction`), and the guarantees in
``docs/EMBEDDING.md``. Everything else below is ordinary pydantic-ai.
"""

import asyncio
import contextlib
from collections.abc import Callable, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field, replace
from typing import Any, TypeGuard

from pydantic_ai import (
    Agent,
    AgentRunResultEvent,
    CancellationToken,
    DeferredToolRequests,
    DeferredToolResults,
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    RunContext,
)
from pydantic_ai.agent import AgentRunEvents
from pydantic_ai.exceptions import CallDeferred, UserError
from pydantic_ai.messages import (
    INTERRUPTED_TOOL_RETURN_CONTENT,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models import Model
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.usage import RunUsage, UsageLimits
from pydantic_ai_harness.compaction import ContextUsage, ReportContextUsage

from elja import build_application_agent
from elja.compaction import build_compaction
from elja.settings import EljaSettings

# Masked tool results must not invite the model to repeat a side effect. elja's
# default says "re-run the tool", which is true of its own idempotent reads and
# false of anything a host writes.
CLEARED = "[result cleared to save context; retrieve the saved result by its receipt id]"


@dataclass
class HostDeps:
    """The host's own dependencies. No elja type, no workspace."""

    tenant: str
    store: dict[str, str]


@dataclass
class RecordedEvent:
    """One event in the host's durable log."""

    seq: int
    kind: str
    detail: str


@dataclass
class TurnRecorder:
    """The host's authoritative record of a turn.

    Sequence numbers are the host's, because nothing in the framework promises a
    durable, contiguous id — and inferring one from event order would be a
    contract the producer never made.
    """

    events: list[RecordedEvent] = field(default_factory=list)
    usage: RunUsage | None = None
    context: list[ContextUsage] = field(default_factory=list)
    partial: list[ModelMessage] = field(default_factory=list)
    """What a failed turn completed, for the host to persist before it re-raises.

    Empty after a turn that finished: the full history is the return value then.
    """
    _seq: int = 0

    def record(self, kind: str, detail: str = "") -> None:
        """Append one event under the next sequence number."""
        self._seq += 1
        self.events.append(RecordedEvent(seq=self._seq, kind=kind, detail=detail))

    def kinds(self) -> list[str]:
        """The recorded event kinds, in order."""
        return [event.kind for event in self.events]


def household_toolset() -> FunctionToolset[HostDeps]:
    """The host's domain tools — the only tools the agent gets."""
    toolset: FunctionToolset[HostDeps] = FunctionToolset()

    @toolset.tool
    async def save_fact(ctx: RunContext[HostDeps], key: str, value: str) -> str:
        """Write a household fact and return its receipt id."""
        receipt = f"{ctx.deps.tenant}:{key}"
        ctx.deps.store[receipt] = value
        return receipt

    @toolset.tool
    async def read_fact(ctx: RunContext[HostDeps], key: str) -> str:
        """Read a household fact."""
        return ctx.deps.store.get(f"{ctx.deps.tenant}:{key}", "unknown")

    @toolset.tool
    async def ask_user(ctx: RunContext[HostDeps], question: str) -> str:
        """Ask the household a question, answered out of band.

        ``CallDeferred`` ends the run with a typed ``DeferredToolRequests``
        output carrying the question — one model call, no retry, no exception
        standing in for a question. The host shows it, collects an answer, and
        resumes the same conversation with ``deferred_tool_results=``.
        """
        raise CallDeferred

    return toolset


def build_host_agent(
    model: Model,
    *,
    settings: EljaSettings | None = None,
    on_context: Callable[[ContextUsage], None] | None = None,
) -> Agent[HostDeps, str | DeferredToolRequests]:
    """Build the host's agent.

    Args:
        model: The host's model, already wrapped for metering if it meters.
        settings: Supplies the compaction policy only. ``None`` means no
            compaction, which is a deliberate choice rather than a default.
        on_context: Receives a context reading per request. Attached LAST, so it
            measures the request that was actually sent; placed earlier it would
            report one that never existed.

    Returns:
        An agent whose output is either the final answer or a typed request for
        the human — never an exception standing in for a question.
    """
    capabilities: list[Any] = []
    if settings is not None:
        capabilities.extend(build_compaction(settings, cleared_placeholder=CLEARED))
    if on_context is not None:
        capabilities.append(ReportContextUsage(on_usage=on_context))
    return build_application_agent(
        model,
        deps_type=HostDeps,
        # A union output: the run can end by answering, or by asking.
        output_type=[str, DeferredToolRequests],
        instructions="You keep one household's records. Save what you are told.",
        toolsets=[household_toolset()],
        capabilities=capabilities,
        name="household-agent",
    )


async def run_turn(
    agent: Agent[HostDeps, str | DeferredToolRequests],
    deps: HostDeps,
    prompt: str | None = None,
    *,
    history: Sequence[ModelMessage] = (),
    recorder: TurnRecorder,
    display: Callable[[str], None] | None = None,
    usage_limits: UsageLimits | None = None,
    deferred_tool_results: DeferredToolResults | None = None,
    cancellation_token: CancellationToken | None = None,
) -> tuple[str | DeferredToolRequests, list[ModelMessage]]:
    """Stream one turn, recording what the host needs to keep.

    Every text part the model emits reaches ``display`` as it arrives — the
    narration and the final answer alike, because the stream does not distinguish
    them while they are arriving. Only the run's final output is *returned*, and
    that is the copy a host should treat as the answer; elja never folds mid-run
    text into it.

    The event stream does not label narration prospectively: ``FinalResultEvent``
    means "a part that could be the final output has started", and fires for
    mid-run text on a turn that goes on to call a tool. So the answer is taken
    from ``AgentRunResultEvent`` and from nowhere else. A host that accumulates
    text deltas into its answer will ship the narration with it.

    **If the turn dies, what it completed is still yours.** A tool that already
    ran has an effect in the host's world, so anything the run finished lands in
    ``recorder.partial`` before the exception re-raises — persist that, or the
    next turn replays a history with no trace of the write and the model writes
    again. This is the host-selected checkpoint: the host decides where its save
    goes, and the stream only promises not to throw the evidence away.

    Args:
        agent: From :func:`build_host_agent`.
        deps: The host's dependencies for this run.
        prompt: The user's message, or ``None`` when resuming a deferred turn
            whose prompt is already in ``history``.
        history: Prior messages, caller-owned.
        recorder: The host's durable log. Its failures are NOT suppressed.
        display: Optional UI sink. Its failures ARE suppressed — a dropped
            subscriber must not abort a paid run, and may reconnect freely.
        usage_limits: The host's ceilings for this run.
        deferred_tool_results: Answers to a previous turn's
            ``DeferredToolRequests``, which close the ask-resume loop in the same
            conversation rather than starting a new one.
        cancellation_token: The host's stop button. See
            :func:`run_with_deadline`.

    Returns:
        The final output and the full message list to persist.
    """
    final: str | DeferredToolRequests | None = None
    messages: list[ModelMessage] = []
    async with agent.run_stream_events(
        prompt,
        deps=deps,
        message_history=list(history),
        usage_limits=usage_limits,
        deferred_tool_results=deferred_tool_results,
        cancellation_token=cancellation_token,
    ) as events:
        try:
            async for event in events:
                if isinstance(event, PartStartEvent) and isinstance(event.part, ThinkingPart):
                    recorder.record("thinking")
                elif isinstance(event, PartStartEvent) and isinstance(event.part, ToolCallPart):
                    recorder.record("tool_call_started", event.part.tool_name)
                elif isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
                    # Shown, never accumulated. This fires for the answer's part as
                    # well as for narration, and nothing here can tell them apart —
                    # which is why the answer is read from the result instead.
                    _show(display, event.part.content)
                elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                    _show(display, event.delta.content_delta)
                elif isinstance(event, FunctionToolCallEvent):
                    recorder.record("tool_call", event.part.tool_name)
                elif isinstance(event, FunctionToolResultEvent):
                    kind = "tool_error" if _is_retry(event) else "tool_result"
                    recorder.record(kind, event.part.tool_name or "")
                elif isinstance(event, FinalResultEvent):
                    # NOT "this is the answer". FinalResultEvent fires whenever a
                    # part that COULD be the final output starts, so a turn that
                    # narrates and then calls a tool emits it too — measured twice in
                    # one turn. The answer comes from AgentRunResultEvent alone.
                    recorder.record("output_part_started")
                elif isinstance(event, AgentRunResultEvent):
                    final = event.result.output
                    messages = list(event.result.all_messages())
                    recorder.usage = event.result.usage
        except BaseException:
            _checkpoint(recorder, events)
            raise
    if final is None:  # pragma: no cover - a failure re-raises from the iterator
        raise RuntimeError("the event stream ended without a result")
    recorder.record("turn_finished")
    return final, messages


# pydantic-ai marks the returns it synthesizes for interrupted calls with this
# metadata key, so repairing a history the way it would leaves nothing for its own
# repair pass to redo. Spelled out rather than imported: the constant lives in
# `pydantic_ai._agent_graph`, and an example should not reach into a private module.
_SYNTHESIZED_RETURN = "pydantic_ai_synthesized_tool_return"


def _is_tool_result(part: ModelRequestPart) -> TypeGuard[ToolReturnPart | RetryPromptPart]:
    """Whether a part answers a ``ToolCallPart``.

    A ``RetryPromptPart`` with no ``tool_name`` is validation feedback rendered as a
    plain user message, not a tool result — so it answers nothing, and treating it as
    an answer would leave a genuinely dangling call open. This mirrors pydantic-ai's
    own ``_is_tool_result_part``.
    """
    return isinstance(part, ToolReturnPart) or (
        isinstance(part, RetryPromptPart) and part.tool_name is not None
    )


def _dangling_calls_by_response(
    messages: Sequence[ModelMessage],
) -> dict[int, list[ToolCallPart]]:
    """Tool calls that will never receive a result, keyed by their response's index.

    An **ordered** walk, not a set of every answered id anywhere in the history. The
    difference is not theoretical: a result only answers a call that is *open* at that
    point, so a call whose id is reused by a later call can no longer be answered —
    any later result answers the new one — and a result that precedes its own call
    answers nothing. A flat set calls both of those "answered" and leaves the frontier
    dangling, which is the wedged history this whole function exists to prevent.

    Providers take the id they are given: ``OpenAIChatModel`` assigns
    ``tool_call_id=c.id`` verbatim with no uniqueness guard, and Chat Completions is
    what elja's ``openai`` provider builds — so a repeated id is a local server's
    choice, not something the framework rules out.
    """
    open_calls: dict[str, tuple[int, ToolCallPart]] = {}
    dangling: dict[int, list[ToolCallPart]] = {}
    for index, message in enumerate(messages):
        if isinstance(message, ModelResponse):
            for response_part in message.parts:
                if not isinstance(response_part, ToolCallPart):
                    continue
                if shadowed := open_calls.get(response_part.tool_call_id):
                    dangling.setdefault(shadowed[0], []).append(shadowed[1])
                open_calls[response_part.tool_call_id] = (index, response_part)
        else:
            for request_part in message.parts:
                if _is_tool_result(request_part) and request_part.tool_call_id is not None:
                    open_calls.pop(request_part.tool_call_id, None)
    for response_index, call in open_calls.values():
        dangling.setdefault(response_index, []).append(call)
    return dangling


def _insert_tool_results(request: ModelRequest, results: list[ModelRequestPart]) -> ModelRequest:
    """Put synthesized results after the request's existing ones, before user-facing parts.

    Where providers expect tool results to sit, and where pydantic-ai puts them.
    """
    insert_at = next(
        (
            index + 1
            for index in range(len(request.parts) - 1, -1, -1)
            if _is_tool_result(request.parts[index])
        ),
        0,
    )
    return replace(
        request, parts=[*request.parts[:insert_at], *results, *request.parts[insert_at:]]
    )


def close_interrupted_calls(
    messages: Sequence[ModelMessage],
    *,
    leave_open: AbstractSet[str] = frozenset(),
) -> list[ModelMessage]:
    """Answer every tool call a FAILED turn never finished, so the history can be replayed.

    Without this a checkpoint is not resumable, in exactly the case that motivates
    taking one. Replaying a history whose last response carries an unanswered tool call
    raises ``UserError('Cannot provide a new user prompt when the message history
    contains unprocessed tool calls.')``, and replaying it *without* a prompt
    re-executes the call — the duplicate side effect the checkpoint exists to prevent,
    one level down.

    The refusal is not a provider's. pydantic-ai repairs dangling calls at send time
    *unconditionally*, last response included, so one never reaches the wire. What
    leaves the last response alone is the earlier pass that decides how to resume a
    history handed back to it, and that is where the ``UserError`` comes from: those
    calls are the live frontier ``deferred_tool_results`` may still answer. A turn that
    died *inside* a tool leaves its dangling call exactly there. The one case that
    self-heals is a response with two calls where one returned — the completed
    sibling's request is recorded and the interrupted one is closed out alongside it —
    which is why a single-tool turn fails where a parallel one does not.

    So repair a history you are about to **persist and replay**, and never a successful
    ``DeferredToolRequests`` history: upstream preserves that frontier on purpose, and
    closing it is what ``leave_open`` exists to prevent.

    ``leave_open`` is the other half of that frontier, and the reason this function is
    for a failed turn only. A deferred call (``CallDeferred``, which is how ``ask_user``
    asks) is a pending question the host has already put to someone, not interrupted
    work. Closing it out makes the answer permanently unacceptable —
    ``UserError('Tool call ... was already executed and its result cannot be
    overridden.')`` — and tells the model the question was interrupted. The history
    cannot distinguish the two (``tool_kind`` is ``None`` on both), so the host names
    its own deferring tools.

    Args:
        messages: The run's messages, as handed back by a dying turn.
        leave_open: Names of tools whose dangling calls are pending questions rather
            than interrupted work, and must stay open for the host to answer. A
            ``Set``, not a ``Container``: a bare ``str`` would type-check and then match
            every tool whose name is one of its substrings. ``CallDeferred`` is not the
            only reason to list a tool — ``ApprovalRequired`` and externally executed
            tools have the same property.

    Returns:
        The history with every unanswered call answered except those named in
        ``leave_open``, which stay dangling by design. Each synthesized result carries
        pydantic-ai's own interrupted content, ``outcome='interrupted'``, the repaired
        response's timestamp (so a second pass over the same history produces the same
        bytes), and the marker pydantic-ai sets on its own synthesized returns. A new
        list either way, never the run's own.

        Resume a checkpoint with something left open using ``deferred_tool_results=``
        keyed on the pending call's ``tool_call_id``, which the host reads off the
        dangling ``ToolCallPart`` — on the failure path the run raised instead of
        returning ``DeferredToolRequests``, so the history is the only place that id
        exists. Replaying it with a new user prompt instead is not an error, and that is
        the trap: pydantic-ai repairs the history at send time, so the question is
        silently closed out as interrupted and the answer can never be supplied.
        Measured. Keeping a call open buys the ``deferred_tool_results`` path and
        nothing else.
    """
    dangling = {
        index: [call for call in calls if call.tool_name not in leave_open]
        for index, calls in _dangling_calls_by_response(messages).items()
    }
    if not any(dangling.values()):
        return list(messages)

    repaired: list[ModelMessage] = []
    synthesized: list[ModelRequestPart] = []
    for index, message in enumerate(messages):
        if isinstance(message, ModelResponse):
            if synthesized:
                repaired.append(ModelRequest(parts=synthesized))
                synthesized = []
            repaired.append(message)
            synthesized = [
                ToolReturnPart(
                    tool_name=call.tool_name,
                    content=INTERRUPTED_TOOL_RETURN_CONTENT,
                    tool_call_id=call.tool_call_id,
                    metadata={_SYNTHESIZED_RETURN: True},
                    timestamp=message.timestamp,
                    outcome="interrupted",
                )
                for call in dangling.get(index, ())
            ]
        elif synthesized:
            repaired.append(_insert_tool_results(message, synthesized))
            synthesized = []
        else:
            repaired.append(message)
    if synthesized:
        repaired.append(ModelRequest(parts=synthesized))
    return repaired


def _checkpoint(
    recorder: TurnRecorder,
    events: AgentRunEvents[str | DeferredToolRequests],
) -> None:
    """Hand the host what the dying turn completed, in a replayable shape.

    ``all_messages()`` returns the run's own live list, so it is copied rather than
    aliased — and copied through :func:`close_interrupted_calls`, because a turn that
    died inside a tool leaves a call the next replay would refuse.

    Both accessors raise ``UserError`` until the first iteration binds the run. That
    window is real rather than theoretical: the background task is created but not
    awaited, so an external cancellation landing in the few event-loop steps before
    the binding reaches here unbound. Nothing completed in that window, so the
    checkpoint is empty — and a framework complaint about iteration order must not
    replace the cancellation the host actually needs to see.
    """
    try:
        messages = events.all_messages()
        usage = events.usage
    except UserError:
        return
    # `ask_user` is this host's deferred tool: a dangling call to it is a question
    # already put to someone, not interrupted work, so it stays open to be answered.
    recorder.partial = close_interrupted_calls(messages, leave_open={"ask_user"})
    recorder.usage = usage


def _show(display: Callable[[str], None] | None, text: str) -> None:
    """Send text to the display, tolerating a broken subscriber.

    Empty deltas are real: a provider can close a text part with a zero-length
    chunk, and forwarding it would make a UI that renders per-chunk emit blanks.
    """
    if display is None or not text:
        return
    with contextlib.suppress(Exception):
        display(text)


def _is_retry(event: FunctionToolResultEvent) -> bool:
    """Whether a tool result was a retry prompt rather than a value.

    A retry prompt is how a tool's ModelRetry reaches the model, so a host that
    wants "the tool errored" in its log reads the part's type — there is no
    separate error event to subscribe to.
    """
    return isinstance(event.part, RetryPromptPart)


def serialize(messages: Sequence[ModelMessage]) -> bytes:
    """Persist history with pydantic-ai's own adapter.

    Opaque provider state — thinking signatures, provider ids — round-trips
    because the adapter owns the schema. Hand-rolling a second representation is
    how that state gets dropped.
    """
    return ModelMessagesTypeAdapter.dump_json(list(messages))


def deserialize(raw: bytes) -> list[ModelMessage]:
    """Load history the same way it was written."""
    return ModelMessagesTypeAdapter.validate_json(raw)


def foreign_thinking_parts(messages: Sequence[ModelMessage], model: Model) -> list[ThinkingPart]:
    """Thinking parts another provider produced, whose content this one will still see.

    A thinking *signature* is opaque provider state and pydantic-ai will not
    replay one across providers: each model attaches a signature only for parts
    whose provider is its own. The reasoning *content* is a different matter — it
    crosses anyway, on every mapping in the installed package. OpenAI's Responses
    path forwards it either as an assistant message wrapped in the profile's
    thinking tags or as a reasoning summary with ``encrypted_content=None``
    (``models/openai.py``, both arms of the ``ThinkingPart`` branch), and the
    Anthropic and Google mappings forward it too. So the words reach the new provider
    with the signature stripped. Filtering on the signature would hide exactly the
    parts that do get replayed, which is why this does not look at it.

    So this is not a safety filter — nothing here needs protecting. It is for
    *deciding explicitly* what to do with another provider's reasoning: log the
    switch, drop it, or start a fresh conversation, rather than replaying it
    unexamined.

    Two precisions about "foreign", both deliberate:

    - A part carrying no ``provider_name`` inherits its message's, because that is
      what OpenAI's **Responses** mapping does when it decides whether a part is its
      own. Only Responses: the Chat Completions mapping — which is what elja's
      ``openai`` provider builds, and so the likeliest one here — compares the part
      alone with no message fallback, so this is the more conservative of the two
      rules rather than a universal one. The precedence is one-way: a part that
      *does* name a provider keeps it, even inside a message naming a different one.
      Reversing the two would call a native part foreign whenever the enclosing
      message came from elsewhere.
    - Provider *families* alias. ``GoogleModel`` accepts ``google`` and
      ``google-gla`` interchangeably, likewise ``google-vertex`` and
      ``google-cloud``, so an in-family switch is listed here while the provider
      would have accepted it. Over-reporting is the safe direction for a function
      whose whole purpose is to make the host look.

    Args:
        messages: Caller-owned history.
        model: The model the history is about to be replayed to.

    Returns:
        Every thinking part attributed to a provider other than this model's.
    """
    return [
        part
        for message in messages
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, ThinkingPart)
        and (part.provider_name or message.provider_name) != model.system
    ]


async def run_with_deadline(
    agent: Agent[HostDeps, str | DeferredToolRequests],
    deps: HostDeps,
    prompt: str | None = None,
    *,
    recorder: TurnRecorder,
    seconds: float,
    history: Sequence[ModelMessage] = (),
    display: Callable[[str], None] | None = None,
    usage_limits: UsageLimits | None = None,
) -> tuple[str | DeferredToolRequests, list[ModelMessage]]:
    """Stop a turn on a deadline, keeping whatever it completed. elja adds nothing.

    This drives pydantic-ai's own cancellation channel rather than wrapping the
    call in ``asyncio.timeout``, and for a host the difference is the whole point:

    - A first-party cancellation arrives as ``RunCancelled``, an ordinary
      catchable outcome. The host can record it, persist, and answer the request.
    - An external ``asyncio`` cancellation must keep propagating for the enclosing
      timeout scope to unwind correctly, so a host on that path has to re-raise
      and dig the state back out of the exception chain with
      ``RunCancelled.from_cancellation``.
    - ``RunCancelled.all_messages()`` is **not** already resumable, and its own
      docstring's promise about synthesized interrupted returns does not cover the
      case that matters here. Measured: a tool stopped *inside its own body* leaves
      its call dangling in that snapshot, and replaying it with a new prompt raises
      ``UserError``. The snapshot closes out interrupted calls only where a sibling
      call in the same response already returned. So on this path too, persist
      ``recorder.partial`` — which has been through
      :func:`close_interrupted_calls` — and not the exception's own history.

    Either way ``recorder.partial`` holds the checkpoint, because the stop lands
    in the same place every other failure does.

    Args:
        agent: From :func:`build_host_agent`.
        deps: The host's dependencies for this run.
        prompt: The user's message, or ``None`` when resuming.
        recorder: The host's durable log.
        seconds: Wall-clock budget for the whole turn.
        history: Prior messages, caller-owned.
        display: Optional UI sink.
        usage_limits: The host's ceilings for this run.

    Raises:
        RunCancelled: If the deadline passes first.
    """
    token = CancellationToken()
    stop = asyncio.get_running_loop().call_later(seconds, token.cancel)
    try:
        return await run_turn(
            agent,
            deps,
            prompt,
            history=history,
            recorder=recorder,
            display=display,
            usage_limits=usage_limits,
            cancellation_token=token,
        )
    finally:
        stop.cancel()
