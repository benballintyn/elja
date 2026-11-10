"""Context compaction: evidence-based defaults, built on pydantic-ai-harness.

Strategy (see the project's compaction research, 2026-08):

1. **Observation masking first** (``ClearToolResults``): old tool results are
   replaced with a placeholder while every action and reasoning step is kept.
   This is the technique with the strongest published evidence — masking wins
   or ties LLM summarization on task success at roughly half the cost
   (JetBrains "Complexity Trap", arXiv:2508.21433; PNNL condenser ablation),
   is deterministic, and adds no hallucination channel. The placeholder tells
   the model results are recoverable by re-running the tool (elja's built-in
   tools are all idempotent reads or re-runnable commands).
2. **Structured summarization only as terminal fallback** — fixed-interval
   whole-history summarization measurably causes trajectory elongation and
   execution instability, so it fires only when masking can't reach the
   target. The harness summarizer pins the first user message (constraint
   dropping is the documented failure mode: 0% → 30% policy violations) and
   updates incrementally rather than rewriting wholesale.

``instructions`` and the deferred-skills catalog live outside message history in
pydantic-ai, so they are never subject to compaction. A **system prompt** is a
different matter and the distinction is load-bearing: ``Agent(system_prompt=...)``
puts a ``SystemPromptPart`` *into* the first request, i.e. into history, where
compaction does see it and has to keep it (see "protected content" below).
Skill BODIES, however, are loaded by a ``load_capability`` exchange inside
history. On harness 0.59-0.61 a summary that drops the exchange silently unloads
the skill (the catalog survives, so the model can re-load it); from 0.62 the
harness records the load again after the summary and the skill stays loaded. The
summary prompt is extended with a note that holds either way: re-load a skill
whose instructions are no longer in view.

**Composing this for a host application.** Every piece of the policy above is
an argument, because the defaults are right for a local workspace and wrong for
a server:

- ``cleared_placeholder`` replaces what a masked tool result says. The default
  invites the model to *re-run the tool*, which is safe only because elja's own
  tools are idempotent reads. A host whose tools have side effects must pass its
  own text — "retrieve the saved result" — or the masking tier is an invitation
  to double-write.
- ``summary_prompt`` replaces the summarization instruction, and
  ``summarizer_model`` the model that writes it.
- ``receipts`` leaves a deterministic note where history was summarized away, so
  the model knows its memory of earlier work is secondhand. With a capability
  implementing the harness's ``TranscriptHandleProvider`` protocol attached, the
  receipt also carries a handle for the persisted transcript.
- ``on_unfit`` decides what happens to a request compaction cannot bring under
  the target: ``None`` raises :class:`TargetUnreachableError`, a gate that returns
  sends it. See "The summarizer is bounded" below.
- Replacing the policy wholesale is always available: pass your own
  ``BoundedTieredCompaction`` with your own tiers as ``capabilities=`` instead of
  calling this factory. Upstream's plain ``TieredCompaction`` works too, and has no
  bound.

**Ordering matters, and it is list order.** None of these capabilities declare
an ordering, so ``ReportContextUsage`` measures whatever the capabilities before
it produced. Placed *after* compaction it reports the request that was actually
sent; placed before, it reports one that never existed. Measured on the
reporting-order test's own config: reporting after compaction read ~1k tokens,
before it ~6k, for the same run — a six-fold difference in what the host is
shown. Put reporting last.

**What survives, verified rather than assumed.** Pinned parts
(``pydantic_ai_harness.compaction.pin``) survive every tier, because the
orchestrator re-injects them after each one. Tool call/result pairing stays valid
across both tiers. The current request, the first user message (when preserved),
leading system policy and every part's own metadata survive too — on the path where
compaction fits the target and on the one where it cannot.

**The summarizer is bounded, and "cannot fit" is typed.** Some content no tier may
remove: instructions and tool definitions, leading system policy, the first user
message (``preserve_first_user_message``, on by default because dropping the
original task is how a long run forgets what it was asked to do), pins, and the
current request. Upstream re-injects or preserves all of it *after* the tail is
trimmed, so ``keep_tokens`` cannot bound it, and when it alone exceeds
``target_tokens`` the estimate can never reach target. Before 0.6 the summarizing
tier then fired again on every model request for the rest of the run — six paid
calls over a six-step turn, one-to-one and unbounded. :class:`BoundedTieredCompaction`
replaces that with four rules:

- at most ``CompactionConfig.max_summarizer_requests`` summarizer requests per run,
  counted at the model so a retry inside the summarizer counts too;
- no summary is paid for when that protected floor alone exceeds the target —
  measured first, reported as ``protected_floor``;
- a summary that ends over target latches the run into overflow mode, where the
  summarizer runs again only once the summarizable remainder exceeds a whole target
  — reported as ``no_progress``;
- every request that would go out over target is put to ``on_unfit`` with a
  :class:`TargetUnreachable` report. With no gate that raises
  :class:`TargetUnreachableError`, so nothing is sent over target unless the host
  said so. ``elja chat`` passes :func:`proceed_over_target`, a status line and send.

All of that state is per run; a new run starts fresh. The target is a *soft*
compaction target, never the provider's input limit — the report carries the
model's window separately, when it can be resolved.

**Protected content no longer grows.** A ``SystemPromptPart`` in caller-owned
history (what ``Agent(system_prompt=...)`` produces) used to multiply: the harness
copies leading system parts into each new summary message, first-message
preservation keeps the original too, and pydantic-ai merges consecutive requests on
replay, so each compacting turn left one more policy copy and one more stale summary
for the next to preserve. elja now removes exactly those copies after each summary
(:func:`_drop_compaction_copies`) and never touches a user-authored message. The same
merge folds each compacted turn's prompt into the preserved first request. Through
harness 0.61 those prompts stay there, as real messages, and the floor measurement
counts them; from 0.62 the harness summarizes what follows a summary in that
request, and the floor leaves it out.

Remedies when the floor is the problem, in order of preference. **Carry policy in
``instructions=``, not ``system_prompt=``** — it is still protected (it is sent on
every request), but it lives outside history, so nothing about it can accumulate.
(elja's own paths use ``instructions=``.) For a large first message, raise
``target_tokens`` or pass ``preserve_first_user_message=False`` and carry the task in
the instructions. For a pin, keep pinned sets small; a pin is the host's own choice
of what is irreducible.
"""

import inspect
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from functools import cache
from string import Formatter
from typing import Any, Literal, TypeGuard, TypeVar

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import AgentRunError
from pydantic_ai.messages import (
    CompactionPart,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    SystemPromptPart,
    TextContent,
    UserPromptPart,
)
from pydantic_ai.models import Model, ModelRequestContext, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage
from pydantic_ai_harness.compaction import (
    ClearToolResults,
    SummarizingCompaction,
    TieredCompaction,
    estimate_context_tokens,
    estimate_token_count,
    is_pinned,
    reinject_pinned,
    resolve_context_window,
)

from elja.deps import notify
from elja.settings import EljaSettings

CLEARED_PLACEHOLDER = (
    "[old tool result cleared to save context; re-run the tool if you need it again "
    "(large outputs may also be preserved under .elja/spill/)]"
)

# The harness's structured summary prompt, extended with a skills warning. A skill
# is loaded by a load_capability exchange in history: on harness 0.59-0.61 a summary
# that drops it unloads the skill, while 0.62+ records the load again after the
# summary. The summarizer cannot tell which, so the wording holds for both.
_SKILLS_WARNING = (
    "If any skills were loaded via load_capability in the conversation, name them under "
    "'## Open questions' and note that a skill whose instructions no longer appear in the "
    "conversation must be re-loaded via load_capability before use.\n\n"
)
_ANCHOR = "<messages>"
_FIELD = "messages"
_PLACEHOLDER = f"{{{_FIELD}}}"
# Two stand-in transcripts, rendered separately, to prove the prompt substitutes the
# transcript AT ALL. Each rendering must contain its own stand-in, so a prompt that
# drops the placeholder, escapes it, or merely mentions one stand-in's text is refused.
# Short on purpose: inferring *truncation* from a stand-in's length is what this check
# used to try, and it only ever moved the boundary — first to 23 characters, then to
# the stand-in's own length. Width is decided by `_substitutes_whole_transcript`
# instead, which reads the format spec rather than guessing from a sample.
_SENTINELS = ("elja-transcript-stand-in-alpha", "elja-transcript-stand-in-beta")

# The standard format-spec grammar for a string, so a precision can be told from a fill
# character that happens to be a dot (`{messages:.<40}` pads; `{messages:.40}` truncates).
_FORMAT_SPEC = re.compile(r"^(?:.?[<>=^])?[-+ ]?#?0?\d*[,_]?(?:\.(?P<precision>\d+))?[a-zA-Z%]?$")


def _substitutes_whole_transcript(summary_prompt: str) -> bool:
    """Whether every ``messages`` reference substitutes the transcript entire.

    Width-independent, which is the point: a precision (``{messages:.2000}``) or an
    index or attribute access (``{messages[0]}``, ``{messages.upper}``) hands the
    summarizer a fragment, and its output still replaces every message before the
    cutoff. Two earlier versions tried to infer this by rendering a sample and looking
    at the result, and each only relocated the boundary — to 23 characters, then to
    the sample's own length, at which point ``{messages:.10000}`` was accepted. Reading
    the spec settles it at every width.
    """
    for _, field_name, format_spec, _ in Formatter().parse(summary_prompt):
        if field_name is None:
            continue
        root = field_name.split(".", 1)[0].split("[", 1)[0]
        if root != _FIELD:
            continue
        if field_name != _FIELD:
            return False
        spec = _FORMAT_SPEC.match(format_spec or "")
        if spec is None or spec.group("precision") is not None:
            return False
    return True


def extend_summary_prompt(harness_default: str) -> str:
    """Insert elja's skills warning into the harness's summary prompt.

    A skill is loaded by a ``load_capability`` exchange inside history. On harness
    0.59-0.61 a summary that drops the exchange silently unloads the skill; the
    catalog survives, so the model can re-load it, if it is told. From 0.62 the
    harness records the load again after the summary, so the warning only asks for a
    re-load where the instructions are gone.

    Args:
        harness_default: The harness's own ``summary_prompt`` default.

    Returns:
        The prompt with the warning inserted before the transcript.

    Raises:
        RuntimeError: If the anchor is absent. A plain ``str.replace`` would
            no-op silently and drop the warning, and the harness pin is a range,
            so a patch release could reword the prompt. Failing loudly beats
            shipping a prompt that quietly lost its warning — at first use rather
            than at import: see ``default_summary_prompt``.
    """
    if _ANCHOR not in harness_default:
        raise RuntimeError(
            f"the harness summary prompt no longer contains the {_ANCHOR!r} anchor, "
            "so elja's skills warning cannot be inserted; update elja.compaction"
        )
    return harness_default.replace(_ANCHOR, f"{_SKILLS_WARNING}{_ANCHOR}")


@cache
def default_summary_prompt() -> str:
    """elja's summary prompt: the harness default plus the skills warning.

    Computed on first use rather than at import, so a reworded harness prompt
    fails the caller who needs compaction instead of making ``import elja`` fail
    for a host that never touches it. The harness pin is a range, so a patch
    release can trigger it.

    Checked the same way a caller's prompt is. ``extend_summary_prompt`` guards the
    ``<messages>`` anchor it needs to insert the warning, which is a different
    string from the ``{messages}`` placeholder the summarizer substitutes — a
    patch release could keep the anchor and rename the variable, and then every
    convenience-path user would get a default prompt that substitutes nothing.

    Raises:
        ValueError: If the harness default no longer substitutes a transcript.
        RuntimeError: If it no longer carries the anchor.
    """
    prompt = extend_summary_prompt(
        str(inspect.signature(SummarizingCompaction.__init__).parameters["summary_prompt"].default)
    )
    check_summary_prompt(prompt)
    return prompt


def check_summary_prompt(summary_prompt: str) -> None:
    """Refuse a summary prompt the summarizer cannot use, at construction time.

    ``SummarizingCompaction`` does ``self.summary_prompt.format(messages=...)`` and
    nothing else. ``str.format`` no-ops on a string with no placeholder, so a prompt
    that does not *substitute* hands the summarizer an instruction with no transcript —
    and its output still **replaces every message before the cutoff**. The run
    completes, nothing logs, and the history is gone. A stray single brace raises at the
    first summarization instead, deep in a conversation.

    Two checks, because one test cannot settle both halves, and trying has put a hole in
    this function three times:

    - **Did the transcript come out at all?** The prompt is rendered with two different
      stand-in transcripts and each rendering must contain its own. That refuses a
      missing placeholder, an escaped one (``{{messages}}`` is a literal, reached by
      obeying this function's own advice to double literal braces), and a prompt that
      merely mentions one stand-in's text.
    - **Did the whole transcript come out?** :func:`_substitutes_whole_transcript` reads
      the format spec. Inferring this from a rendered sample is what failed twice: it
      only ever moved the boundary, first to 23 characters and then to the sample's own
      length, at which point ``{messages:.10000}`` was accepted and the summarizer
      rewrote the history from a fragment.

    Permissive where it should be: ``{messages!r}``, ``{messages!s}``, ``{messages!a}``,
    padding and centring (``{messages:>10}``, ``{messages:.<40}`` — a dot there is a
    fill character, not a precision), literal braces either side, and the placeholder
    twice over all substitute in full and are accepted.

    Args:
        summary_prompt: The caller's prompt.

    Raises:
        ValueError: If the prompt does not substitute the whole transcript, or a field
            cannot be resolved. Every failure arrives as ``ValueError``: ``str.format``
            raises at least five different types on a bad field (``KeyError``,
            ``IndexError``, ``ValueError``, ``AttributeError``, ``TypeError``), and a
            caller guarding its construction path should not have to enumerate them.
    """
    try:
        rendered = [summary_prompt.format(messages=transcript) for transcript in _SENTINELS]
    except Exception as exc:
        raise ValueError(
            f"summary_prompt contains a field str.format cannot resolve ({exc!r}); "
            "double any literal braces as {{ }} — but leave the "
            f"{_PLACEHOLDER!r} placeholder single. Unchecked this raises at the "
            "first summarization, deep in a conversation."
        ) from exc
    substituted = all(
        stand_in in rendering for stand_in, rendering in zip(_SENTINELS, rendered, strict=True)
    )
    # Both checks always run: `and` would short-circuit the spec read away for a prompt
    # that already fails containment, and the spec read is the half that is
    # width-independent.
    whole = _substitutes_whole_transcript(summary_prompt)
    if substituted and whole:
        return
    raise ValueError(
        "summary_prompt does not substitute the whole transcript. Either the "
        f"{_PLACEHOLDER!r} placeholder is missing, or it is escaped (a doubled "
        "{{messages}} is a literal, not a placeholder — leave this one single even "
        "when doubling the rest), or the field keeps only part of what it substitutes: "
        "a precision truncates ({messages:.2000} keeps two thousand characters of "
        "however many there are) and an index or attribute access replaces it "
        "({messages[0]}, {messages.upper}). Unchecked, the summarizer is handed an "
        "instruction with no transcript, or a fragment of one, and its output still "
        "replaces the history."
    )


# --- Bounded compaction ------------------------------------------------------------
#
# Upstream's `TieredCompaction` stops escalating when the history fits and otherwise
# returns what its last tier produced. Nothing remembers that the summarizing tier
# already failed, so a history whose irreducible part exceeds the target pays for a
# summary before every model request of the run. Everything below exists to make that
# impossible, without a second agent loop: it is the same orchestrator, with a budget,
# a prediction, a latch, and a typed way to say "this cannot fit".

DepsT = TypeVar("DepsT")

UnfitReason = Literal["protected_floor", "no_progress", "budget_exhausted"]
"""Why compaction could not bring a request under ``target_tokens``.

- ``protected_floor``: the content no tier may remove is itself larger than the target.
  Detected *before* any summarizer request is paid for.
- ``no_progress``: a summary was written this run and the request still does not fit, so
  the run is in overflow mode (see :class:`BoundedTieredCompaction`); or there is no
  summarizing tier and the free tiers cannot reach the target.
- ``budget_exhausted``: the run has spent its ``max_summarizer_requests``.
"""

# The harness's own markers, spelled out rather than imported from its private modules,
# so a harness patch that moves them cannot make `import elja` fail. Each one is pinned
# against the harness's real behavior in tests/test_compaction_bounds.py, so drift fails
# CI instead of quietly changing what counts as protected.
#
# A summary has two shapes. Through harness 0.61 it is a `SystemPromptPart` opening with
# `_SUMMARY_PREFIX`; from 0.62 it is a `CompactionPart` in a request of its own, marked
# with `_SUMMARY_METADATA`, and the older shape is still recognized in histories saved
# before then. elja recognizes both on every supported release.
_SUMMARY_PREFIX = "Summary of previous conversation:\n\n"
_SUMMARY_METADATA = "pydantic-ai-harness.compaction.summary.v1"
_RECEIPT_METADATA = "pydantic-ai-harness.compaction.receipt.v1"


@dataclass(frozen=True)
class TargetUnreachable:
    """A typed report that compaction cannot bring this request under its target.

    **This is about a soft target, not a hard limit.** ``target_tokens`` is elja's
    compaction target (``CompactionConfig.target_tokens``), chosen well below any real
    context window. Exceeding it costs prefill time and quality, not a refused request.
    The provider's input limit is ``window_tokens`` when pydantic-ai can resolve it, and
    elja never claims to know it otherwise. A host deciding whether a request is safe to
    send compares ``estimated_tokens`` with that, not with the target.

    Every number is an *estimate*, in the same units the target is compared against:
    provider-reported usage where the history carries it (``anchored``), the harness's
    characters-per-token heuristic where it does not.
    """

    reason: UnfitReason
    """Why the target cannot be met. See :data:`UnfitReason`."""

    target_tokens: int
    """The soft target compaction was aiming for."""

    estimated_tokens: int
    """The request's estimate when compaction gave up on it.

    For ``protected_floor`` this is measured before the summarizing tier, because the
    report is made before paying for a summary. A host that returns from
    ``on_unfit`` may still get one summary that shrinks the request further, so the
    request actually sent is at most this; a ``ReportContextUsage`` placed after
    compaction reads the exact figure.
    """

    protected_tokens: int
    """What no tier may remove: instructions and tool definitions, leading system
    policy, the first user message when it is preserved, pins, and the current
    request. A summary cannot make the request smaller than this."""

    summarizer_requests: int
    """Summarizer requests dispatched in this run. One refused at the budget is not
    counted: it was never sent, so it cost nothing."""

    max_summarizer_requests: int
    """The run's budget (``CompactionConfig.max_summarizer_requests``)."""

    occurrence: int
    """1 for the first unreachable report in a run, 2 for the next, and so on. A gate
    that should act once per run (a status line, a log entry) checks for 1."""

    anchored: bool
    """Whether the estimates rest on provider-reported usage rather than the heuristic."""

    window_tokens: int | None
    """The request model's context window, when pydantic-ai can resolve it; else None."""

    def describe(self) -> str:
        """One line for a log or a status display."""
        why = {
            "protected_floor": (
                f"its protected content alone is ~{self.protected_tokens} tokens "
                "(instructions, system policy, the first user message, pins and the "
                "current request)"
            ),
            "no_progress": "a summary this run could not bring it under",
            "budget_exhausted": (
                f"this run has spent its {self.max_summarizer_requests} summarizer requests"
            ),
        }[self.reason]
        window = (
            f"; the model's window is {self.window_tokens}"
            if self.window_tokens is not None
            else ""
        )
        return (
            f"context ~{self.estimated_tokens} tokens is over the compaction target of "
            f"{self.target_tokens}: {why}{window}"
        )


class TargetUnreachableError(AgentRunError):
    """Raised before a request is sent over the target, when no ``on_unfit`` gate is set.

    The default, so nothing is sent over target without the host deciding it should
    be. Carries the :class:`TargetUnreachable` report as ``report``; read that, not
    the message.
    """

    def __init__(self, report: TargetUnreachable) -> None:
        """Wrap a report."""
        self.report = report
        super().__init__(report.describe())


OnUnfit = Callable[[RunContext[DepsT], TargetUnreachable], Awaitable[None] | None]
"""A host's gate for a request compaction cannot fit: raise to stop, return to send."""


async def proceed_over_target(ctx: RunContext[Any], report: TargetUnreachable) -> None:
    """The convenience path's gate: say so once per run, then send the request.

    Used by ``elja chat`` and by configured sub-agents, whose target is a conservative
    fraction of a local model's window. It reports through ``deps.on_status`` when the
    deps have one, which is where the CLI shows tool activity, and is otherwise silent
    — so it is NOT the right gate for a host that must account for every over-target
    request. Such a host passes its own, or none, to get :class:`TargetUnreachableError`.
    """
    if report.occurrence == 1:
        notify(getattr(ctx.deps, "on_status", None), report.describe())


@dataclass
class _RunState:
    """Everything the bound remembers, and only for one run."""

    spent: int = 0
    overflow: bool = False
    reports: int = 0


class _SummarizerBudgetSpentError(Exception):
    """A summarizer request refused at the budget, before it reached the host's guard."""


class _MeteredSummarizer(WrapperModel):
    """Counts every request the summarizer makes, and refuses one past the budget.

    The bound has to be on *requests*, not on summarization attempts: a summarizer
    reply with no text makes pydantic-ai's private summary agent retry once, so one
    attempt can be two paid requests. Wrapping the model the summarizer was going to
    use anyway catches that retry and anything else upstream might add. It sits
    *outside* the host's guard and delegates to it, so the guard still admits every
    dispatch that happens, and a refused one never reaches the guard at all.
    ``WrapperModel`` forwards settings, profile and request preparation, so what the
    summarizer sends is unchanged.
    """

    def __init__(self, wrapped: Model, state: _RunState, limit: int) -> None:
        """Meter ``wrapped`` against the run's ``state``."""
        super().__init__(wrapped)
        self._state = state
        self._limit = limit

    def _admit(self, *, spend: bool) -> None:
        if self._state.spent >= self._limit:
            raise _SummarizerBudgetSpentError
        if spend:
            self._state.spent += 1

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Spend one request from the budget, then delegate."""
        self._admit(spend=True)
        return await super().request(messages, model_settings, model_request_parameters)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        """The streamed path spends from the same budget."""
        self._admit(spend=True)
        async with super().request_stream(
            messages, model_settings, model_request_parameters, run_context
        ) as stream:
            yield stream

    async def count_tokens(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> RequestUsage:
        """A pre-request count is refused once the budget is spent, and costs nothing."""
        self._admit(spend=False)
        return await super().count_tokens(messages, model_settings, model_request_parameters)


def _is_legacy_summary(part: ModelRequestPart) -> TypeGuard[SystemPromptPart]:
    """A summary written through harness 0.61, recognized by the prefix it opens with."""
    return isinstance(part, SystemPromptPart) and part.content.startswith(_SUMMARY_PREFIX)


def _is_marked_summary(part: ModelRequestPart) -> TypeGuard[CompactionPart]:
    """A summary written by harness 0.62 or later, as the harness's ``is_summary_part`` sees it.

    The marker is in the part's metadata; a provider's own compaction carries provider
    data instead and is never one. ``getattr``: ``CompactionPart`` has no ``metadata``
    before pydantic-ai 2.62, and no summary of this shape exists there.
    """
    if not isinstance(part, CompactionPart):
        return False
    metadata = getattr(part, "metadata", None)
    return (
        isinstance(metadata, dict)
        and metadata.get(_SUMMARY_METADATA) is True
        and part.provider_name is None
        and part.id is None
        and part.provider_details is None
        and bool(part.content)
    )


def _is_summary(part: ModelRequestPart) -> TypeGuard[SystemPromptPart | CompactionPart]:
    """A summary the harness wrote, in either shape."""
    return _is_legacy_summary(part) or _is_marked_summary(part)


def _summary_text(part: SystemPromptPart | CompactionPart) -> str:
    """What a summary says, without the prefix the older shape opens with."""
    return (part.content or "").removeprefix(_SUMMARY_PREFIX)


def _is_receipt(part: ModelRequestPart) -> bool:
    """A compaction receipt, recognized by the metadata the harness writes it with."""
    return (
        isinstance(part, UserPromptPart)
        and not isinstance(part.content, str)
        and any(
            isinstance(item, TextContent) and item.metadata == _RECEIPT_METADATA
            for item in part.content
        )
    )


def _is_harness_marker(part: ModelRequestPart) -> bool:
    """A pin or a compaction receipt: user-role bookkeeping, not a user turn."""
    return is_pinned(part) or _is_receipt(part)


def _leading_system_parts(messages: list[ModelMessage]) -> list[SystemPromptPart]:
    """The system parts the summarizer copies into its summary message.

    Mirrors the harness's ``_extract_system_prompts``: leading requests only, skipping
    pins and receipts, stopping at the first summary or ordinary part.
    """
    parts: list[SystemPromptPart] = []
    for message in messages:
        if not isinstance(message, ModelRequest):
            break
        for part in message.parts:
            if isinstance(part, SystemPromptPart) and not _is_summary(part):
                parts.append(part)
            elif not _is_harness_marker(part):
                return parts
    return parts


def _first_user_message(messages: list[ModelMessage]) -> ModelRequest | None:
    """The request ``preserve_first_user_message`` keeps, as the harness finds it."""
    for message in messages:
        if isinstance(message, ModelRequest) and any(
            isinstance(part, UserPromptPart) and not _is_harness_marker(part)
            for part in message.parts
        ):
            return message
    return None


def _kept_first_message_parts(
    message: ModelRequest, leading: list[SystemPromptPart]
) -> list[ModelRequestPart]:
    """What of the first user message the summarizer keeps, as the harness decides it.

    Mirrors the harness's ``_kept_first_user_message``. Receipts, summaries and copies of
    the leading system parts are dropped, on every supported release. From harness 0.62
    the request is also cut at the first summary after the user's turn: pydantic-ai
    merges consecutive requests when a history is replayed, so what follows that summary
    belongs to later turns, and the new summary covers them. No summary of that shape
    exists before 0.62, so there the cut never applies.
    """
    carried = {part.content for part in leading}
    parts = message.parts
    turn = next(
        (
            index
            for index, part in enumerate(parts)
            if isinstance(part, UserPromptPart) and not _is_harness_marker(part)
        ),
        0,
    )
    end = next(
        (index for index in range(turn, len(parts)) if _is_marked_summary(parts[index])),
        len(parts),
    )
    return [
        part
        for part in parts[:end]
        if not _is_receipt(part)
        and not _is_summary(part)
        and not (isinstance(part, SystemPromptPart) and part.content in carried)
    ]


def _protected_floor(
    messages: list[ModelMessage], *, preserve_first_user_message: bool
) -> list[ModelMessage]:
    """What survives every tier, as messages the estimator can measure.

    A lower bound on any summarized result, not a guess: the summarizer keeps the
    leading system parts, the first user message when preserving it, every pin (the
    orchestrator re-injects them) and at least the last message, and adds a summary
    on top. Each part is counted once — copies collapse, and superseded summaries are
    dropped by :func:`_drop_compaction_copies` — and the latest instructions are
    carried so the estimator counts them as it does for the real request.
    """
    first = _first_user_message(messages) if preserve_first_user_message else None
    last = messages[-1]
    leading = _leading_system_parts(messages)
    candidates: list[ModelRequestPart] = [
        *leading,
        *(_kept_first_message_parts(first, leading) if first is not None else []),
        *(
            part
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if is_pinned(part)
        ),
        *(last.parts if isinstance(last, ModelRequest) else []),
    ]
    kept: list[ModelRequestPart] = []
    for candidate in candidates:
        if not _is_summary(candidate) and not any(candidate == other for other in kept):
            kept.append(candidate)
    instructions = next(
        (
            message.instructions
            for message in reversed(messages)
            if isinstance(message, ModelRequest) and message.instructions
        ),
        None,
    )
    floor: list[ModelMessage] = [ModelRequest(parts=kept, instructions=instructions)]
    # A response last only happens on the manual path; it is kept verbatim there too.
    return [*floor, last] if isinstance(last, ModelResponse) else floor


def _capped_tail_start(
    messages: list[ModelMessage], keep_tokens: int, tokenizer: Callable[[str], int] | None
) -> int:
    """Where the longest tail of at most ``keep_tokens`` begins, keeping at least one message.

    How harness 0.27 read ``keep_tokens``. Every harness this release supports (0.59+)
    reads it as a *minimum* instead: the shortest tail holding at least that many tokens,
    which keeps one message past the bound however large it is, so a single big tool
    result could keep every request over target. :class:`BoundedTieredCompaction` hands
    the tier this tail as a message count, which the harness then moves earlier only to
    keep tool pairs whole, as 0.27 did.
    """
    if estimate_token_count(messages, tokenizer) <= keep_tokens:
        return 0
    low, high = 0, len(messages)
    while low < high:
        middle = (low + high) // 2
        if estimate_token_count(messages[middle:], tokenizer) <= keep_tokens:
            high = middle
        else:
            low = middle + 1
    return min(low, len(messages) - 1)


def _single_previous_summary(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Fold several earlier summaries into one, so the summarizer's anchor holds them all.

    Histories written before :func:`_drop_compaction_copies` existed carry one summary
    per compacting turn. The harness anchors an incremental summary on the newest and
    leaves every other summary out of the transcript it summarizes, so their content
    would be lost the first time copies are dropped. Folding them all, newest first,
    into the anchor's place makes it the only anchor. The newest is the last summary of
    the newer shape if there is one, else the first of the older shape, which each pass
    put ahead of the ones before it: the harness's own choice, in both shapes.
    """
    found: list[SystemPromptPart | CompactionPart] = [
        part
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if _is_summary(part)
    ]
    newest_first: list[SystemPromptPart | CompactionPart] = [
        *reversed([part for part in found if _is_marked_summary(part)]),
        *(part for part in found if _is_legacy_summary(part)),
    ]
    if len(newest_first) < 2:
        return messages
    anchor = newest_first[0]
    text = "\n\n".join(_summary_text(part) for part in newest_first)
    folded = replace(
        anchor, content=text if isinstance(anchor, CompactionPart) else _SUMMARY_PREFIX + text
    )
    result: list[ModelMessage] = []
    for message in messages:
        if isinstance(message, ModelRequest) and any(_is_summary(p) for p in message.parts):
            parts = [
                folded if part is anchor else part
                for part in message.parts
                if not _is_summary(part) or part is anchor
            ]
            if not parts:
                # A summary of the newer shape is a request of its own.
                continue
            message = replace(message, parts=parts)
        result.append(message)
    return result


def _drop_compaction_copies(
    before: list[ModelMessage], after: list[ModelMessage]
) -> list[ModelMessage]:
    """Remove what the summarizing tier copied, so protected content cannot grow.

    The harness copies every leading system part into its new summary message, and
    ``preserve_first_user_message`` keeps the original first request too. pydantic-ai
    merges consecutive requests when a history is replayed, so on the next turn that
    "first request" also holds the previous summary and the previous copy — and the
    harness preserves all of it again. Each compaction left one more policy copy and
    one more summary for the next to find.

    Two rules, both limited to content compaction itself manufactured:

    - **A system part appearing twice is a copy.** Only the first occurrence is kept,
      which is the one leading the summary message, so policy still opens the
      conversation. Matched by exact equality — content, timestamp and dynamic ref —
      so two system parts that merely say the same thing are both kept.
    - **A summary this compaction did not write is superseded.** The new summary was
      written from it (as the incremental anchor, or from the transcript), so it is
      dropped rather than preserved alongside its replacement.

    User prompts, pins, receipts, tool parts and every message's own metadata are
    never touched. A message left with no parts carried only copies and is dropped.
    """
    seen = {id(part) for message in before for part in message.parts}
    if not any(
        _is_summary(part) and id(part) not in seen
        for message in after
        if isinstance(message, ModelRequest)
        for part in message.parts
    ):
        return after
    policy: list[SystemPromptPart] = []
    result: list[ModelMessage] = []
    for message in after:
        if not isinstance(message, ModelRequest):
            result.append(message)
            continue
        parts: list[ModelRequestPart] = []
        for part in message.parts:
            if _is_summary(part) and id(part) in seen:
                continue
            if isinstance(part, SystemPromptPart) and not _is_summary(part):
                if any(part == other for other in policy):
                    continue
                policy.append(part)
            parts.append(part)
        if len(parts) == len(message.parts):
            result.append(message)
        elif parts:
            result.append(replace(message, parts=parts))
    return result


@dataclass
class BoundedTieredCompaction(TieredCompaction[DepsT]):
    """``TieredCompaction`` that cannot pay for summaries without limit.

    The tiers run exactly as upstream's do, cheap to expensive, stopping once the
    request fits ``target_tokens``. Four things are added, all of them about the
    summarizing tier — the one that makes a paid request:

    1. **A budget.** At most ``max_summarizer_requests`` summarizer requests per run,
       counted at the model, so a hidden retry inside the summarizer counts too and is
       refused before dispatch once the budget is spent.
    2. **A prediction.** Before paying, the protected floor — what no tier may remove,
       see :attr:`TargetUnreachable.protected_tokens` — is measured. If it alone
       exceeds the target, a summary cannot help, and the gate is consulted before
       anything is spent.
    3. **A latch.** A summary that ends over the target puts the run in *overflow
       mode*: further summaries happen only when the summarizable remainder (estimate
       minus floor) exceeds a whole ``target_tokens`` — real progress, not a re-summary
       of the summary on every request. A later request that compacts back under the
       target leaves overflow mode. The floor-exceeded case behaves the same way.
    4. **A gate.** Every request that would go out over the target is put to
       ``on_unfit`` with a :class:`TargetUnreachable` report. ``None`` (the default)
       raises :class:`TargetUnreachableError`; a gate that returns lets the request go
       as compacted, and a gate that raises stops the run with its own exception.

    **Scope of the state.** All of it is per run: ``for_run`` hands each run a fresh
    copy, so the budget, the latch and the report count reset when a new run starts,
    and two concurrent runs of one agent cannot see each other's. A manual
    ``compact_now`` call is its own run. Nesting this as a tier of another
    orchestrator makes each of its compactions a run.

    After each summary, copies the harness manufactured are removed (see
    :func:`_drop_compaction_copies`), so protected content does not grow turn by turn.

    Only ``SummarizingCompaction`` tiers given directly are metered, and their
    ``model`` must be ``None`` (inherit the request's model) or a ``Model`` instance:
    a model *name* would make the summarizer build its own client, outside the bound.

    A metered tier's ``keep_tokens`` caps its verbatim tail (see
    :func:`_capped_tail_start`), as it did on harness 0.27; the harness itself now reads
    it as a minimum. A tier with ``keep_user_messages`` keeps the harness's reading,
    since its retention rules are built around it.
    """

    max_summarizer_requests: int = field(default=3, kw_only=True)
    """Paid summarizer requests allowed per run. 0 means masking only."""

    on_unfit: OnUnfit[DepsT] | None = field(default=None, kw_only=True)
    """Gate for a request compaction cannot fit. ``None`` raises."""

    _state: _RunState = field(default_factory=_RunState, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate upstream's fields, then this class's own."""
        super().__post_init__()
        if self.max_summarizer_requests < 0:
            raise ValueError("max_summarizer_requests must be non-negative.")
        for tier in self.tiers:
            if isinstance(tier, SummarizingCompaction) and isinstance(tier.model, str):
                raise ValueError(
                    "a summarizing tier's model must be None or a Model instance, not the "
                    f"name {tier.model!r}: a name makes the summarizer build its own "
                    "provider client, outside both the host's guard and this bound."
                )

    async def for_run(self, ctx: RunContext[DepsT]) -> "BoundedTieredCompaction[DepsT]":
        """A fresh copy per run, so no run inherits another's budget or latch."""
        return replace(self)

    async def compact(
        self, messages: list[ModelMessage], ctx: RunContext[DepsT]
    ) -> list[ModelMessage]:
        """The manual path (``compact_now``, or use as a tier): one call, one run."""
        run = replace(self)
        target = run._target(ctx.model)
        return messages if target is None else await run._escalate_bounded(messages, ctx, target)

    async def before_model_request(
        self, ctx: RunContext[DepsT], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        """Upstream's hook, driving the bounded escalation instead of its own.

        Overridden whole rather than by replacing upstream's private ``_escalate``: if a
        harness release renamed that, an override would silently stop being called and
        the bound with it. The helpers imported here are what keep receipts, the
        ``compact_messages`` span and ``ReportContextUsage``'s correction identical to
        upstream's; a rename of any of them fails at the first compaction, loudly, and
        not at ``import elja``.
        """
        from pydantic_ai_harness.compaction._shared import (
            compact_with_span,
            context_for_request,
            record_compaction_reclaim,
        )

        # The run's history, as upstream reads it. On every supported pydantic-ai a hook's
        # `request_context.messages` is, during the before-chain, a HistoryMirroringMessages:
        # assigning it changes only the request, while `append`/`extend` still write to
        # history (deprecated). So the history to compact and to write back is
        # `ctx.messages`.
        messages: list[ModelMessage] = list(ctx.messages)
        request_ctx = context_for_request(ctx, request_context)
        target = self._target(request_ctx.model)
        parameters = request_context.model_request_parameters
        if target is None or (
            estimate_context_tokens(messages, self.tokenizer, model_request_parameters=parameters)
            <= target
        ):
            return request_context
        compacted = await compact_with_span(
            request_ctx,
            # Upstream's name, so a dashboard filtering compaction spans keeps working.
            strategy="TieredCompaction",
            messages=messages,
            compact=lambda: self._escalate_bounded(messages, request_ctx, target, parameters),
            tokenizer=self.tokenizer,
        )
        # Assigning `request_context.messages` no longer persists anything: the compacted
        # history is written to the run's own list, so the next request and the run's
        # result carry it, and the request goes out with it.
        ctx.messages[:] = compacted
        record_compaction_reclaim(
            ctx,
            estimate_token_count(messages, self.tokenizer),
            estimate_token_count(compacted, self.tokenizer),
        )
        return replace(request_context, messages=compacted)

    async def _escalate_bounded(
        self,
        messages: list[ModelMessage],
        ctx: RunContext[DepsT],
        target: int,
        parameters: ModelRequestParameters | None = None,
    ) -> list[ModelMessage]:
        """Upstream's escalation loop, with the budget, prediction, latch and gate."""
        original = messages
        # Measured the way upstream measures: the anchored baseline, minus the
        # heuristic reclaim of what the tiers removed. The floor is measured in the same
        # frame, so the overhead the heuristic cannot see (tool definitions in an
        # anchored count) is part of it — as it is part of every request.
        baseline = estimate_context_tokens(
            messages, self.tokenizer, model_request_parameters=parameters
        )
        offset = baseline - estimate_token_count(messages, self.tokenizer)

        def measure(candidate: list[ModelMessage]) -> int:
            return max(offset + estimate_token_count(candidate, self.tokenizer), 0)

        estimate = baseline
        protected: int | None = None
        reported = summarized = metered = False
        for tier in self.tiers:
            if estimate <= target:
                break
            if not isinstance(tier, SummarizingCompaction):
                compacted = await tier.compact(messages, ctx)
            else:
                metered = True
                protected = measure(
                    _protected_floor(
                        messages,
                        # Upstream: `keep_user_messages` supersedes first-message
                        # preservation, and keeps truncated copies instead.
                        preserve_first_user_message=tier.preserve_first_user_message
                        and not tier.keep_user_messages,
                    )
                )
                if protected > target:
                    await self._unreachable(
                        ctx, "protected_floor", target, estimate, protected, original
                    )
                    reported = True
                if (reported or self._state.overflow) and estimate - protected <= target:
                    continue
                if self._state.spent >= self.max_summarizer_requests:
                    continue
                compacted = await self._summarize(tier, messages, ctx)
                summarized = summarized or compacted is not messages
            messages = reinject_pinned(original, compacted)
            estimate = measure(messages)
        if estimate <= target:
            self._state.overflow = False
            return messages
        self._state.overflow = self._state.overflow or summarized
        if not reported:
            exhausted = metered and self._state.spent >= self.max_summarizer_requests
            reason: UnfitReason = (
                "budget_exhausted" if exhausted and not self._state.overflow else "no_progress"
            )
            if protected is None:
                protected = measure(_protected_floor(messages, preserve_first_user_message=False))
            await self._unreachable(ctx, reason, target, estimate, protected, original)
        return messages

    async def _summarize(
        self,
        tier: SummarizingCompaction[DepsT],
        messages: list[ModelMessage],
        ctx: RunContext[DepsT],
    ) -> list[ModelMessage]:
        """One metered summarizing pass, with manufactured copies removed after it.

        Returns ``messages`` itself when nothing was summarized — the tier found nothing
        before its cutoff, or its request was refused at the budget — so the caller can
        tell a pass that happened from one that did not.
        """
        if tier.model is not None:
            assert isinstance(tier.model, Model)  # names are refused in __post_init__
            tier = replace(
                tier,
                model=_MeteredSummarizer(tier.model, self._state, self.max_summarizer_requests),
            )
        elif isinstance(ctx.model, Model):
            ctx = replace(
                ctx,
                model=_MeteredSummarizer(ctx.model, self._state, self.max_summarizer_requests),
            )
        prepared = _single_previous_summary(messages)
        # With keep_tokens=None upstream also skips its "nothing to summarize" early
        # return and its protected_tail, both keyed on keep_tokens. So one summarizer
        # request may re-summarize only standing content (a summary, the kept first
        # message); the budget and the latch bound that, and harness 0.27 did the same.
        if tier.keep_tokens is not None and not tier.keep_user_messages:
            tier = replace(
                tier,
                keep_tokens=None,
                keep_messages=len(prepared)
                - _capped_tail_start(prepared, tier.keep_tokens, tier.tokenizer),
            )
        try:
            compacted = await tier.compact(prepared, ctx)
        except _SummarizerBudgetSpentError:
            return messages
        if compacted is prepared:
            return messages
        return _drop_compaction_copies(prepared, compacted)

    async def _unreachable(
        self,
        ctx: RunContext[DepsT],
        reason: UnfitReason,
        target: int,
        estimate: int,
        protected: int,
        original: list[ModelMessage],
    ) -> None:
        """Report to the gate, which raises to stop or returns to send."""
        self._state.reports += 1
        report = TargetUnreachable(
            reason=reason,
            target_tokens=target,
            estimated_tokens=estimate,
            protected_tokens=protected,
            summarizer_requests=self._state.spent,
            max_summarizer_requests=self.max_summarizer_requests,
            occurrence=self._state.reports,
            anchored=any(
                isinstance(message, ModelResponse) and bool(message.usage.input_tokens)
                for message in original
            ),
            window_tokens=resolve_context_window(ctx.model),
        )
        if self.on_unfit is None:
            raise TargetUnreachableError(report)
        outcome = self.on_unfit(ctx, report)
        if inspect.isawaitable(outcome):
            await outcome


def build_compaction(
    settings: EljaSettings,
    *,
    summarizer_model: Model | None = None,
    summarizer_model_settings: ModelSettings | None = None,
    cleared_placeholder: str | None = None,
    summary_prompt: str | None = None,
    receipts: bool = False,
    preserve_first_user_message: bool = True,
    on_unfit: OnUnfit[Any] | None = None,
) -> list[AbstractCapability[Any]]:
    """Build the compaction capability from settings (empty list if disabled).

    The capability is a :class:`BoundedTieredCompaction`: masking, then summarization,
    with at most ``settings.compaction.max_summarizer_requests`` paid summarizer
    requests per run, no summary paid for when the protected content alone exceeds
    the target, and no re-summarizing of a history that a summary already failed to
    fit. Its upstream type is unchanged, so ``isinstance(cap, TieredCompaction)`` holds.

    Args:
        settings: Resolved elja settings.
        summarizer_model: The model the summarization tier writes summaries
            with. ``None`` (the default, and the behavior before this argument
            existed) inherits the running agent's own model *object*, so a
            wrapper that meters or gates requests governs summarization too —
            verified in ``tests/test_metering.py``. Pass a model explicitly when
            summarization should use a different provider, a cheaper model, or
            its own separately-attributed guard. An instance is handed to the
            summarizer untouched, never rebuilt from its display name.

            Deliberately a ``Model``, not a model *name*: a name makes the
            summarizer build a fresh provider client from environment
            credentials, ignoring this config's ``base_url`` and bypassing
            anything the host wrapped — and an unknown one fails at the first
            summarization, deep in a long conversation.
        summarizer_model_settings: Agent-level settings for the summarizer's own
            request. These reach ``Model.request``/``request_stream`` as the
            ``model_settings`` argument, which is how **one** guard instance can
            tell a compaction request from a main one without a second model
            object: tag it, e.g.
            ``{"extra_headers": {"x-phase": "compaction"}}``. Note the merge is
            shallow, so an ``extra_headers`` here replaces whatever the model
            already carries, for the summarizer request only.
        cleared_placeholder: What a masked tool result is replaced with.
            ``None`` keeps :data:`CLEARED_PLACEHOLDER`, which tells the model to
            re-run the tool — correct for elja's idempotent built-ins, and an
            invitation to repeat a side effect for anything else. A host with
            write tools should pass text pointing at its own store instead.
        summary_prompt: Replaces the summarization instruction — the summarizer's
            *user* turn, not its system prompt (upstream's ``instructions``,
            which elja does not expose). ``None`` keeps elja's, which is the
            harness default plus a note about re-loading skills; a host with no
            skills has no reason to carry that note. Must SUBSTITUTE
            ``{messages}``; see :func:`check_summary_prompt` for why that is
            refused at construction rather than at first use.
        receipts: Leave a deterministic receipt where history was summarized
            away, so the model treats its memory of earlier work as secondhand.
            Off by default, as upstream has it. A caller-owned history carries one
            receipt however many compactions it has been through: each compaction
            replaces the previous receipt, so its dropped-message count describes
            the latest compaction only.
        preserve_first_user_message: Keep the run's original task verbatim through
            every tier. On by default, because dropping it is how a long run forgets
            what it was asked to do. It is protected content, so a first message
            bigger than ``target_tokens`` makes the target unreachable: elja detects
            that before paying for a summary and puts it to ``on_unfit`` with reason
            ``protected_floor``. (Before 0.6 it paid for a summary on every model
            request instead.) A host whose first message is a large document either
            raises the target or turns this off and carries the task in its own
            instructions, where compaction cannot reach it.
        on_unfit: The gate for a request compaction cannot bring under the target,
            called with the run context and a :class:`TargetUnreachable` report.
            Raise to stop the run; return to send the request as compacted. ``None``
            (the default) raises :class:`TargetUnreachableError`, so nothing goes out
            over target unless the host decided it should. The convenience path
            passes :func:`proceed_over_target`.

    Returns:
        A single tiered compaction capability, or ``[]`` when disabled.
    """
    if summary_prompt is not None:
        check_summary_prompt(summary_prompt)
    if cleared_placeholder is not None and not cleared_placeholder.strip():
        # Upstream assigns this string as given and does not fall back to its own
        # default, so an empty one leaves the model a cleared tool result with no cue
        # that anything was cleared — the masking tier's whole signal, silently gone.
        raise ValueError(
            "cleared_placeholder must not be empty: it replaces a tool result's "
            "content, and an empty one leaves the model no indication that anything "
            "was cleared. Pass text naming your own store, e.g. "
            "'[result cleared; retrieve it by its receipt id]'."
        )
    cfg = settings.compaction
    if not cfg.enabled:
        return []
    return [
        BoundedTieredCompaction(
            tiers=[
                # max_tokens=1 / max_messages=1 are always-eligible sentinels:
                # inside TieredCompaction the orchestrator's target_tokens is
                # the real trigger (per-tier TRIGGERS are bypassed; the keep_*
                # retention params below are honored).
                ClearToolResults(
                    max_tokens=1,
                    keep_pairs=cfg.keep_tool_pairs,
                    placeholder=(
                        CLEARED_PLACEHOLDER if cleared_placeholder is None else cleared_placeholder
                    ),
                ),
                SummarizingCompaction(
                    model=summarizer_model,
                    max_messages=1,
                    keep_messages=cfg.keep_messages,
                    # Token-bound the verbatim tail, so the ordinary tail can never be
                    # what keeps a request over target. What it cannot bound — pins,
                    # the first user message, system policy — is the protected floor,
                    # which BoundedTieredCompaction measures before paying. A cap only
                    # because BoundedTieredCompaction makes it one: the harness alone
                    # reads keep_tokens as a minimum.
                    keep_tokens=cfg.target_tokens // 3,
                    preserve_first_user_message=preserve_first_user_message,
                    incremental=True,
                    model_settings=summarizer_model_settings,
                    summary_prompt=(
                        default_summary_prompt() if summary_prompt is None else summary_prompt
                    ),
                    receipts=receipts,
                ),
            ],
            target_tokens=cfg.target_tokens,
            max_summarizer_requests=cfg.max_summarizer_requests,
            on_unfit=on_unfit,
        )
    ]
