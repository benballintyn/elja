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
compaction does see it — and duplicates it (see "irreducible content" below).
Loaded skill BODIES, however, travel as tool returns inside history: if the
summarization tier drops a load, the skill silently unloads (the catalog
survives, so the model can re-load it) — the summary prompt is extended to
call this out.

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
- Replacing the policy wholesale is always available: pass your own
  ``TieredCompaction``/``ClearToolResults``/``SummarizingCompaction`` as
  ``capabilities=`` instead of calling this factory at all.

**Ordering matters, and it is list order.** None of these capabilities declare
an ordering, so ``ReportContextUsage`` measures whatever the capabilities before
it produced. Placed *after* compaction it reports the request that was actually
sent; placed before, it reports one that never existed. Measured on the
reporting-order test's own config: reporting after compaction read ~1k tokens,
before it ~6k, for the same run — a six-fold difference in what the host is
shown. Put reporting last.

**What survives, verified rather than assumed.** Pinned parts
(``pydantic_ai_harness.compaction.pin``) survive every tier, because
``TieredCompaction`` re-injects them after each one. Tool call/result pairing
stays valid across both tiers. There is no upstream signal for "the target could
not be reached": nothing is silently dropped, but a host that needs to know
should compare a post-compaction ``ReportContextUsage`` reading against its own
target.

**Keep every piece of irreducible content well under the target.** This is the
one cost rule that matters, and it is about *any* content compaction cannot reduce
— not only a pin. If what survives every tier is itself larger than
``target_tokens``, the post-compaction estimate can never fall to target and the
summarizing tier fires again on **every** model request for the rest of the run,
one paid LLM call each. Measured over a six-step turn: one summarizer call in the
ordinary case, **six** when something irreducible exceeds target — one-to-one with
requests, and unbounded.

Three things are irreducible, and the ones a host trips over are the last two:

- **A pin.** Re-injection happens *after* the tail is trimmed, so ``keep_tokens``
  cannot bound it.
- **The first user message**, which elja preserves on the host's behalf —
  ``preserve_first_user_message``, on by default because dropping the original task
  is how a long run forgets what it was asked to do. A server-side agent whose first
  message is a task brief, a ticket body or a pasted document reaches this with no
  pin anywhere. Measured: a 20k-character first user message at
  ``target_tokens=1000`` gives six summarizer calls over six requests, and the same
  turn with ``preserve_first_user_message=False`` gives one.
- **A ``SystemPromptPart`` sitting in caller-owned history**, which is what
  ``Agent(system_prompt=...)`` produces. This one *grows*: the harness copies every
  leading system part into the summary message on each compaction, and
  ``preserve_first_user_message`` keeps the original request carrying it as well, so
  each compaction leaves one more copy for the next to find. Measured over five
  caller-owned turns with a 350-character policy: **1, 2, 3, 4, 5** copies and the
  history 1.1k → 4.4k characters, versus a flat **1** with
  ``preserve_first_user_message=False``. A policy that starts comfortably under
  target therefore reaches the unbounded regime by growth alone.

Remedies, in order of preference. **Carry policy in ``instructions=``, not
``system_prompt=``** — instructions live outside message history, so compaction never
sees them and nothing accumulates. (elja's own paths use ``instructions=``, which is
why the CLI never hits this.) Failing that, ``preserve_first_user_message=False``
stops the *growth* but does not help a single copy that is already above target; for
that, raise ``target_tokens`` above it. elja does not *bound* any of this; doing so
needs a latch that refuses to re-enter the summarizing tier once it has failed to
reach target, which is not built.
"""

import inspect
from functools import cache
from typing import Any

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings
from pydantic_ai_harness.compaction import (
    ClearToolResults,
    SummarizingCompaction,
    TieredCompaction,
)

from elja.settings import EljaSettings

CLEARED_PLACEHOLDER = (
    "[old tool result cleared to save context; re-run the tool if you need it again "
    "(large outputs may also be preserved under .elja/spill/)]"
)

# The harness's structured summary prompt, extended with a skills warning:
# loaded skill bodies travel inside history as tool returns, so a summary
# that drops the load silently unloads the skill.
_SKILLS_WARNING = (
    "If any skills were loaded via load_capability in the conversation, state under "
    "'## Open questions' that they are no longer loaded and must be re-loaded via "
    "load_capability before use.\n\n"
)
_ANCHOR = "<messages>"
_PLACEHOLDER = "{messages}"
# Two stand-in transcripts, rendered separately, to prove the prompt substitutes the
# WHOLE transcript. Requiring each rendering to contain its own stand-in verbatim is
# what catches a field that substitutes only part of it: a previous version compared
# the two renderings for difference, which let `{messages:.23}` through — 23 being the
# length at which the two stand-ins stopped agreeing. Any truncation is now refused,
# whatever its width, because a truncated rendering cannot contain the whole thing.
# Long and repetitive on purpose: the distinguishing token sits at the END, so no
# plausible width slices it off and accidentally satisfies containment.
_TRANSCRIPT_STAND_IN = "elja-transcript-stand-in " * 200
_SENTINELS = (_TRANSCRIPT_STAND_IN + "alpha", _TRANSCRIPT_STAND_IN + "beta")


def extend_summary_prompt(harness_default: str) -> str:
    """Insert elja's skills warning into the harness's summary prompt.

    Loaded skill bodies travel as tool returns inside history, so a summary that
    drops the load silently unloads the skill. The catalog survives, so the model
    can re-load it — if it is told.

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

        ``SummarizingCompaction`` does ``self.summary_prompt.format(messages=...)``
        and nothing else. ``str.format`` no-ops on a string with no placeholder, so a
        prompt that does not *substitute* hands the summarizer an instruction with no
        transcript — and its output still **replaces every message before the
        cutoff**. The run completes, nothing logs, and the history is gone. A stray
        single brace raises ``KeyError`` instead, at the first summarization, deep in
        a conversation.

    So this renders the prompt twice with two different stand-in transcripts and
        requires each rendering to contain its own stand-in **whole**, rather than looking
        for ``{messages}`` in the text. Those are not the same test, and the difference is
        a hole this function has had three times over:

        - ``{messages}`` is a substring of ``{{messages}}``, an escaped brace that renders
          as literal text and substitutes nothing — reached by obeying this very function's
          advice to double literal braces, since doubling all of them takes the placeholder
          with it.
        - Checking for one sentinel *by name* accepted a prompt that merely contained the
          sentinel's own text, with no placeholder at all.
        - Comparing two renderings for *difference* accepted ``{messages:.23}`` — the
          width at which the two stand-ins stopped agreeing — handing the summarizer 23
          characters of a 40k transcript while its output still replaced the history.

        Containment is the property that actually matters: the whole transcript has to come
        out. It stays permissive where it should — ``{messages!r}``, ``{messages!s}``,
        ``{messages:>10}`` and ``{{{messages}}}`` all substitute in full and are accepted —
        and refuses every field that keeps only part of it, at any width.

        Args:
            summary_prompt: The caller's prompt.

        Raises:
            ValueError: If the prompt does not substitute the transcript, or a brace
                cannot be resolved. Every failure arrives as ``ValueError``:
                ``str.format`` raises at least five different types on a bad field
                (``KeyError``, ``IndexError``, ``ValueError``, ``AttributeError``,
                ``TypeError``), and a caller guarding its construction path should not
                have to enumerate them.
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
    if all(
        stand_in in rendering for stand_in, rendering in zip(_SENTINELS, rendered, strict=True)
    ):
        return
    raise ValueError(
        "summary_prompt does not substitute the whole transcript: rendering it with a "
        "stand-in transcript did not produce that transcript. Either the "
        f"{_PLACEHOLDER!r} placeholder is missing, or it is escaped (a doubled "
        "{{messages}} is a literal, not a placeholder — leave this one single even "
        "when doubling the rest), or the field keeps only part of what it substitutes "
        "({messages[0]} keeps one character, {messages:.2000} keeps two thousand of "
        "however many there are). Unchecked, the summarizer is handed an instruction "
        "with no transcript, or a fragment of one, and its output still replaces the "
        "history."
    )


def build_compaction(
    settings: EljaSettings,
    *,
    summarizer_model: Model | None = None,
    summarizer_model_settings: ModelSettings | None = None,
    cleared_placeholder: str | None = None,
    summary_prompt: str | None = None,
    receipts: bool = False,
    preserve_first_user_message: bool = True,
) -> list[AbstractCapability[Any]]:
    """Build the compaction capability from settings (empty list if disabled).

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
            harness default plus a note about skills unloading; a host with no
            skills has no reason to carry that note. Must SUBSTITUTE
            ``{messages}``; see :func:`check_summary_prompt` for why that is
            refused at construction rather than at first use.
        receipts: Leave a deterministic receipt where history was summarized
            away, so the model treats its memory of earlier work as secondhand.
            Off by default, as upstream has it. Note one receipt accumulates per
            compaction across a long caller-owned history, each with its own
            dropped-message count — upstream's de-accumulation does not match
            once the receipt has been merged into a mixed-parts message.
        preserve_first_user_message: Keep the run's original task verbatim through
            every tier. On by default, because dropping it is how a long run forgets
            what it was asked to do. **Turn it off when that message is large.** It
            is irreducible content, so a first message bigger than
            ``target_tokens`` makes the summarizing tier fire on every model
            request for the rest of the run — one paid LLM call each, unbounded.
            Measured: a 20k-character first user message at ``target_tokens=1000``
            gives six summarizer calls over six requests; the same turn with this
            off gives one. A host turning it off should carry the task in its own
            instructions instead, where compaction cannot reach it.

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
        TieredCompaction(
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
                    # Token-bound the verbatim tail so the target is always
                    # reachable — otherwise an irreducible tail above target
                    # re-fires a summarizer LLM call on EVERY request.
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
        )
    ]
