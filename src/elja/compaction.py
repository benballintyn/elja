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

The system prompt/instructions and the deferred-skills catalog live outside
message history in pydantic-ai, so they are never subject to compaction.
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

**Keep a pinned set well under the target.** Re-injection happens *after* the
tail is trimmed, so ``keep_tokens`` cannot bound a pin: when the pinned text's own
estimate exceeds ``target_tokens``, the post-compaction estimate can never fall
to target and the summarizing tier fires again on **every** model request for the
rest of the run. Measured over a six-step turn: one paid summarizer call with no
pin or a small pin, **six** with an oversized one, one-to-one with requests and
unbounded. Treat an oversized pin as a host-side error. elja does not yet bound
this; doing so needs a latch that refuses to re-enter the summarizing tier once
it has failed to reach target, which is not built.
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
# Substituted in to prove the placeholder is a placeholder. Alphanumeric on
# purpose, so `{messages!r}` — which works at format time — still renders it.
_SENTINEL = "eljatranscriptsentinel"


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
            so a patch release could reword the prompt. Failing loudly at import
            beats shipping a prompt that quietly lost its warning. Loudly at
            first use rather than at import: see ``default_summary_prompt``.
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

    So this substitutes a sentinel and checks the sentinel came out, rather than
    looking for ``{messages}`` in the text. The two are not the same, and the
    difference is a hole this function used to have: ``{messages}`` is a substring
    of ``{{messages}}``, which is an escaped brace that renders as the literal
    text and substitutes nothing. A caller reaches that by obeying this very
    function's advice to double their literal braces — doubling all of them takes
    the placeholder with it. Rendering is also the more permissive test, and
    rightly: ``{messages!r}`` and ``{messages:>10}`` both work at format time and
    are now accepted.

    Args:
        summary_prompt: The caller's prompt.

    Raises:
        ValueError: If the prompt does not substitute the transcript, or a brace
            cannot be resolved.
    """
    try:
        rendered = summary_prompt.format(messages=_SENTINEL)
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError(
            f"summary_prompt contains a brace str.format cannot resolve ({exc!r}); "
            "double any literal braces as {{ }} — but leave the "
            f"{_PLACEHOLDER!r} placeholder single. Unchecked this raises at the "
            "first summarization, deep in a conversation."
        ) from exc
    if _SENTINEL in rendered:
        return
    if _PLACEHOLDER in summary_prompt:
        raise ValueError(
            f"summary_prompt contains {_PLACEHOLDER!r} only as literal text: a doubled "
            "{{messages}} is an escaped brace, not a placeholder. Leave this one single "
            "even when doubling the rest. Unchecked, the summarizer is handed an "
            "instruction with no transcript and its output still replaces the history."
        )
    raise ValueError(
        f"summary_prompt must substitute the {_PLACEHOLDER!r} placeholder; without it the "
        "summarizer is handed an instruction with no transcript, and the summarized "
        "history is replaced by a summary written from nothing"
    )


def build_compaction(
    settings: EljaSettings,
    *,
    summarizer_model: Model | None = None,
    summarizer_model_settings: ModelSettings | None = None,
    cleared_placeholder: str | None = None,
    summary_prompt: str | None = None,
    receipts: bool = False,
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
            skills has no reason to carry that note. Must contain
            ``{messages}``; see :func:`check_summary_prompt` for why that is
            refused at construction rather than at first use.
        receipts: Leave a deterministic receipt where history was summarized
            away, so the model treats its memory of earlier work as secondhand.
            Off by default, as upstream has it. Note one receipt accumulates per
            compaction across a long caller-owned history, each with its own
            dropped-message count — upstream's de-accumulation does not match
            once the receipt has been merged into a mixed-parts message.

    Returns:
        A single tiered compaction capability, or ``[]`` when disabled.
    """
    if summary_prompt is not None:
        check_summary_prompt(summary_prompt)
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
                    preserve_first_user_message=True,
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
