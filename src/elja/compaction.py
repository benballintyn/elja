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
            FIRST USE, not at import: see ``default_summary_prompt``.
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
    """
    return extend_summary_prompt(
        str(inspect.signature(SummarizingCompaction.__init__).parameters["summary_prompt"].default)
    )


def build_compaction(
    settings: EljaSettings,
    *,
    summarizer_model: Model | None = None,
    summarizer_model_settings: ModelSettings | None = None,
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
            ``{"extra_headers": {"x-phase": "compaction"}}``.

    Returns:
        A single tiered compaction capability, or ``[]`` when disabled.
    """
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
                    placeholder=CLEARED_PLACEHOLDER,
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
                    summary_prompt=default_summary_prompt(),
                ),
            ],
            target_tokens=cfg.target_tokens,
        )
    ]
