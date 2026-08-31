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
from typing import Any

from pydantic_ai.capabilities import AbstractCapability
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
_SUMMARY_PROMPT: str = str(
    inspect.signature(SummarizingCompaction.__init__).parameters["summary_prompt"].default
).replace(
    "<messages>",
    "If any skills were loaded via load_capability in the conversation, state under "
    "'## Open questions' that they are no longer loaded and must be re-loaded via "
    "load_capability before use.\n\n<messages>",
)


def build_compaction(settings: EljaSettings) -> list[AbstractCapability[Any]]:
    """Build the compaction capability from settings (empty list if disabled).

    Args:
        settings: Resolved elja settings.

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
                    max_messages=1,
                    keep_messages=cfg.keep_messages,
                    # Token-bound the verbatim tail so the target is always
                    # reachable — otherwise an irreducible tail above target
                    # re-fires a summarizer LLM call on EVERY request.
                    keep_tokens=cfg.target_tokens // 3,
                    preserve_first_user_message=True,
                    incremental=True,
                    summary_prompt=_SUMMARY_PROMPT,
                ),
            ],
            target_tokens=cfg.target_tokens,
        )
    ]
