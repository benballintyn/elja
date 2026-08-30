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

The system prompt/instructions live outside message history in pydantic-ai,
so they are never subject to compaction at all.
"""

from typing import Any

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai_harness.compaction import (
    ClearToolResults,
    SummarizingCompaction,
    TieredCompaction,
)

from elja.settings import EljaSettings

CLEARED_PLACEHOLDER = (
    "[old tool result cleared to save context; re-run the tool if you need it again]"
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
                # the real trigger and per-tier limits are overridden.
                ClearToolResults(
                    max_tokens=1,
                    keep_pairs=cfg.keep_tool_pairs,
                    placeholder=CLEARED_PLACEHOLDER,
                ),
                SummarizingCompaction(
                    max_messages=1,
                    keep_messages=cfg.keep_messages,
                    preserve_first_user_message=True,
                    incremental=True,
                ),
            ],
            target_tokens=cfg.target_tokens,
        )
    ]
