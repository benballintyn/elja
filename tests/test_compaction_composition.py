"""E4 acceptance: composing compaction for a host that owns its own tools.

The request's script, as a test: a transcript carrying both read and write tool
results is forced through compaction, and then every property a host depends on
is checked — the write outcomes stay recoverable, no mutation rerun is
solicited, pins survive, call/result pairing is valid, an unfittable protected
floor does not silently drop it, and the summarizer is bounded.
"""

from pathlib import Path

import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai_harness.compaction import ReportContextUsage, is_pinned, pin
from pydantic_ai_harness.compaction._receipts import is_receipt_part

from elja.compaction import CLEARED_PLACEHOLDER, build_compaction
from elja.deps import EljaDeps
from elja.settings import CompactionConfig, EljaSettings, WorkspaceConfig

# A host's own wording: the evidence is in its store, not re-derivable by
# repeating a write.
HOST_PLACEHOLDER = "[result cleared; retrieve the saved evidence by its receipt id]"


def _mixed_history(
    pairs: int, *, result_size: int = 600, pinned: str | None = None
) -> list[ModelMessage]:
    """A transcript of alternating read and write tool results, too big to fit."""
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="original task: reconcile the ledger")])
    ]
    if pinned is not None:
        messages.append(ModelRequest(parts=[pin(pinned)]))
    for i in range(pairs):
        tool = "read_row" if i % 2 == 0 else "write_row"
        messages.append(
            ModelResponse(
                parts=[ToolCallPart(tool_name=tool, args={"row": i}, tool_call_id=f"c{i}")]
            )
        )
        messages.append(
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name=tool,
                        content=f"row{i} " * result_size,
                        tool_call_id=f"c{i}",
                    )
                ]
            )
        )
    return messages


def _settings(tmp_path: Path, target: int = 3000, keep_messages: int = 2) -> EljaSettings:
    return EljaSettings(
        workspace=WorkspaceConfig(root=tmp_path),
        compaction=CompactionConfig(
            target_tokens=target, keep_tool_pairs=1, keep_messages=keep_messages
        ),
    )


async def _drive(
    settings: EljaSettings,
    capabilities: list[AbstractCapability[object]],
    history: list[ModelMessage],
) -> tuple[list[list[ModelMessage]], list[str]]:
    """Run one turn, returning what the agent saw and which roles were asked."""
    views: list[list[ModelMessage]] = []
    roles: list[str] = []

    def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if "summarization assistant" in (info.instructions or ""):
            roles.append("summarizer")
            return ModelResponse(parts=[TextPart(content="## Intent\nreconcile the ledger")])
        roles.append("agent")
        views.append(list(messages))
        return ModelResponse(parts=[TextPart(content="ok")])

    agent: Agent[EljaDeps, str] = Agent(
        FunctionModel(script), deps_type=EljaDeps, capabilities=capabilities
    )
    result = await agent.run(
        "carry on", message_history=history, deps=EljaDeps.from_settings(settings)
    )
    assert result.output == "ok"
    return views, roles


def _rendered(views: list[list[ModelMessage]]) -> str:
    assert views, "the agent never ran"
    return str(views[0])


def _receipts(views: list[list[ModelMessage]]) -> list[object]:
    """Receipt parts in the model's view, found structurally.

    ``is_receipt_part`` is not in the harness's ``__all__``, but matching on the
    receipt's wording would break when upstream rewords content it explicitly
    calls provisional.
    """
    return [
        part
        for message in views[0]
        for part in getattr(message, "parts", [])
        if is_receipt_part(part)
    ]


def _assert_compaction_fired(views: list[list[ModelMessage]], history: list[ModelMessage]) -> None:
    """A canary, because an untouched history is trivially well-formed.

    "Pins survive", "pairing stays valid" and "the protected floor is kept" are
    all true of a transcript nothing touched, so each of those tests has to show
    elja's machinery ran before asserting what it left behind.

    Measured on rendered CONTENT, not message count: masking replaces tool-result
    text in place without removing a message, and pydantic-ai's own
    message-merging shrinks the count by one when a pin is present even with
    compaction switched off. Content separates cleanly — 27.5k characters through
    to the model with compaction off, 0.6k with it on, for this file's histories.
    """
    assert views, "the agent never ran"
    assert len(str(views[0])) < len(str(history)), (
        "compaction never fired; the assertion is vacuous"
    )


class TestThePlaceholderIsTheHosts:
    async def test_a_custom_placeholder_replaces_the_rerun_guidance(self, tmp_path: Path) -> None:
        """E4: never require the application to accept 're-run the tool'."""
        settings = _settings(tmp_path)
        views, _ = await _drive(
            settings,
            list(build_compaction(settings, cleared_placeholder=HOST_PLACEHOLDER)),
            _mixed_history(8),
        )
        rendered = _rendered(views)
        assert HOST_PLACEHOLDER in rendered
        assert "re-run the tool" not in rendered
        assert ".elja/spill/" not in rendered

    async def test_the_default_is_unchanged_when_nothing_is_passed(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        views, _ = await _drive(settings, list(build_compaction(settings)), _mixed_history(8))
        rendered = _rendered(views)
        assert CLEARED_PLACEHOLDER in rendered
        assert HOST_PLACEHOLDER not in rendered

    async def test_write_outcomes_are_masked_but_their_calls_remain_auditable(
        self, tmp_path: Path
    ) -> None:
        """The action survives even when the observation is cleared.

        That is what makes a host's receipt lookup possible: the tool_call_id is
        still in history, so the host can find its own record of the write.
        """
        settings = _settings(tmp_path)
        views, _ = await _drive(
            settings,
            list(build_compaction(settings, cleared_placeholder=HOST_PLACEHOLDER)),
            _mixed_history(8),
        )
        calls = [
            part
            for message in views[0]
            for part in getattr(message, "parts", [])
            if isinstance(part, ToolCallPart)
        ]
        write_ids = [c.tool_call_id for c in calls if c.tool_name == "write_row"]
        assert write_ids, "every write call was dropped, so no receipt is findable"
        assert HOST_PLACEHOLDER in _rendered(views)


class TestStructuralInvariants:
    @pytest.mark.parametrize("target", [1000, 3000])
    async def test_tool_call_and_result_pairing_stays_valid(
        self, tmp_path: Path, target: int
    ) -> None:
        """Both tiers, and neither leaves an orphaned call or return."""
        settings = _settings(tmp_path, target=target)
        history = _mixed_history(8)
        views, _ = await _drive(settings, list(build_compaction(settings)), history)
        _assert_compaction_fired(views, history)
        call_ids = set()
        return_ids = set()
        for message in views[0]:
            for part in getattr(message, "parts", []):
                if isinstance(part, ToolCallPart):
                    call_ids.add(part.tool_call_id)
                elif isinstance(part, ToolReturnPart):
                    return_ids.add(part.tool_call_id)
        assert call_ids - return_ids == set()
        assert return_ids - call_ids == set()

    @pytest.mark.parametrize("target", [1000, 3000])
    async def test_a_pinned_part_survives_every_tier(self, tmp_path: Path, target: int) -> None:
        settings = _settings(tmp_path, target=target)
        history = _mixed_history(8, pinned="NEVER DROP: tenant=acme, currency=USD")
        views, _ = await _drive(settings, list(build_compaction(settings)), history)
        _assert_compaction_fired(views, history)
        pinned = [
            part
            for message in views[0]
            for part in getattr(message, "parts", [])
            if is_pinned(part)
        ]
        assert len(pinned) == 1
        assert "tenant=acme" in _rendered(views)

    async def test_protected_material_larger_than_the_target_is_kept_not_dropped(
        self, tmp_path: Path
    ) -> None:
        """E4: do not silently drop protected material.

        There is no upstream "could not fit" signal, so the guarantee elja can
        make is the one that matters: the pin survives even when it alone
        exceeds the target.
        """
        settings = _settings(tmp_path, target=1000)
        giant = "IRREDUCIBLE CONSTRAINT " * 500
        history = _mixed_history(8, pinned=giant)
        views, _ = await _drive(settings, list(build_compaction(settings)), history)
        _assert_compaction_fired(views, history)
        assert "IRREDUCIBLE CONSTRAINT" in _rendered(views)
        pinned = [
            part
            for message in views[0]
            for part in getattr(message, "parts", [])
            if is_pinned(part)
        ]
        assert len(pinned) == 1


class TestAnOversizedPinIsNotBounded:
    """The cost of the pin guarantee, measured rather than claimed safe.

    Re-injection happens AFTER the tail is trimmed, so ``keep_tokens`` cannot
    bound a pin. When the pinned text's own estimate exceeds ``target_tokens``
    the post-compaction estimate never falls to target and the summarizing tier
    fires again on every model request for the rest of the run.
    """

    async def _summarizer_calls_over_a_multi_step_turn(
        self, tmp_path: Path, pinned: str | None, steps: int
    ) -> tuple[int, int]:
        settings = _settings(tmp_path, target=1000)
        toolset: FunctionToolset[EljaDeps] = FunctionToolset()

        @toolset.tool
        async def ping(ctx: RunContext[EljaDeps]) -> str:
            """A cheap tool, so one turn takes several model requests."""
            return "pong"

        roles: list[str] = []
        requests: list[int] = []

        def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if "summarization assistant" in (info.instructions or ""):
                roles.append("summarizer")
                return ModelResponse(parts=[TextPart(content="## Intent\nledger")])
            requests.append(1)
            if len(requests) < steps:
                return ModelResponse(parts=[ToolCallPart(tool_name="ping", args={})])
            return ModelResponse(parts=[TextPart(content="ok")])

        agent: Agent[EljaDeps, str] = Agent(
            FunctionModel(script),
            deps_type=EljaDeps,
            toolsets=[toolset],
            capabilities=build_compaction(settings),
        )
        await agent.run(
            "carry on",
            message_history=_mixed_history(8, pinned=pinned),
            deps=EljaDeps.from_settings(settings),
        )
        return len(requests), roles.count("summarizer")

    @pytest.mark.parametrize("pinned", [None, "NEVER DROP: tenant=acme"])
    async def test_an_ordinary_history_summarizes_once_per_turn(
        self, tmp_path: Path, pinned: str | None
    ) -> None:
        """The control: keep_tokens does its job when the pin fits."""
        requests, summarizations = await self._summarizer_calls_over_a_multi_step_turn(
            tmp_path, pinned, steps=6
        )
        assert requests == 6
        assert summarizations == 1

    async def test_a_pin_larger_than_the_target_summarizes_once_per_request(
        self, tmp_path: Path
    ) -> None:
        """Unbounded, and elja does not yet bound it.

        Written in the direction that documents the real behaviour rather than
        the behaviour E4 asks for, so the gap is visible instead of implied. A
        fix needs a latch that refuses to re-enter the summarizing tier once it
        has failed to reach target; when that lands, this test changes with it.
        """
        requests, summarizations = await self._summarizer_calls_over_a_multi_step_turn(
            tmp_path, "IRREDUCIBLE CONSTRAINT " * 500, steps=6
        )
        assert requests == 6
        assert summarizations == requests


class TestSummarizerComposition:
    async def test_a_custom_summary_prompt_is_what_the_summarizer_receives(
        self, tmp_path: Path
    ) -> None:
        settings = _settings(tmp_path, target=1000)
        seen_prompts: list[str] = []

        def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if "SUMMARIZE FOR A HOUSEHOLD" in str(messages):
                seen_prompts.append("custom")
                return ModelResponse(parts=[TextPart(content="## Intent\nledger")])
            return ModelResponse(parts=[TextPart(content="ok")])

        agent: Agent[EljaDeps, str] = Agent(
            FunctionModel(script),
            deps_type=EljaDeps,
            capabilities=build_compaction(
                settings, summary_prompt="SUMMARIZE FOR A HOUSEHOLD\n\n{messages}"
            ),
        )
        await agent.run(
            "carry on",
            message_history=_mixed_history(8),
            deps=EljaDeps.from_settings(settings),
        )
        assert seen_prompts == ["custom"]

    async def test_eljas_skills_note_is_absent_from_a_custom_prompt(self, tmp_path: Path) -> None:
        """A host with no skills should not be told to reload them."""
        settings = _settings(tmp_path, target=1000)
        prompts: list[str] = []

        def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if "SUMMARIZE FOR A HOUSEHOLD" in str(messages):
                prompts.append(str(messages))
                return ModelResponse(parts=[TextPart(content="## Intent\nledger")])
            return ModelResponse(parts=[TextPart(content="ok")])

        agent: Agent[EljaDeps, str] = Agent(
            FunctionModel(script),
            deps_type=EljaDeps,
            capabilities=build_compaction(
                settings, summary_prompt="SUMMARIZE FOR A HOUSEHOLD\n\n{messages}"
            ),
        )
        await agent.run(
            "carry on",
            message_history=_mixed_history(8),
            deps=EljaDeps.from_settings(settings),
        )
        assert prompts, "the summarizer never ran"
        assert "load_capability" not in prompts[0]

    async def test_the_default_summary_prompt_is_eljas_own(self, tmp_path: Path) -> None:
        """Passing nothing must keep elja's prompt, skills note included.

        The other tests here detect the summarizer by its *instructions*, which
        come from a different argument, so without this the default prompt's
        content is unpinned.
        """
        settings = _settings(tmp_path, target=1000)
        prompts: list[str] = []

        def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if "summarization assistant" in (info.instructions or ""):
                prompts.append(str(messages))
                return ModelResponse(parts=[TextPart(content="## Intent\nledger")])
            return ModelResponse(parts=[TextPart(content="ok")])

        agent: Agent[EljaDeps, str] = Agent(
            FunctionModel(script), deps_type=EljaDeps, capabilities=build_compaction(settings)
        )
        await agent.run(
            "carry on",
            message_history=_mixed_history(8),
            deps=EljaDeps.from_settings(settings),
        )
        assert prompts, "the summarizer never ran"
        # The harness's section headings, plus elja's own skills warning.
        assert "## Key decisions" in prompts[0]
        assert "load_capability" in prompts[0]

    async def test_receipts_leave_a_note_where_history_was_summarized_away(
        self, tmp_path: Path
    ) -> None:
        settings = _settings(tmp_path, target=1000)
        views, roles = await _drive(
            settings, list(build_compaction(settings, receipts=True)), _mixed_history(8)
        )
        assert "summarizer" in roles
        # Structural and a COUNT, not a match on upstream's wording: receipts
        # accumulate one per compaction across a long caller-owned history.
        assert len(_receipts(views)) == 1

    async def test_receipts_are_off_by_default(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path, target=1000)
        views, roles = await _drive(settings, list(build_compaction(settings)), _mixed_history(8))
        assert "summarizer" in roles
        assert _receipts(views) == []

    async def test_the_summarizer_is_called_at_most_once_per_turn(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path, target=1000)
        _, roles = await _drive(settings, list(build_compaction(settings)), _mixed_history(8))
        assert roles.count("summarizer") == 1


class TestRetentionKnobsActuallyRetain:
    """Two config knobs whose effect nothing observed."""

    def test_keep_messages_cannot_bind_because_of_two_settings_elja_chooses(
        self, tmp_path: Path
    ) -> None:
        """``keep_messages`` is not an upper bound; upstream never reads it here.

        ``SummarizingCompaction`` consults it in exactly two places: as the cutoff
        when ``keep_tokens is None``, and under ``keep_user_messages``. elja always
        sets ``keep_tokens`` (``target_tokens // 3``, and target is ``ge=1000``)
        and never sets ``keep_user_messages``, so neither branch is reachable.

        Asserting the two *reasons* rather than an equality of outcomes, because
        the equality held for every input ever tried — a parameter that is not
        read cannot produce a difference, so an equality assertion pins nothing.
        """
        (tier,) = [
            tier
            for tier in build_compaction(_settings(tmp_path))[0].tiers  # type: ignore[attr-defined]
            if type(tier).__name__ == "SummarizingCompaction"
        ]
        assert tier.keep_tokens is not None
        assert tier.keep_user_messages is False

    async def test_an_incremental_summary_is_given_the_previous_one(self, tmp_path: Path) -> None:
        """``incremental=True`` only engages on the SECOND summarization.

        A single-summarization test cannot pin it, which is why it went
        unobserved: the flag's whole effect is that the second summarizer call
        receives the first summary instead of rewriting from scratch.
        """
        settings = _settings(tmp_path, target=1000, keep_messages=2)
        summarizer_prompts: list[str] = []

        def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if "summarization assistant" in (info.instructions or ""):
                summarizer_prompts.append(str(messages))
                return ModelResponse(
                    parts=[TextPart(content=f"## Intent\nsummary {len(summarizer_prompts)}")]
                )
            return ModelResponse(parts=[TextPart(content="ok")])

        agent: Agent[EljaDeps, str] = Agent(
            FunctionModel(script), deps_type=EljaDeps, capabilities=build_compaction(settings)
        )
        deps = EljaDeps.from_settings(settings)
        history = _mixed_history(20)
        for turn in range(3):
            result = await agent.run(f"turn{turn}", message_history=history, deps=deps)
            # Re-inflate with fresh bulk so a second boundary is crossed.
            history = [*result.all_messages(), *_mixed_history(20)[1:]]
        assert len(summarizer_prompts) >= 2, "only one summarization; nothing to be incremental"
        # The SECOND call, which is exactly what the flag does: it receives the
        # first summary instead of rewriting from scratch. Asserting on the LAST
        # call would pin an upstream defect as the contract — with caller-owned
        # history the summaries accumulate and _extract_previous_summary returns
        # the oldest, so the third call is still anchored on summary 1.
        assert "<previous-summary>" in summarizer_prompts[1]
        assert "summary 1" in summarizer_prompts[1]


class TestReportingOrder:
    async def test_reporting_after_compaction_measures_the_request_that_was_sent(
        self, tmp_path: Path
    ) -> None:
        """List order decides, because none of these declare an ordering.

        Placed last it reports the real request; placed first it reports one that
        never existed. Both directions are asserted, because a rule stated in one
        direction only is not pinned.
        """
        settings = _settings(tmp_path)
        readings: dict[str, list[int]] = {"after": [], "before": []}

        async def measure(where: str) -> None:
            taken: list[int] = []
            report: ReportContextUsage[EljaDeps] = ReportContextUsage(
                on_usage=lambda usage: taken.append(usage.used_tokens)
            )
            compaction = list(build_compaction(settings))
            capabilities = [*compaction, report] if where == "after" else [report, *compaction]
            await _drive(settings, capabilities, _mixed_history(8))
            readings[where] = taken

        await measure("after")
        await measure("before")
        assert readings["after"] and readings["before"]
        assert readings["after"][0] < readings["before"][0]
        # And the one placed last is the one near the target it was given.
        assert readings["after"][0] <= settings.compaction.target_tokens
        assert readings["before"][0] > settings.compaction.target_tokens


class TestTheSummaryPromptIsCheckedAtConstruction:
    """A prompt the summarizer cannot use must fail before the run, not during."""

    def test_a_prompt_without_the_placeholder_is_refused(self, tmp_path: Path) -> None:
        """Otherwise the summary is written from nothing and replaces history."""
        with pytest.raises(ValueError, match=r"must contain the '\{messages\}' placeholder"):
            build_compaction(_settings(tmp_path), summary_prompt="Summarize concisely.")

    def test_an_unresolvable_brace_is_refused(self, tmp_path: Path) -> None:
        """str.format raises on a stray brace, at the first summarization."""
        with pytest.raises(ValueError, match=r"double any literal braces"):
            build_compaction(
                _settings(tmp_path),
                summary_prompt='Output JSON like {"intent": "x"}\n\n{messages}',
            )

    def test_a_doubled_brace_is_accepted(self, tmp_path: Path) -> None:
        """The escape the error message tells the caller to use actually works."""
        capabilities = build_compaction(
            _settings(tmp_path),
            summary_prompt='Output JSON like {{"intent": "x"}}\n\n{messages}',
        )
        assert capabilities

    def test_a_valid_prompt_is_accepted(self, tmp_path: Path) -> None:
        assert build_compaction(_settings(tmp_path), summary_prompt="Summarize.\n\n{messages}")

    def test_nothing_is_checked_when_no_prompt_is_given(self, tmp_path: Path) -> None:
        assert build_compaction(_settings(tmp_path))
