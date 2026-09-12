"""E4 acceptance: composing compaction for a host that owns its own tools.

The request's script, as a test: a transcript carrying both read and write tool
results is forced through compaction, and then every property a host depends on
is checked — the write outcomes stay recoverable, no mutation rerun is
solicited, pins survive, call/result pairing is valid, an unfittable protected
floor does not silently drop it, and the summarizer is bounded.
"""

import inspect
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
from pydantic_ai_harness.compaction import (
    ReportContextUsage,
    SummarizingCompaction,
    is_pinned,
    pin,
)
from pydantic_ai_harness.compaction._receipts import is_receipt_part

from elja.compaction import CLEARED_PLACEHOLDER, build_compaction, default_summary_prompt
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


# A pin this size exceeds every target used here, which forces the summarizing
# tier to run while leaving the recent tool tail in place. It is the only regime
# in which the summarizing tier can be asked about tool pairing at all.
_GIANT_PIN = "IRREDUCIBLE CONSTRAINT " * 500


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

    async def test_masking_keeps_a_write_call_auditable_after_clearing_its_outcome(
        self, tmp_path: Path
    ) -> None:
        """Under MASKING, the action survives even when the observation is cleared.

        That is what makes a host's receipt lookup possible: the tool_call_id is
        still in history, so the host can find its own record of the write. Scoped
        to masking in the name deliberately — see the test below for what the
        summarizing tier does to the same claim.
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

    async def test_summarizing_past_a_write_takes_its_call_id_with_it(
        self, tmp_path: Path
    ) -> None:
        """And the host must not read the claim above as unconditional.

        Masking rewrites a result in place, so the call stays. Summarization
        replaces everything before the cutoff, so a write whose call sits back
        there loses its tool_call_id entirely — the receipt is findable only in the
        host's own store, never by walking the history it gets back. Measured: 4
        write ids survive where masking alone runs, and 0 once the summarizing tier
        reaches past them.

        This is not a defect to fix in elja: there is nowhere for the id to go once
        the message holding it is gone. It is a fact a host has to persist against,
        which is why the example writes its receipt to its own store at tool time.

        Twelve pairs rather than eight because `target_tokens` floors at 1000 and
        the host's short placeholder reclaims enough at eight to keep the whole run
        inside the masking tier — the boundary moves with the placeholder's own
        length, which is itself worth knowing.
        """
        settings = _settings(tmp_path, target=1000)
        history = _mixed_history(12)
        views, roles = await _drive(
            settings,
            list(build_compaction(settings, cleared_placeholder=HOST_PLACEHOLDER)),
            history,
        )
        assert "summarizer" in roles, "the summarizing tier never ran; the case is the wrong one"
        write_ids = [
            part.tool_call_id
            for message in views[0]
            for part in getattr(message, "parts", [])
            if isinstance(part, ToolCallPart) and part.tool_name == "write_row"
        ]
        assert write_ids == []


class TestStructuralInvariants:
    @pytest.mark.parametrize(
        ("target", "pinned", "summarizes", "pairs_survive"),
        [
            (3000, None, False, True),
            (1000, None, True, False),
            (3000, _GIANT_PIN, True, True),
        ],
        ids=["masking-only", "summarizing-empties-the-tail", "summarizing-keeps-a-tail"],
    )
    async def test_tool_call_and_result_pairing_stays_valid(
        self,
        tmp_path: Path,
        target: int,
        pinned: str | None,
        summarizes: bool,
        pairs_survive: bool,
    ) -> None:
        """Neither tier leaves an orphaned call or return.

        Three regimes, because two of them cannot see the invariant. Under
        summarization alone the tool tail is emptied — 0 calls and 0 returns — and
        ``set() - set() == set()`` holds for every possible implementation, so that
        case certifies nothing about the summarizing tier. The third regime is the
        one that asks it the question: a pin larger than the target forces
        summarization AND leaves the recent pairs standing. Measured across the
        three: 8/8 pairs, 0/0, 7/7.

        The canary says *something* compacted; ``pairs_survive`` says whether there
        was anything left to pair. Both are needed, because round 1 of this PR's
        review found exactly this disease one layer up.
        """
        settings = _settings(tmp_path, target=target)
        history = _mixed_history(8, pinned=pinned)
        views, roles = await _drive(settings, list(build_compaction(settings)), history)
        _assert_compaction_fired(views, history)
        assert ("summarizer" in roles) is summarizes, "this regime is not the one named"
        call_ids = set()
        return_ids = set()
        for message in views[0]:
            for part in getattr(message, "parts", []):
                if isinstance(part, ToolCallPart):
                    call_ids.add(part.tool_call_id)
                elif isinstance(part, ToolReturnPart):
                    return_ids.add(part.tool_call_id)
        assert bool(call_ids) is pairs_survive, (
            "the regime did not leave what it is supposed to leave, so the "
            "pairing assertions below may be vacuous"
        )
        assert call_ids - return_ids == set()
        assert return_ids - call_ids == set()

    @pytest.mark.parametrize(
        ("target", "pin_text", "summarizes"),
        [
            (3000, "NEVER DROP: tenant=acme, currency=USD", False),
            (1000, "NEVER DROP: tenant=acme, currency=USD", True),
            (3000, f"NEVER DROP: tenant=acme, currency=USD {_GIANT_PIN}", True),
        ],
        ids=["masking-only", "summarizing", "summarizing-with-a-tail"],
    )
    async def test_a_pinned_part_survives_whichever_tiers_run(
        self, tmp_path: Path, target: int, pin_text: str, summarizes: bool
    ) -> None:
        """ "Every tier" is a claim about the pair of cases, not about each one.

        A small pin at target=3000 never reaches the summarizing tier at all, so
        ``summarizes`` is asserted per case: it is what stops the parametrization
        quietly collapsing to one regime after an estimator change upstream.
        """
        settings = _settings(tmp_path, target=target)
        history = _mixed_history(8, pinned=pin_text)
        views, roles = await _drive(settings, list(build_compaction(settings)), history)
        _assert_compaction_fired(views, history)
        assert ("summarizer" in roles) is summarizes, "this regime is not the one named"
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
        self,
        tmp_path: Path,
        pinned: str | None,
        steps: int,
        *,
        first_message: str = "original task: reconcile the ledger",
        preserve_first_user_message: bool = True,
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
            capabilities=build_compaction(
                settings, preserve_first_user_message=preserve_first_user_message
            ),
        )
        history = _mixed_history(8, pinned=pinned)
        history[0] = ModelRequest(parts=[UserPromptPart(content=first_message)])
        await agent.run(
            "carry on",
            message_history=history,
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

    async def test_a_large_first_message_does_it_too_with_no_pin_anywhere(
        self, tmp_path: Path
    ) -> None:
        """The realistic shape of the same failure, and the one the docs missed.

        A pin is something a host opts into. The first user message is something elja
        preserves on the host's behalf, so a server-side agent whose first message is
        a task brief, a ticket body or a pasted document reaches the unbounded
        summarizer with no pin anywhere — having followed the advice exactly.
        Measured: 1 summarizer call over 6 requests with a short first message, 6
        with a 20k-character one.
        """
        requests, summarizations = await self._summarizer_calls_over_a_multi_step_turn(
            tmp_path, None, steps=6, first_message="TASK BRIEF " * 2000
        )
        assert requests == 6
        assert summarizations == requests

    async def test_turning_off_first_message_preservation_is_the_escape(
        self, tmp_path: Path
    ) -> None:
        """Which is why the knob is exposed rather than hard-coded.

        Same oversized first message, same target, one argument different: back to one
        summarizer call for the whole turn. A host that takes this should carry the
        task in its own instructions, where compaction cannot reach it.
        """
        requests, summarizations = await self._summarizer_calls_over_a_multi_step_turn(
            tmp_path,
            None,
            steps=6,
            first_message="TASK BRIEF " * 2000,
            preserve_first_user_message=False,
        )
        assert requests == 6
        assert summarizations == 1


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
        with pytest.raises(ValueError, match=r"does not substitute the transcript"):
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

    @pytest.mark.parametrize(
        "unresolvable",
        [
            'Output JSON like {"intent": "x"}\n\n{messages}',
            "{messages} then {0}",
            "a { brace",
            "{messages.user}",
            "{messages[intent]}",
        ],
        ids=[
            "stray-open-brace-pair",
            "positional-field",
            "lone-open-brace",
            "attribute-access",
            "string-key-index",
        ],
    )
    def test_every_kind_of_unresolvable_field_is_refused_with_guidance(
        self, tmp_path: Path, unresolvable: str
    ) -> None:
        """str.format raises at least FIVE exception types on a bad field.

        A literal-brace pair raises `KeyError`, a positional field `IndexError`, a
        lone brace `ValueError` from the parser, attribute access `AttributeError`,
        and a string key into a string `TypeError`. Every one has to arrive as the
        same `ValueError` carrying the escape instructions, or a caller guarding its
        construction path with `except ValueError` crashes instead — and the last two
        were escaping bare, because the first version of this test listed three and
        the caught set was narrowed to match it.
        """
        with pytest.raises(ValueError, match=r"double any literal braces"):
            build_compaction(_settings(tmp_path), summary_prompt=unresolvable)

    @pytest.mark.parametrize(
        "doubled",
        [
            "Summarize.\n\n{{messages}}",
            'Output JSON like {{"intent": "x"}}\n\n{{messages}}',
            "{messages[0]}",
            "{messages:.5}",
            "Summarize: eljatranscriptsentinelalpha",
        ],
        ids=[
            "placeholder-alone",
            "everything-doubled",
            "first-character-only",
            "truncated-to-five",
            "sentinel-text-but-no-placeholder",
        ],
    )
    def test_a_prompt_that_does_not_substitute_is_refused_despite_the_text(
        self, tmp_path: Path, doubled: str
    ) -> None:
        """The hole a substring test leaves, and the route a caller takes into it.

        `{messages}` is a substring of `{{messages}}`, so looking for the
        placeholder in the text accepts an escaped brace that renders as literal
        text and substitutes nothing — the summarizer is then handed an instruction
        with no transcript and its output replaces the history anyway. The caller
        gets there by obeying this module's own advice to double their literal
        braces: doubling all of them takes the placeholder with it.

        Three more shapes that substitute something useless rather than nothing:
        `{messages[0]}` keeps one character and `{messages:.5}` keeps five, which is
        a transcript in name only. And a prompt containing the sentinel's own text
        with no placeholder at all used to be ACCEPTED, which is why the check now
        renders twice with two different transcripts and requires the results to
        differ rather than looking for a magic string.
        """
        with pytest.raises(ValueError, match=r"does not substitute the transcript"):
            build_compaction(_settings(tmp_path), summary_prompt=doubled)

    @pytest.mark.parametrize(
        "accepted",
        ["Summarize.\n\n{messages}", "{messages!r}", "{messages:>10}", "{{{messages}}}"],
        ids=["plain", "repr-conversion", "format-spec", "braced-placeholder"],
    )
    def test_a_prompt_that_substitutes_is_accepted(self, tmp_path: Path, accepted: str) -> None:
        """Rendering is the more permissive test, and correctly so.

        A conversion and a format spec both substitute at format time, so refusing
        them would be the guard over-reaching. `{{{messages}}}` is a literal brace
        either side of a real placeholder.
        """
        assert build_compaction(_settings(tmp_path), summary_prompt=accepted)

    def test_the_prompt_is_checked_even_when_compaction_is_disabled(self, tmp_path: Path) -> None:
        """`enabled` comes from settings, so the error must not depend on it.

        A host with `compaction.enabled = false` in dev and `true` in prod would
        otherwise meet its broken prompt for the first time in prod. Validating
        before honoring the flag is deliberate, and moving the check below the early
        return leaves the whole suite green without this.
        """
        settings = EljaSettings(
            workspace=WorkspaceConfig(root=tmp_path),
            compaction=CompactionConfig(enabled=False, target_tokens=3000),
        )
        with pytest.raises(ValueError, match=r"does not substitute the transcript"):
            build_compaction(settings, summary_prompt="Summarize concisely.")
        # And the disabled path still returns nothing for a prompt that is fine.
        assert build_compaction(settings, summary_prompt="Summarize.\n\n{messages}") == []

    @pytest.mark.parametrize(
        ("substitute", "refusal"),
        [("{transcript}", r"cannot resolve"), ("", r"does not substitute the transcript")],
        ids=["variable-renamed", "substitution-dropped"],
    )
    def test_eljas_own_default_prompt_is_checked_the_same_way(
        self, substitute: str, refusal: str
    ) -> None:
        """The anchor guard and the placeholder guard are different strings.

        `extend_summary_prompt` needs `<messages>` to insert the skills warning; the
        summarizer substitutes `{messages}`. The harness pin is a range, so a patch
        release could keep the anchor and rename the variable — and then every
        convenience-path caller would silently get a prompt that substitutes
        nothing, on the one path no caller can override.

        Fabricated by editing the upstream default in place, which is the only way
        to reach the state a future release would put us in. Two shapes of drift: a
        renamed variable, which `str.format` refuses outright, and a dropped
        substitution, which it silently no-ops. Both must be refused, and only one
        of them announces itself.
        """
        function = SummarizingCompaction.__init__
        defaulted = [
            parameter.name
            for parameter in inspect.signature(function).parameters.values()
            if parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
            and parameter.default is not inspect.Parameter.empty
        ]
        original = function.__defaults__
        assert original is not None and len(original) == len(defaulted)
        index = defaulted.index("summary_prompt")
        drifted = str(original[index]).replace("{messages}", substitute)
        assert "<messages>" in drifted, "the fabricated prompt must keep the anchor"
        assert "{messages}" not in drifted

        default_summary_prompt.cache_clear()
        function.__defaults__ = (*original[:index], drifted, *original[index + 1 :])
        try:
            with pytest.raises(ValueError, match=refusal):
                default_summary_prompt()
        finally:
            function.__defaults__ = original
            default_summary_prompt.cache_clear()
        # And the real default passes the same check it is now subject to.
        assert "{messages}" in default_summary_prompt()

    def test_a_valid_prompt_is_accepted(self, tmp_path: Path) -> None:
        assert build_compaction(_settings(tmp_path), summary_prompt="Summarize.\n\n{messages}")

    def test_nothing_is_checked_when_no_prompt_is_given(self, tmp_path: Path) -> None:
        assert build_compaction(_settings(tmp_path))
