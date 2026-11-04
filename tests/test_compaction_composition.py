"""E4 acceptance: composing compaction for a host that owns its own tools.

The request's script, as a test: a transcript carrying both read and write tool
results is forced through compaction, and then every property a host depends on
is checked — the write outcomes stay recoverable, no mutation rerun is
solicited, pins survive, call/result pairing is valid, an unfittable protected
floor does not silently drop it, and the summarizer is bounded.

The bound's own mechanics — the budget, the prediction, the latch, the reset, the
metering wrapper — are pinned in ``tests/test_compaction_bounds.py``.
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
    SystemPromptPart,
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

from elja.compaction import (
    CLEARED_PLACEHOLDER,
    OnUnfit,
    TargetUnreachable,
    TargetUnreachableError,
    build_compaction,
    default_summary_prompt,
)
from elja.deps import EljaDeps
from elja.settings import CompactionConfig, EljaSettings, WorkspaceConfig

# A host's own wording: the evidence is in its store, not re-derivable by
# repeating a write.
HOST_PLACEHOLDER = "[result cleared; retrieve the saved evidence by its receipt id]"


def _recording(reports: list[TargetUnreachable]) -> OnUnfit[object]:
    """A gate that records the report and lets the request go — the host's choice."""

    def gate(ctx: RunContext[object], report: TargetUnreachable) -> None:
        reports.append(report)

    return gate


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
        ("target", "pinned", "summarizes", "pairs_survive", "fits"),
        [
            (3000, None, False, True, True),
            (1000, None, True, False, True),
            (3000, _GIANT_PIN, True, True, False),
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
        fits: bool,
    ) -> None:
        """Neither tier leaves an orphaned call or return.

        Three regimes, because two of them cannot see the invariant. Under
        summarization alone the tool tail is emptied — 0 calls and 0 returns — and
        ``set() - set() == set()`` holds for every possible implementation, so that
        case certifies nothing about the summarizing tier. The third regime is the
        one that asks it the question: a pin just under the target forces
        summarization AND leaves the recent pairs standing. Measured across the
        three: 8/8 pairs, 0/0, 7/7.

        The third is also the cannot-fit path — pin plus summary plus tail is over
        target — so it runs with a gate that lets the request go, and asserts the
        typed outcome arrived. Pairing has to hold there too: it is the request
        that gets sent.

        The canary says *something* compacted; ``pairs_survive`` says whether there
        was anything left to pair. Both are needed, because round 1 of this PR's
        review found exactly this disease one layer up.
        """
        settings = _settings(tmp_path, target=target)
        history = _mixed_history(8, pinned=pinned)
        reports: list[TargetUnreachable] = []
        views, roles = await _drive(
            settings, list(build_compaction(settings, on_unfit=_recording(reports))), history
        )
        _assert_compaction_fired(views, history)
        assert ("summarizer" in roles) is summarizes, "this regime is not the one named"
        assert (reports == []) is fits
        assert all(report.reason == "no_progress" for report in reports)
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
        ("target", "pin_text", "summarizes", "fits"),
        [
            (3000, "NEVER DROP: tenant=acme, currency=USD", False, True),
            (1000, "NEVER DROP: tenant=acme, currency=USD", True, True),
            (3000, f"NEVER DROP: tenant=acme, currency=USD {_GIANT_PIN}", True, False),
        ],
        ids=["masking-only", "summarizing", "summarizing-with-a-tail"],
    )
    async def test_a_pinned_part_survives_whichever_tiers_run(
        self, tmp_path: Path, target: int, pin_text: str, summarizes: bool, fits: bool
    ) -> None:
        """ "Every tier" is a claim about the pair of cases, not about each one.

        A small pin at target=3000 never reaches the summarizing tier at all, so
        ``summarizes`` is asserted per case: it is what stops the parametrization
        quietly collapsing to one regime after an estimator change upstream. The
        last case cannot fit, and the pin survives on that path as well.
        """
        settings = _settings(tmp_path, target=target)
        history = _mixed_history(8, pinned=pin_text)
        reports: list[TargetUnreachable] = []
        views, roles = await _drive(
            settings, list(build_compaction(settings, on_unfit=_recording(reports))), history
        )
        _assert_compaction_fired(views, history)
        assert ("summarizer" in roles) is summarizes, "this regime is not the one named"
        assert (reports == []) is fits
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
        """E4: do not meet the target by silently dropping protected material.

        The target is unreachable here, and the host is told so — typed, before the
        request is sent. A host that chooses to send anyway sends the pin intact.
        """
        settings = _settings(tmp_path, target=1000)
        giant = "IRREDUCIBLE CONSTRAINT " * 500
        history = _mixed_history(8, pinned=giant)
        reports: list[TargetUnreachable] = []
        views, _ = await _drive(
            settings, list(build_compaction(settings, on_unfit=_recording(reports))), history
        )
        _assert_compaction_fired(views, history)
        assert [report.reason for report in reports] == ["protected_floor"]
        assert "IRREDUCIBLE CONSTRAINT" in _rendered(views)
        pinned = [
            part
            for message in views[0]
            for part in getattr(message, "parts", [])
            if is_pinned(part)
        ]
        assert len(pinned) == 1

    async def test_by_default_nothing_is_sent_over_target(self, tmp_path: Path) -> None:
        """No gate means the request never leaves: a typed error, not a silent send."""
        settings = _settings(tmp_path, target=1000)
        history = _mixed_history(8, pinned="IRREDUCIBLE CONSTRAINT " * 500)
        with pytest.raises(TargetUnreachableError) as raised:
            await _drive(settings, list(build_compaction(settings)), history)
        assert raised.value.report.reason == "protected_floor"


class TestAnOversizedFloorIsBounded:
    """The issue's three shapes of irreducible content, each over a six-step turn.

    Before the bound, each of these paid for a summary before EVERY request — six
    summarizer calls over six requests, one-to-one and unbounded — because
    re-injection and first-message preservation happen after the tail is trimmed, so
    ``keep_tokens`` could not bound them. Now the protected floor is measured before
    paying: with no gate the run stops at the first request with a typed outcome and
    zero summarizer requests; with a gate that proceeds, the turn completes having
    paid for at most the configured number.
    """

    STEPS = 6

    async def _turn(
        self,
        tmp_path: Path,
        *,
        pinned: str | None = None,
        first_message: str = "original task: reconcile the ledger",
        system_prompt: str | None = None,
        preserve_first_user_message: bool = True,
        on_unfit: OnUnfit[EljaDeps] | None = None,
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
            if len(requests) < self.STEPS:
                return ModelResponse(parts=[ToolCallPart(tool_name="ping", args={})])
            return ModelResponse(parts=[TextPart(content="ok")])

        agent: Agent[EljaDeps, str] = Agent(
            FunctionModel(script),
            deps_type=EljaDeps,
            toolsets=[toolset],
            capabilities=build_compaction(
                settings,
                preserve_first_user_message=preserve_first_user_message,
                on_unfit=on_unfit,
            ),
        )
        history = _mixed_history(8, pinned=pinned)
        # Policy where `Agent(system_prompt=...)` leaves it in caller-owned history:
        # inside the first request, ahead of the user's prompt.
        policy = [SystemPromptPart(content=system_prompt)] if system_prompt else []
        history[0] = ModelRequest(parts=[*policy, UserPromptPart(content=first_message)])
        try:
            await agent.run(
                "carry on",
                message_history=history,
                deps=EljaDeps.from_settings(settings),
            )
        finally:
            self.roles = roles
        return len(requests), roles.count("summarizer")

    # The issue's three shapes: an oversized pin, a large preserved first message
    # with no pin anywhere, and protected system policy (`Agent(system_prompt=...)`).
    SHAPES = pytest.mark.parametrize(
        "shape",
        [
            {"pinned": "IRREDUCIBLE CONSTRAINT " * 500},
            {"first_message": "TASK BRIEF " * 2000},
            {"system_prompt": "POLICY: never disclose tenant ids. " * 200},
        ],
        ids=["oversized-pin", "large-first-message", "protected-system-policy"],
    )

    @pytest.mark.parametrize("pinned", [None, "NEVER DROP: tenant=acme"])
    async def test_an_ordinary_history_still_summarizes_once_per_turn(
        self, tmp_path: Path, pinned: str | None
    ) -> None:
        """The positive control: compaction runs, fits, and reports nothing."""
        reports: list[TargetUnreachable] = []
        requests, summarizations = await self._turn(
            tmp_path, pinned=pinned, on_unfit=_recording(reports)
        )
        assert requests == self.STEPS
        assert summarizations == 1
        assert reports == []

    @SHAPES
    async def test_by_default_the_run_stops_before_paying_for_a_futile_summary(
        self, tmp_path: Path, shape: dict[str, str]
    ) -> None:
        """No gate: a typed outcome at the first request, and nothing spent."""
        with pytest.raises(TargetUnreachableError) as raised:
            await self._turn(tmp_path, **shape)  # type: ignore[arg-type]
        assert self.roles == [], "a summary was paid for, or the request was sent"
        report = raised.value.report
        assert report.reason == "protected_floor"
        assert report.protected_tokens > report.target_tokens == 1000
        assert report.estimated_tokens >= report.protected_tokens
        assert (report.summarizer_requests, report.max_summarizer_requests) == (0, 3)
        assert report.occurrence == 1

    @SHAPES
    async def test_a_host_that_proceeds_pays_for_at_most_the_budget(
        self, tmp_path: Path, shape: dict[str, str]
    ) -> None:
        """Proceeding: every over-target request is reported, the summarizer is bounded.

        Measured: at most one summary over the six requests — it shrinks the
        reducible history once, and the run never re-summarizes the summary — where
        the unbounded version paid for six.
        """
        reports: list[TargetUnreachable] = []
        requests, summarizations = await self._turn(
            tmp_path,
            on_unfit=_recording(reports),
            **shape,  # type: ignore[arg-type]
        )
        assert requests == self.STEPS
        assert summarizations <= 1 < self.STEPS
        assert summarizations <= reports[0].max_summarizer_requests
        # Every request went out over target, so every one was put to the gate —
        # nothing over target is sent without it being asked.
        assert [report.occurrence for report in reports] == list(range(1, self.STEPS + 1))
        assert {report.reason for report in reports} == {"protected_floor"}

    async def test_turning_off_first_message_preservation_still_fits(self, tmp_path: Path) -> None:
        """The knob still works: the large message is summarizable, so the turn fits."""
        reports: list[TargetUnreachable] = []
        requests, summarizations = await self._turn(
            tmp_path,
            first_message="TASK BRIEF " * 2000,
            preserve_first_user_message=False,
            on_unfit=_recording(reports),
        )
        assert requests == self.STEPS
        assert summarizations == 1
        assert reports == []


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
        assert len(summarizer_prompts) >= 3, "too few summarizations to see the anchor move"
        # The SECOND call is what the flag does: it receives the first summary
        # instead of rewriting from scratch.
        assert "<previous-summary>" in summarizer_prompts[1]
        assert "summary 1" in summarizer_prompts[1]
        # And every later call is anchored on the summary just before it. Upstream
        # alone gets this wrong with caller-owned history: summaries accumulate in
        # the preserved first request and `_extract_previous_summary` returns the
        # oldest, so the third call used to be anchored on summary 1. elja drops a
        # superseded summary once its replacement is written, so there is only ever
        # one to find.
        assert len(summarizer_prompts) < 10, "single-digit labels keep the matches exact"
        for index in range(2, len(summarizer_prompts)):
            anchor = summarizer_prompts[index].split("<previous-summary>", 1)[1]
            assert f"summary {index}" in anchor
            assert f"summary {index - 1}" not in anchor


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
        with pytest.raises(ValueError, match=r"does not substitute the whole transcript"):
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
            "{messages:.23}",
            "{messages:.2000}",
            "{messages:.40000}",
            "{messages.upper}",
            "Summarize: elja-transcript-stand-in-alpha",
        ],
        ids=[
            "placeholder-alone",
            "everything-doubled",
            "first-character-only",
            "truncated-to-five",
            "truncated-to-the-first-boundary",
            "truncated-to-a-plausible-cap",
            "truncated-past-the-stand-in",
            "attribute-not-substitution",
            "one-stand-ins-text-but-no-placeholder",
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

        The rest substitute something useless rather than nothing, and the widths are
        the point. A check that compared two renderings for *difference* accepted
        `{messages:.23}` — 23 being exactly where the two stand-ins stopped agreeing —
        so a host could cap the transcript at any width above that and the summarizer
        would rewrite the history from a fragment. `{messages:.2000}` is the plausible
        shape of that mistake: someone capping cost. Requiring each rendering to
        contain its whole stand-in refuses every width.

        `{messages.upper}` is the same class from the other side: it renders a method
        repr, which differs between the two stand-ins because the objects differ, so
        difference accepted it and the spec read does not.

        `{messages:.40000}` is the width that exposed the second relocation — containment
        against a finite stand-in can only detect a truncation narrower than the stand-in
        itself, so the boundary moved rather than closing. Width is now read from the spec.

        The last one needs BOTH renderings checked, not either: a prompt quoting one
        stand-in's text with no placeholder renders identically twice, so it contains
        alpha and not beta. `any` would accept it.
        """
        with pytest.raises(ValueError, match=r"does not substitute the whole transcript"):
            build_compaction(_settings(tmp_path), summary_prompt=doubled)

    @pytest.mark.parametrize(
        "accepted",
        [
            "Summarize.\n\n{messages}",
            "{messages!r}",
            "{messages!s}",
            "{messages!a}",
            "{messages:>10}",
            "{messages:^50}",
            "{{{messages}}}",
            "{messages} and again {messages}",
        ],
        ids=[
            "plain",
            "repr-conversion",
            "str-conversion",
            "ascii-conversion",
            "pad-right",
            "centre",
            "braced-placeholder",
            "twice",
        ],
    )
    def test_a_prompt_that_substitutes_is_accepted(self, tmp_path: Path, accepted: str) -> None:
        """Rendering is the more permissive test, and correctly so.

        Every one of these substitutes the transcript in full — a conversion, padding,
        centring, literal braces either side, or twice over — so refusing any of them
        would be the guard over-reaching. Padding and centring matter: they change the
        rendering's length without removing anything, which is the case a naive
        length check would get wrong.
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
        with pytest.raises(ValueError, match=r"does not substitute the whole transcript"):
            build_compaction(settings, summary_prompt="Summarize concisely.")
        # And the disabled path still returns nothing for a prompt that is fine.
        assert build_compaction(settings, summary_prompt="Summarize.\n\n{messages}") == []

    @pytest.mark.parametrize(
        ("substitute", "refusal"),
        [("{transcript}", r"cannot resolve"), ("", r"does not substitute the whole transcript")],
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


class TestTheAdviceTextNamesTheRightPlaceholder:
    """The error messages ARE the product of `check_summary_prompt`, so pin their content.

    Changing `_PLACEHOLDER` to anything else left all tests green while both refusals
    then told the caller to keep the wrong brace single — advice that would send them
    round the exact loop the guard exists to end.
    """

    def test_both_refusals_name_the_placeholder_the_summarizer_substitutes(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match=r"'\{messages\}' placeholder is missing"):
            build_compaction(_settings(tmp_path), summary_prompt="Summarize concisely.")
        with pytest.raises(ValueError, match=r"leave the '\{messages\}' placeholder single"):
            build_compaction(_settings(tmp_path), summary_prompt="{messages} and {0}")


class TestAnEmptyClearedPlaceholderIsRefused:
    """The one caller-supplied string that was not validated at all.

    `cleared_placeholder=""` was accepted and produced tool returns with literally empty
    content — upstream assigns the string as given and does not fall back to its own
    default — so the model lost every cue that anything had been cleared. Asymmetric
    with the care taken over `summary_prompt`, and the same class of mistake: a host
    string that reaches the model unchecked.
    """

    def test_an_empty_placeholder_is_refused_at_construction(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match=r"cleared_placeholder must not be empty"):
            build_compaction(_settings(tmp_path), cleared_placeholder="")

    def test_whitespace_only_is_refused_too(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match=r"cleared_placeholder must not be empty"):
            build_compaction(_settings(tmp_path), cleared_placeholder="   \n")

    def test_a_real_placeholder_is_accepted(self, tmp_path: Path) -> None:
        assert build_compaction(_settings(tmp_path), cleared_placeholder="[cleared]")


class TestTheFormatSpecReaderIsGeneralOnPurpose:
    """`_substitutes_whole_transcript` is the width-independent half, tested directly.

    `check_summary_prompt` only ever supplies one field, so a prompt naming any other
    raises `KeyError` at render and never reaches the spec read. The reader is written to
    ignore other fields anyway — a future prompt with a second substitution would
    otherwise be refused for the wrong reason — and that branch is only reachable from
    here.
    """

    @pytest.mark.parametrize(
        ("prompt", "whole"),
        [
            ("{messages}", True),
            ("{messages!r}", True),
            ("{messages:>10}", True),
            ("{messages:.<40}", True),
            ("{messages} plus {other}", True),
            ("{other} alone", True),
            ("{messages:.5}", False),
            ("{messages:.40000}", False),
            ("{messages:>10.50}", False),
            ("{messages[0]}", False),
            ("{messages.upper}", False),
        ],
        ids=[
            "plain",
            "repr",
            "pad",
            "dot-as-fill",
            "another-field-alongside",
            "another-field-alone",
            "precision-small",
            "precision-huge",
            "width-and-precision",
            "index",
            "attribute",
        ],
    )
    def test_it_reads_the_spec_rather_than_sampling_a_rendering(
        self, prompt: str, whole: bool
    ) -> None:
        from elja.compaction import _substitutes_whole_transcript

        assert _substitutes_whole_transcript(prompt) is whole


class TestASystemPromptInHistoryDoesNotGrow:
    """The third irreducible source, which used to GROW, held to a single copy.

    `Agent(system_prompt=...)` puts a `SystemPromptPart` *into* the first request, i.e.
    into history — unlike `instructions=`, which lives outside it. The harness copies
    every leading system part into its new summary message, `preserve_first_user_message`
    keeps the original request too, and pydantic-ai merges consecutive requests when the
    history is replayed — so the next "first request" also held the previous summary and
    copy, and the harness preserved all of it again. Before the fix this measured one more
    policy copy per compacting turn, strictly increasing, until a policy that started under
    target reached the unbounded regime by growth alone.

    elja now removes what compaction manufactured after each summary: exact copies of a
    system part (the first, leading one is kept) and summaries a newer one superseded.
    """

    POLICY = "POLICY: never disclose tenant ids. " * 10

    async def _per_turn(
        self, tmp_path: Path, *, preserve: bool
    ) -> tuple[list[int], list[int], list[list[str]], int]:
        settings = _settings(tmp_path, target=1000)
        summarizations = 0

        def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal summarizations
            if "summarization assistant" in (info.instructions or ""):
                summarizations += 1
                return ModelResponse(
                    parts=[TextPart(content=f"## Intent\nledger {summarizations}")]
                )
            return ModelResponse(parts=[TextPart(content="ok")])

        agent: Agent[EljaDeps, str] = Agent(
            FunctionModel(script),
            deps_type=EljaDeps,
            system_prompt=self.POLICY,
            capabilities=build_compaction(settings, preserve_first_user_message=preserve),
        )
        deps = EljaDeps.from_settings(settings)
        history: list[ModelMessage] | None = None
        policies: list[int] = []
        summaries: list[int] = []
        prompts: list[list[str]] = []
        for turn in range(5):
            result = await agent.run(f"turn {turn}", message_history=history, deps=deps)
            messages = result.all_messages()
            parts = [part for message in messages for part in message.parts]
            policies.append(
                sum(
                    1
                    for part in parts
                    if isinstance(part, SystemPromptPart) and part.content.startswith("POLICY")
                )
            )
            summaries.append(
                sum(
                    1
                    for part in parts
                    if isinstance(part, SystemPromptPart) and part.content.startswith("Summary")
                )
            )
            prompts.append(
                [
                    part.content
                    for part in parts
                    if isinstance(part, UserPromptPart) and isinstance(part.content, str)
                ]
            )
            history = [*messages, *_mixed_history(8)]
        return policies, summaries, prompts, summarizations

    @pytest.mark.parametrize("preserve", [True, False], ids=["preserving", "not-preserving"])
    async def test_one_policy_copy_and_one_summary_however_many_compactions(
        self, tmp_path: Path, preserve: bool
    ) -> None:
        policies, summaries, _, summarizations = await self._per_turn(tmp_path, preserve=preserve)
        assert summarizations >= 3, "compaction did not keep firing; the measurement is vacuous"
        assert policies == [1] * len(policies)
        assert summaries == [0] + [1] * (len(summaries) - 1)

    async def test_the_policy_still_leads_the_conversation(self, tmp_path: Path) -> None:
        """Provenance: the kept copy is the original part, and it comes first."""
        settings = _settings(tmp_path, target=1000)
        seen: list[list[ModelMessage]] = []

        def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if "summarization assistant" in (info.instructions or ""):
                return ModelResponse(parts=[TextPart(content="## Intent\nledger")])
            seen.append(messages)
            return ModelResponse(parts=[TextPart(content="ok")])

        agent: Agent[EljaDeps, str] = Agent(
            FunctionModel(script),
            deps_type=EljaDeps,
            system_prompt=self.POLICY,
            capabilities=build_compaction(settings),
        )
        deps = EljaDeps.from_settings(settings)
        first = await agent.run("turn 0", deps=deps)
        (original,) = [
            part
            for message in first.all_messages()
            for part in message.parts
            if isinstance(part, SystemPromptPart)
        ]
        history = [*first.all_messages(), *_mixed_history(8)]
        for turn in range(1, 3):
            result = await agent.run(f"turn {turn}", message_history=history, deps=deps)
            history = [*result.all_messages(), *_mixed_history(8)]
        leading = seen[-1][0].parts[0]
        assert leading == original

    async def test_no_user_prompt_is_ever_removed(self, tmp_path: Path) -> None:
        """Only compaction's own copies go: every prompt the user wrote is still there.

        These ARE protected-content growth — pydantic-ai merges each compacted turn's
        prompt into the preserved first request on replay — but they are distinct
        messages the user wrote, not copies, so they are kept and the floor
        measurement counts them instead.
        """
        _, _, prompts, summarizations = await self._per_turn(tmp_path, preserve=True)
        assert summarizations >= 3
        for turn, kept in enumerate(prompts):
            assert [f"turn {n}" for n in range(turn + 1)] == [
                p for p in kept if p.startswith("turn")
            ]
