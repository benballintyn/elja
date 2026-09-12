"""E4 acceptance: composing compaction for a host that owns its own tools.

The request's script, as a test: a transcript carrying both read and write tool
results is forced through compaction, and then every property a host depends on
is checked — the write outcomes stay recoverable, no mutation rerun is
solicited, pins survive, call/result pairing is valid, an unfittable protected
floor does not silently drop it, and the summarizer is bounded.
"""

from pathlib import Path

import pytest
from pydantic_ai import Agent
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
from pydantic_ai_harness.compaction import ReportContextUsage, is_pinned, pin

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
        if "summarization assistant" in (info.instructions or "") or "SUMMARIZE" in (
            info.instructions or ""
        ):
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
        views, _ = await _drive(settings, list(build_compaction(settings)), _mixed_history(8))
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
        views, _ = await _drive(
            settings,
            list(build_compaction(settings)),
            _mixed_history(8, pinned="NEVER DROP: tenant=acme, currency=USD"),
        )
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
        views, _ = await _drive(
            settings, list(build_compaction(settings)), _mixed_history(8, pinned=giant)
        )
        assert "IRREDUCIBLE CONSTRAINT" in _rendered(views)
        pinned = [
            part
            for message in views[0]
            for part in getattr(message, "parts", [])
            if is_pinned(part)
        ]
        assert len(pinned) == 1


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
        assert "History before this point" in _rendered(views)

    async def test_receipts_are_off_by_default(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path, target=1000)
        views, roles = await _drive(settings, list(build_compaction(settings)), _mixed_history(8))
        assert "summarizer" in roles
        assert "History before this point" not in _rendered(views)

    async def test_the_summarizer_is_called_at_most_once_per_turn(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path, target=1000)
        _, roles = await _drive(settings, list(build_compaction(settings)), _mixed_history(8))
        assert roles.count("summarizer") == 1


class TestRetentionKnobsActuallyRetain:
    """Two config knobs whose effect nothing observed."""

    async def test_the_token_bound_dominates_keep_messages(self, tmp_path: Path) -> None:
        """``keep_messages`` is an upper bound the token bound usually pre-empts.

        elja always sets ``keep_tokens = target_tokens // 3``, and the verbatim
        tail is whichever bound binds first. Measured across three regimes — few
        large messages, many small ones, and the default 24k target — the token
        bound always won, so the knob is inert at elja's settings. Asserted as
        equality rather than left unobserved: if a future change makes the
        message count bind, this fails and the docs need updating with it.
        """
        outcomes: list[int] = []
        for keep_messages in (2, 30):
            settings = _settings(tmp_path, target=1000, keep_messages=keep_messages)
            views, roles = await _drive(
                settings, list(build_compaction(settings)), _mixed_history(20)
            )
            assert "summarizer" in roles
            outcomes.append(len(views[0]))
        assert outcomes[0] == outcomes[1], outcomes

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
        assert "<previous-summary>" in summarizer_prompts[-1]
        assert "summary 1" in summarizer_prompts[-1]


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
