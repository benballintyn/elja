"""Tests for elja.compaction."""

from pathlib import Path

import pytest
from pydantic_ai import Agent
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

from elja.compaction import build_compaction
from elja.deps import EljaDeps
from elja.settings import CompactionConfig, EljaSettings, WorkspaceConfig


def _history_with_tool_pairs(n: int, result_size: int = 400) -> list[ModelMessage]:
    """A synthetic transcript of n tool call/return pairs with chunky results."""
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="original task: audit the files")])
    ]
    for i in range(n):
        messages.append(
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="read_file", args={"path": f"f{i}.txt"}, tool_call_id=f"c{i}"
                    )
                ]
            )
        )
        messages.append(
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name="read_file",
                        content=f"data{i} " * result_size,
                        tool_call_id=f"c{i}",
                    )
                ]
            )
        )
    return messages


class TestConfig:
    def test_defaults(self) -> None:
        cfg = EljaSettings().compaction
        assert cfg.enabled is True
        assert cfg.target_tokens == 24_000
        assert cfg.keep_tool_pairs == 10
        assert cfg.keep_messages == 20

    def test_disabled_builds_nothing(self) -> None:
        settings = EljaSettings(compaction=CompactionConfig(enabled=False))
        assert build_compaction(settings) == []

    def test_enabled_builds_one_tiered_capability(self) -> None:
        (cap,) = build_compaction(EljaSettings())
        assert type(cap).__name__ == "TieredCompaction"


class TestMaskingBehavior:
    async def test_old_tool_results_cleared_recent_kept(self, tmp_path: Path) -> None:
        """Old observations are masked; the recent tail and all actions survive."""
        settings = EljaSettings(
            workspace=WorkspaceConfig(root=tmp_path),
            compaction=CompactionConfig(target_tokens=2000, keep_tool_pairs=2),
        )
        seen: list[list[ModelMessage]] = []

        def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen.append(list(messages))
            return ModelResponse(parts=[TextPart(content="done")])

        agent: Agent[EljaDeps, str] = Agent(
            FunctionModel(script),
            deps_type=EljaDeps,
            capabilities=build_compaction(settings),
        )
        history = _history_with_tool_pairs(6)
        result = await agent.run(
            "continue", message_history=history, deps=EljaDeps.from_settings(settings)
        )
        assert result.output == "done"
        sent = seen[0]
        returns = [p for m in sent for p in m.parts if isinstance(p, ToolReturnPart)]
        assert len(returns) == 6
        cleared = [r for r in returns if "cleared" in str(r.content)]
        intact = [r for r in returns if "cleared" not in str(r.content)]
        assert len(cleared) >= 3  # old observations masked
        assert len(intact) >= 2  # recent tail kept verbatim
        # The last pairs specifically are the intact ones.
        assert "data5" in str(returns[-1].content)
        # Actions (tool calls) all survive masking.
        calls = [p for m in sent for p in m.parts if isinstance(p, ToolCallPart)]
        assert len(calls) == 6
        # The original task (first user message) survives.
        assert any(
            "original task" in str(p.content)
            for m in sent
            for p in m.parts
            if isinstance(p, UserPromptPart)
        )

    async def test_under_target_untouched(self, tmp_path: Path) -> None:
        settings = EljaSettings(
            workspace=WorkspaceConfig(root=tmp_path),
            compaction=CompactionConfig(target_tokens=1_000_000),
        )
        seen: list[list[ModelMessage]] = []

        def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen.append(list(messages))
            return ModelResponse(parts=[TextPart(content="ok")])

        agent: Agent[EljaDeps, str] = Agent(
            FunctionModel(script),
            deps_type=EljaDeps,
            capabilities=build_compaction(settings),
        )
        history = _history_with_tool_pairs(3)
        await agent.run("go", message_history=history, deps=EljaDeps.from_settings(settings))
        returns = [p for m in seen[0] for p in m.parts if isinstance(p, ToolReturnPart)]
        assert all("cleared" not in str(r.content) for r in returns)


class TestSummarizationTier:
    async def test_summarizer_runs_on_inherited_model_and_persists(self, tmp_path: Path) -> None:
        """Tier 2 fires when masking can't reach target; summary replaces old history."""
        settings = EljaSettings(
            workspace=WorkspaceConfig(root=tmp_path),
            compaction=CompactionConfig(target_tokens=1000, keep_tool_pairs=1, keep_messages=2),
        )
        prompts: list[str] = []

        def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if "summarization assistant" in (info.instructions or ""):
                prompts.append("summarizer")
                return ModelResponse(parts=[TextPart(content="## Intent\naudit files")])
            prompts.append("agent")
            return ModelResponse(parts=[TextPart(content="done")])

        agent: Agent[EljaDeps, str] = Agent(
            FunctionModel(script),
            deps_type=EljaDeps,
            capabilities=build_compaction(settings),
        )
        history = _history_with_tool_pairs(8, result_size=600)
        result = await agent.run(
            "continue", message_history=history, deps=EljaDeps.from_settings(settings)
        )
        assert result.output == "done"
        assert "summarizer" in prompts
        # The compacted (summarized) history is what the run now carries.
        all_after = result.all_messages()
        assert len(all_after) < len(history)

    async def test_no_repeated_summarization_when_tail_bounded(self, tmp_path: Path) -> None:
        """Stuck-state regression: the summarizer must not re-fire every request."""
        settings = EljaSettings(
            workspace=WorkspaceConfig(root=tmp_path),
            compaction=CompactionConfig(target_tokens=1000, keep_tool_pairs=1, keep_messages=2),
        )
        summarizer_calls: list[int] = []
        agent_calls: list[int] = []

        def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if "summarization assistant" in (info.instructions or ""):
                summarizer_calls.append(1)
                return ModelResponse(parts=[TextPart(content="## Intent\nshort summary")])
            agent_calls.append(1)
            return ModelResponse(parts=[TextPart(content="ok")])

        agent: Agent[EljaDeps, str] = Agent(
            FunctionModel(script),
            deps_type=EljaDeps,
            capabilities=build_compaction(settings),
        )
        deps = EljaDeps.from_settings(settings)
        history = _history_with_tool_pairs(8, result_size=600)
        result = await agent.run("one", message_history=history, deps=deps)
        first_round = len(summarizer_calls)
        # Continue from the compacted history: no new bulk, no new summarization.
        result2 = await agent.run("two", message_history=list(result.all_messages()), deps=deps)
        assert result2.output == "ok"
        assert len(summarizer_calls) == first_round

    async def test_streaming_path_persists_compacted_session(self, tmp_path: Path) -> None:
        """The CLI's run_turn (event streaming) saves the compacted history."""
        from collections.abc import AsyncIterator

        from elja.cli import run_turn
        from elja.session import Session

        settings = EljaSettings(
            workspace=WorkspaceConfig(root=tmp_path),
            compaction=CompactionConfig(target_tokens=2000, keep_tool_pairs=2),
        )

        async def sf(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
            yield "streamed-done"

        agent: Agent[EljaDeps, str] = Agent(
            FunctionModel(stream_function=sf),
            deps_type=EljaDeps,
            capabilities=build_compaction(settings),
        )
        session = Session.for_name(settings, "c")
        session.save(_history_with_tool_pairs(6))
        output = await run_turn(agent, settings, session, "continue", lambda d: None)
        assert output == "streamed-done"
        assert "cleared" in session.path.read_text()


class TestAgentWiring:
    def test_build_agent_includes_compaction(self, tmp_path: Path) -> None:
        from elja.agent import build_agent

        settings = EljaSettings(workspace=WorkspaceConfig(root=tmp_path))
        agent = build_agent(settings)  # must not raise; capability composed in
        assert agent is not None

    def test_placeholder_mentions_rerun(self) -> None:
        """The masking placeholder tells the model how to recover the data."""
        from elja.compaction import CLEARED_PLACEHOLDER

        assert "re-run" in CLEARED_PLACEHOLDER


@pytest.mark.integration
async def test_live_compaction_survives_turn(tmp_path: Path) -> None:
    """With a tiny budget, a multi-tool session compacts and the next turn still works."""
    from elja.agent import build_agent
    from elja.cli import run_turn
    from elja.session import Session

    settings = EljaSettings(
        workspace=WorkspaceConfig(root=tmp_path),
        compaction=CompactionConfig(target_tokens=1000, keep_tool_pairs=1, keep_messages=4),
    )
    for i in range(4):
        (tmp_path / f"note{i}.txt").write_text(f"note {i}: the magic number is {i * 11}\n" * 30)
    session = Session.for_name(settings, "compact")
    agent = build_agent(settings)
    await run_turn(
        agent,
        settings,
        session,
        "Read note0.txt, note1.txt, note2.txt and note3.txt one by one.",
        lambda d: None,
    )
    output = await run_turn(
        agent, settings, session, "Say DONE if you can still respond.", lambda d: None
    )
    assert output  # the agent survived compaction and answered
    # Compaction actually happened: cleared placeholders or a summary in the saved history.
    raw = session.path.read_text()
    assert ("cleared" in raw) or ("summary" in raw.lower())
