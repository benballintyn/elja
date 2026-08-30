"""Tests for elja.cli."""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import (
    AgentInfo,
    DeltaThinkingCalls,
    DeltaThinkingPart,
    DeltaToolCall,
    DeltaToolCalls,
    FunctionModel,
)
from pytest_mock import MockerFixture

from elja.cli import build_parser, main, repl, run_turn
from elja.deps import EljaDeps
from elja.session import Session
from elja.settings import EljaSettings, LimitsConfig, WorkspaceConfig
from elja.tools import build_toolset

StreamItem = str | DeltaToolCalls | DeltaThinkingCalls


def _streaming_agent(*replies: str) -> Agent[EljaDeps, str]:
    """Streams each reply string character by character."""

    async def sf(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        reply = replies[min(len(messages) // 2, len(replies) - 1)]
        for ch in reply:
            yield ch

    return Agent(FunctionModel(stream_function=sf), deps_type=EljaDeps)


def _tool_calling_agent(settings: EljaSettings) -> Agent[EljaDeps, str]:
    """Narrates, calls list_dir, then answers — the Qwen3.8 interleaving shape."""

    async def sf(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[StreamItem]:
        if len(messages) == 1:
            yield "let me check. "
            yield {1: DeltaToolCall(name="list_dir", json_args='{"path": "."}')}
        else:
            yield "all done"

    return Agent(
        FunctionModel(stream_function=sf),
        deps_type=EljaDeps,
        toolsets=[build_toolset(settings)],
    )


@pytest.fixture
def settings(tmp_path: Path) -> EljaSettings:
    return EljaSettings(workspace=WorkspaceConfig(root=tmp_path))


class TestParser:
    def test_chat_defaults(self) -> None:
        args = build_parser().parse_args(["chat"])
        assert args.command == "chat"
        assert args.config is None
        assert args.session == "default"
        assert args.once is None

    def test_chat_options(self, tmp_path: Path) -> None:
        args = build_parser().parse_args(
            ["chat", "--config", str(tmp_path / "e.toml"), "--session", "s1", "--once", "hi"]
        )
        assert args.config == tmp_path / "e.toml"
        assert args.session == "s1"
        assert args.once == "hi"


class TestRunTurn:
    async def test_streams_persists_and_returns(self, settings: EljaSettings) -> None:
        session = Session.for_name(settings, "t")
        deltas: list[str] = []
        output = await run_turn(
            _streaming_agent("hello world"), settings, session, "hi", deltas.append
        )
        assert output == "hello world"
        assert "".join(deltas) == "hello world"
        assert len(session.load()) == 2

    async def test_second_turn_extends_history(self, settings: EljaSettings) -> None:
        agent = _streaming_agent("again")
        session = Session.for_name(settings, "t")
        await run_turn(agent, settings, session, "one", lambda d: None)
        await run_turn(agent, settings, session, "two", lambda d: None)
        assert len(session.load()) == 4

    async def test_leading_newlines_stripped(self, settings: EljaSettings) -> None:
        """Local models leave blank lines after their thinking block."""
        session = Session.for_name(settings, "t")
        deltas: list[str] = []
        await run_turn(_streaming_agent("\n\nok"), settings, session, "hi", deltas.append)
        assert "".join(deltas) == "ok"

    async def test_thinking_emits_status(self, settings: EljaSettings) -> None:
        async def sf(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[StreamItem]:
            yield {0: DeltaThinkingPart(content="pondering")}
            yield "answer"

        agent: Agent[EljaDeps, str] = Agent(FunctionModel(stream_function=sf), deps_type=EljaDeps)
        session = Session.for_name(settings, "t")
        statuses: list[str] = []
        output = await run_turn(agent, settings, session, "hi", lambda d: None, statuses.append)
        assert output == "answer"
        assert "thinking…" in statuses


class TestRunTurnWithTools:
    async def test_text_alongside_tool_call_does_not_end_run(self, settings: EljaSettings) -> None:
        """Qwen-style narration + tool call in one response must not truncate the run."""
        session = Session.for_name(settings, "tools")
        deltas: list[str] = []
        statuses: list[str] = []
        output = await run_turn(
            _tool_calling_agent(settings),
            settings,
            session,
            "look around",
            deltas.append,
            statuses.append,
        )
        assert output == "all done"
        assert statuses == ["list_dir"]
        assert "let me check. " in "".join(deltas)
        assert "all done" in "".join(deltas)
        # Full history persisted: request, tool-call response, tool return, final.
        assert len(session.load()) == 4


class TestRepl:
    async def test_once_mode(self, settings: EljaSettings, mocker: MockerFixture) -> None:
        mocker.patch("elja.cli.build_agent", return_value=_tool_calling_agent(settings))
        await repl(settings, "s", once="do it")
        # Tool-call turn: request, tool-call response, tool return, final answer.
        assert len(Session.for_name(settings, "s").load()) == 4

    async def test_once_empty_prompt_is_noop(
        self, settings: EljaSettings, mocker: MockerFixture
    ) -> None:
        mocker.patch("elja.cli.build_agent", return_value=_streaming_agent("x"))
        await repl(settings, "s", once="   ")
        assert Session.for_name(settings, "s").load() == []

    async def test_model_text_is_not_rich_markup(
        self,
        settings: EljaSettings,
        mocker: MockerFixture,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Bracketed model output must render verbatim, not as (broken) markup."""
        reply = "see [/usr/local/bin] and list[int] :smile:"
        mocker.patch("elja.cli.build_agent", return_value=_streaming_agent(reply))
        await repl(settings, "s", once="go")
        out = capsys.readouterr().out
        assert "[/usr/local/bin]" in out
        assert "list[int]" in out
        assert ":smile:" in out

    async def test_mid_turn_error_does_not_kill_repl(
        self,
        settings: EljaSettings,
        mocker: MockerFixture,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        async def sf(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
            yield "starting"
            raise ConnectionError("server gone")

        agent: Agent[EljaDeps, str] = Agent(FunctionModel(stream_function=sf), deps_type=EljaDeps)
        mocker.patch("elja.cli.build_agent", return_value=agent)
        prompts = iter(["boom", "exit"])
        await repl(settings, "s", input_fn=lambda _: next(prompts))
        out = capsys.readouterr().out
        assert "server gone" in out
        assert "turn not saved" in out
        assert Session.for_name(settings, "s").load() == []

    async def test_usage_limit_exceeded_is_friendly(
        self,
        settings: EljaSettings,
        mocker: MockerFixture,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        tight = EljaSettings(
            workspace=WorkspaceConfig(root=settings.workspace.root),
            limits=LimitsConfig(request_limit=1),
        )
        mocker.patch("elja.cli.build_agent", return_value=_tool_calling_agent(tight))
        await repl(tight, "s", once="go")
        out = capsys.readouterr().out
        assert "limits.request_limit" in out

    async def test_loop_until_exit(self, settings: EljaSettings, mocker: MockerFixture) -> None:
        mocker.patch("elja.cli.build_agent", return_value=_streaming_agent("resp"))
        prompts = iter(["hello", "   ", "exit"])
        await repl(settings, "s", input_fn=lambda _: next(prompts))
        # Only the non-empty, non-exit prompt produced a turn.
        assert len(Session.for_name(settings, "s").load()) == 2

    async def test_eof_ends_loop(self, settings: EljaSettings, mocker: MockerFixture) -> None:
        mocker.patch("elja.cli.build_agent", return_value=_streaming_agent("resp"))

        def raise_eof(_: str) -> str:
            raise EOFError

        await repl(settings, "s", input_fn=raise_eof)
        assert Session.for_name(settings, "s").load() == []


def test_main_wires_everything(tmp_path: Path, mocker: MockerFixture) -> None:
    config = tmp_path / "elja.toml"
    config.write_text(f'[workspace]\nroot = "{tmp_path}"\n')
    mocker.patch("elja.cli.build_agent", return_value=_streaming_agent("done"))
    mocker.patch(
        "sys.argv",
        ["elja", "chat", "--config", str(config), "--session", "m", "--once", "go"],
    )
    main()
    settings = EljaSettings(workspace=WorkspaceConfig(root=tmp_path))
    assert len(Session.for_name(settings, "m").load()) == 2


def test_main_handles_keyboard_interrupt(
    tmp_path: Path, mocker: MockerFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    mocker.patch("elja.cli.repl", side_effect=KeyboardInterrupt)
    mocker.patch("sys.argv", ["elja", "chat"])
    main()
    assert "interrupted" in capsys.readouterr().out
