"""Tests for elja.cli."""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pytest_mock import MockerFixture

from elja.cli import build_parser, main, repl, run_turn
from elja.deps import EljaDeps
from elja.session import Session
from elja.settings import EljaSettings, WorkspaceConfig


def _streaming_agent(reply: str) -> Agent[EljaDeps, str]:
    async def sf(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        for ch in reply:
            yield ch

    return Agent(FunctionModel(stream_function=sf), deps_type=EljaDeps)


def _tool_calling_agent(settings: EljaSettings) -> Agent[EljaDeps, str]:
    """Narrates, calls list_dir, then answers — the Qwen3.8 interleaving shape."""
    from pydantic_ai.models.function import DeltaToolCall

    from elja.tools import build_toolset

    async def sf(
        messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
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
    async def test_streams_persists_and_returns(
        self, settings: EljaSettings, mocker: MockerFixture
    ) -> None:
        mocker.patch("elja.cli.build_agent", return_value=_streaming_agent("hello world"))
        session = Session.for_name(settings, "t")
        deltas: list[str] = []
        output = await run_turn(settings, session, "hi", deltas.append)
        assert output == "hello world"
        assert "".join(deltas) == "hello world"
        # History persisted: one request + one response.
        assert len(session.load()) == 2

    async def test_second_turn_extends_history(
        self, settings: EljaSettings, mocker: MockerFixture
    ) -> None:
        mocker.patch("elja.cli.build_agent", return_value=_streaming_agent("again"))
        session = Session.for_name(settings, "t")
        await run_turn(settings, session, "one", lambda d: None)
        await run_turn(settings, session, "two", lambda d: None)
        assert len(session.load()) == 4


class TestRunTurnWithTools:
    async def test_text_alongside_tool_call_does_not_end_run(
        self, settings: EljaSettings, mocker: MockerFixture
    ) -> None:
        """Qwen-style narration + tool call in one response must not truncate the run."""
        mocker.patch("elja.cli.build_agent", return_value=_tool_calling_agent(settings))
        session = Session.for_name(settings, "tools")
        deltas: list[str] = []
        tools: list[str] = []
        output = await run_turn(settings, session, "look around", deltas.append, tools.append)
        assert output == "all done"
        assert tools == ["list_dir"]
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
