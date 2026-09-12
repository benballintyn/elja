"""Tests for elja.cli."""

import asyncio
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

    async def test_blank_error_message_falls_back_to_repr(
        self,
        settings: EljaSettings,
        mocker: MockerFixture,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A ClosedResourceError-style exception with empty str() must still be legible."""

        async def sf(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
            yield "x"
            raise ConnectionError()

        agent: Agent[EljaDeps, str] = Agent(FunctionModel(stream_function=sf), deps_type=EljaDeps)
        mocker.patch("elja.cli.build_agent", return_value=agent)
        await repl(settings, "s", once="go")
        assert "ConnectionError()" in capsys.readouterr().out

    async def test_repl_recovers_with_fresh_agent_after_error(
        self, settings: EljaSettings, mocker: MockerFixture
    ) -> None:
        """After a failed turn the next turn runs on a rebuilt agent."""

        async def bad_sf(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
            yield "x"
            raise ConnectionError("gone")

        bad: Agent[EljaDeps, str] = Agent(
            FunctionModel(stream_function=bad_sf), deps_type=EljaDeps
        )
        build = mocker.patch(
            "elja.cli.build_agent", side_effect=[bad, _streaming_agent("recovered")]
        )
        prompts = iter(["boom", "works", "exit"])
        await repl(settings, "s", input_fn=lambda _: next(prompts))
        assert build.call_count == 2
        assert len(Session.for_name(settings, "s").load()) == 2

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
    mocker.patch.dict("os.environ", {"FORCE_COLOR": "1"})
    mocker.patch("elja.cli.repl", side_effect=KeyboardInterrupt)
    mocker.patch("sys.argv", ["elja", "chat"])
    main()
    out = capsys.readouterr().out
    assert "interrupted" in out
    # Through `show_on` like every other diagnostic, not a bare `print`. The text is
    # fixed so the flags cannot bite today; the style is what makes the routing
    # observable, and this is the site where a future edit interpolates a path.
    assert "\x1b[33m" in out


class TestABrokenStatusSinkCannotKillATurn:
    """Status is display telemetry; enforcement belongs in a model wrapper.

    Before this, a sink that raised aborted `run_turn` *and* took the turn's
    history with it, because the exception escaped before the session was saved.
    """

    async def test_a_raising_sink_neither_aborts_the_run_nor_loses_the_history(
        self, tmp_path: Path
    ) -> None:
        settings = EljaSettings(workspace=WorkspaceConfig(root=tmp_path))
        (tmp_path / "a.txt").write_text("hello")
        turns: list[int] = []

        async def sf(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[StreamItem]:
            turns.append(1)
            if len(turns) == 1:
                # BOTH status sites, in order. There are two — a thinking part and a
                # streamed tool call — and a test driving only the tool call leaves
                # the other's guard unpinned: removing it survived the whole suite.
                # A sink that raises on the FIRST label is what makes the thinking
                # site's guard load-bearing.
                yield {0: DeltaThinkingPart(content="which directory")}
                yield {1: DeltaToolCall(name="list_dir", json_args='{"path": "."}')}
            else:
                yield "done"

        labels: list[str] = []

        def broken_sink(label: str) -> None:
            labels.append(label)
            raise RuntimeError(f"sink died on {label!r}")

        agent: Agent[EljaDeps, str] = Agent(
            FunctionModel(stream_function=sf),
            deps_type=EljaDeps,
            toolsets=[build_toolset(settings)],
        )
        session = Session(tmp_path / "s.json")
        answer = await run_turn(
            agent,
            settings,
            session,
            "go",
            on_delta=lambda _d: None,
            on_status=broken_sink,
        )
        assert answer == "done"
        # The sink really was reached, and raised, at BOTH sites, and neither
        # killed the turn nor took its history with it.
        assert labels == ["thinking…", "list_dir"]
        assert session.load(), "the turn's history was lost with the sink"

    def test_notify_tolerates_no_sink_at_all(self) -> None:
        from elja.deps import notify

        notify(None, "label")  # must not raise

    def test_notify_passes_the_label_through_when_the_sink_works(self) -> None:
        from elja.deps import notify

        seen: list[str] = []
        notify(seen.append, "thinking…")
        assert seen == ["thinking…"]

    @pytest.mark.parametrize("escaping", [asyncio.CancelledError, KeyboardInterrupt, SystemExit])
    def test_notify_does_not_swallow_a_base_exception(self, escaping: type[BaseException]) -> None:
        """Cancellation is the host's; telemetry must not eat it.

        suppress(Exception) rather than suppress(BaseException) is the whole
        point, and widening it would have passed a green suite.
        """
        from elja.deps import notify

        def raising(_label: str) -> None:
            raise escaping

        with pytest.raises(escaping):
            notify(raising, "x")


class TestTheDeltaSinkIsDeliberatelyNotSuppressed:
    """The other half of an asymmetry the docs state as intentional.

    `docs/EMBEDDING.md` says the status sink is suppressed and the text-delta sink
    is not — "two sinks, two different answers, on purpose". Only the suppressed
    half was tested. With `notify` now living in the same module, "make all the
    sinks safe" is the obvious next refactor, and it would silently invert a
    documented contract.
    """

    async def test_a_raising_delta_sink_aborts_the_turn_and_its_history_is_not_saved(
        self, tmp_path: Path
    ) -> None:
        """Swallowing the model's own output would be worse than failing loudly."""
        settings = EljaSettings(workspace=WorkspaceConfig(root=tmp_path))

        async def sf(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[StreamItem]:
            yield "the answer"

        def broken_delta(_delta: str) -> None:
            raise RuntimeError("stdout closed")

        agent: Agent[EljaDeps, str] = Agent(FunctionModel(stream_function=sf), deps_type=EljaDeps)
        session = Session(tmp_path / "s.json")
        with pytest.raises(RuntimeError, match="stdout closed"):
            await run_turn(
                agent, settings, session, "go", on_delta=broken_delta, on_status=lambda _s: None
            )
        # History is saved only on success, so a dead output sink loses the turn —
        # which is the documented trade, not an accident.
        assert session.load() == []


class TestShowOn:
    """Every flag on the diagnostic printer, pinned on a forced terminal.

    A captured non-tty emits no colour, so `style` and `highlight` are invisible
    through capsys — which is why both went unpinned until this helper moved to
    module level.
    """

    @staticmethod
    def _render(message: str, style: str | None = None, width: int = 40) -> str:
        """Render through `show_on`. `style=None` exercises its own default."""
        from io import StringIO

        from rich.console import Console

        from elja.cli import show_on

        buffer = StringIO()
        # Every knob pinned rather than detected: `force_terminal=True` is what makes
        # rich consult the environment at all, and then `TERM`, `NO_COLOR` and
        # `TTY_COMPATIBLE` each decide whether these assertions can pass. Stating
        # `color_system` and `no_color` makes this class immune to any of them,
        # present or future, instead of relying on the fixture to strip the ones we
        # currently know about.
        console = Console(
            file=buffer,
            force_terminal=True,
            width=width,
            color_system="standard",
            no_color=False,
        )
        if style is None:
            show_on(console, message)
        else:
            show_on(console, message, style)
        return buffer.getvalue()

    def test_a_long_path_is_not_broken_mid_token(self) -> None:
        path = "/private/var/folders/6h/tmpqjz2q66_/skills/broken.md"
        assert path in self._render(f"cannot start agent: invalid skill file {path}", "red")

    def test_markup_in_the_message_is_not_interpreted(self) -> None:
        out = self._render("invalid skill file [bold]notes.md", "red")
        assert "[bold]notes.md" in out

    def test_emoji_shortcodes_are_not_interpreted(self) -> None:
        out = self._render("bad image: :camera:.png is not a file", "red")
        assert ":camera:.png" in out
        assert "📷" not in out

    def test_the_style_reaches_the_terminal(self) -> None:
        """Red for an error, yellow for a warning — distinguishable, not decorative."""
        red = self._render("boom", "red")
        yellow = self._render("boom", "yellow")
        assert "\x1b[31m" in red
        assert "\x1b[33m" in yellow
        assert red != yellow

    def test_the_default_style_is_the_error_one(self) -> None:
        """Every call site that passes no style is reporting a failure.

        Pinned separately because passing `"red"` explicitly, as the test above
        does, leaves the default free to be anything.
        """
        assert "\x1b[31m" in self._render("boom")

    def test_numbers_and_quotes_are_not_separately_highlighted(self) -> None:
        """highlight=False keeps one colour instead of fragmenting the message."""
        out = self._render("read 42 bytes from 'a.md'", "red")
        # One style-open sequence for the whole line, not one per token.
        assert out.count("\x1b[") == 2, out


class TestTheReplsOwnDiagnosticsReachTheTerminal:
    """Style and width at repl's call sites, not only on the helper.

    `TestShowOn` pins every flag on `show_on` itself. Nothing pinned that repl's
    own sites pass the right style or go through the door at all: five mutants
    that dropped a `"yellow"` argument — or the `style` parameter from the helper
    call — survived the whole suite, which would make every warning
    indistinguishable from a fatal error.

    A captured non-tty emits no colour, which is why those mutants were invisible.
    `FORCE_COLOR` makes rich emit SGR through a plain file while `COLUMNS` still
    decides the width, so both are observable through capsys.
    """

    @staticmethod
    def _forced(mocker: MockerFixture, width: str = "40") -> None:
        mocker.patch.dict("os.environ", {"FORCE_COLOR": "1", "COLUMNS": width})

    @staticmethod
    def _line(out: str, needle: str) -> str:
        match = [line for line in out.splitlines() if needle in line]
        assert match, f"{needle!r} never printed:\n{out}"
        return match[0]

    async def test_a_warning_is_yellow_and_an_error_is_red(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], mocker: MockerFixture
    ) -> None:
        """The distinction the style exists to make, driven through the REPL."""
        self._forced(mocker)
        settings = EljaSettings(workspace=WorkspaceConfig(root=tmp_path))
        prompts = iter(["/img x", f"/img {tmp_path / 'nope.png'} describe", "exit"])
        await repl(settings, "s", input_fn=lambda _: next(prompts))
        out = capsys.readouterr().out
        assert "\x1b[33m" in self._line(out, "usage: /img")
        assert "\x1b[31m" in self._line(out, "image not found")

    async def test_a_status_label_is_not_broken_mid_token(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], mocker: MockerFixture
    ) -> None:
        """A status label carries a tool name, and a sub-agent's carries a config name.

        Long MCP tool names exceed a narrow pane easily, and the status sink used to
        print around `show_on` with neither `soft_wrap` nor `emoji=False`.
        """
        from tests.conftest import narrow_terminal

        self._forced(mocker)
        long_tool = "filesystem_read_multiple_text_files_with_encoding"
        narrow_terminal(mocker, long_tool)
        calls: list[int] = []

        async def stream(
            messages: list[ModelMessage], info: AgentInfo
        ) -> AsyncIterator[StreamItem]:
            calls.append(1)
            if len(calls) == 1:
                yield {1: DeltaToolCall(name=long_tool, json_args="{}")}
            else:
                yield "done"

        settings = EljaSettings(workspace=WorkspaceConfig(root=tmp_path))
        agent: Agent[EljaDeps, str] = Agent(
            FunctionModel(stream_function=stream),
            deps_type=EljaDeps,
            toolsets=[build_toolset(settings)],
        )
        mocker.patch("elja.cli.build_agent", return_value=agent)
        await repl(settings, "s", once="go")
        out = capsys.readouterr().out
        assert long_tool in out
        # Dim, not the helper's default red. Dropping the style here left the suite
        # green while every ordinary tool call printed in error red — a user watching
        # a normal turn would read it as a failed one.
        assert "\x1b[2m" in self._line(out, "⚙")

    async def test_an_emoji_shortcode_in_a_status_label_stays_literal(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], mocker: MockerFixture
    ) -> None:
        """A sub-agent label is `f"{name} → {tool}"`, and `name` comes from elja.toml."""
        self._forced(mocker, width="200")
        calls: list[int] = []

        async def stream(
            messages: list[ModelMessage], info: AgentInfo
        ) -> AsyncIterator[StreamItem]:
            calls.append(1)
            if len(calls) == 1:
                yield {1: DeltaToolCall(name="ops:fire:_reader", json_args="{}")}
            else:
                yield "done"

        settings = EljaSettings(workspace=WorkspaceConfig(root=tmp_path))
        agent: Agent[EljaDeps, str] = Agent(
            FunctionModel(stream_function=stream),
            deps_type=EljaDeps,
            toolsets=[build_toolset(settings)],
        )
        mocker.patch("elja.cli.build_agent", return_value=agent)
        await repl(settings, "s", once="go")
        out = capsys.readouterr().out
        assert "ops:fire:_reader" in out
        assert "🔥" not in out

    async def test_model_output_is_neither_wrapped_nor_interpreted(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], mocker: MockerFixture
    ) -> None:
        """Streamed deltas are data too, and a path in one breaks the same way.

        This is the only site whose text comes from the MODEL rather than from config
        or from elja's own f-strings, which makes it the likeliest source of a stray
        `:100:` in prose — and `emoji=False` here was the one flag left unpinned after
        two rounds of claiming the set was complete.
        """
        self._forced(mocker)
        path = "/private/var/folders/6h/tmpqjz2q66_/skills/broken.md"

        async def stream(
            messages: list[ModelMessage], info: AgentInfo
        ) -> AsyncIterator[StreamItem]:
            yield f"I read the skill at {path} and [bold]notes.md and notes:100:.md too"

        settings = EljaSettings(workspace=WorkspaceConfig(root=tmp_path))
        agent: Agent[EljaDeps, str] = Agent(
            FunctionModel(stream_function=stream), deps_type=EljaDeps
        )
        mocker.patch("elja.cli.build_agent", return_value=agent)
        await repl(settings, "s", once="go")
        out = capsys.readouterr().out
        assert path in out
        assert "[bold]notes.md" in out
        assert "notes:100:.md" in out
        assert "💯" not in out


class TestTheBannerDoesNotInterpretConfigValues:
    """The banner is the one line that legitimately uses markup, over config values.

    Four lines after a comment promising no tracebacks, a model name containing a
    closing tag raised an uncaught `MarkupError` on REPL start, and an IPv6
    `base_url` had its bracketed host eaten — showing an endpoint that was not the
    one in use to someone debugging connectivity.
    """

    async def test_a_model_name_carrying_a_tag_does_not_crash_the_repl(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        settings = EljaSettings(
            workspace=WorkspaceConfig(root=tmp_path),
            model={"name": "model[/bold]x"},  # type: ignore[arg-type]
        )
        prompts = iter(["exit"])
        await repl(settings, "s", input_fn=lambda _: next(prompts))  # must not raise
        assert "model[/bold]x" in capsys.readouterr().out

    async def test_an_ipv6_endpoint_is_shown_as_configured(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """rich ate the bracketed host, so the banner named a different endpoint."""
        settings = EljaSettings(
            workspace=WorkspaceConfig(root=tmp_path),
            model={"base_url": "http://[fe80::1]:1234/v1"},  # type: ignore[arg-type]
        )
        prompts = iter(["exit"])
        await repl(settings, "s", input_fn=lambda _: next(prompts))
        assert "[fe80::1]:1234" in capsys.readouterr().out

    async def test_an_emoji_shortcode_in_a_model_name_stays_literal(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """`ModelConfig.name` is a bare string, and this is the line that shows it.

        Markup is deliberately on here, which is why the values are escaped — but
        escaping does not stop shortcode expansion, and a banner reading
        `qwen3💯instruct` names a model that does not exist to the person reading it
        precisely to check which model they are on.
        """
        settings = EljaSettings(
            workspace=WorkspaceConfig(root=tmp_path),
            model={"name": "qwen3:100:instruct"},  # type: ignore[arg-type]
        )
        prompts = iter(["exit"])
        await repl(settings, "s", input_fn=lambda _: next(prompts))
        out = capsys.readouterr().out
        assert "qwen3:100:instruct" in out
        assert "💯" not in out

    async def test_a_long_endpoint_is_not_folded_mid_token(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], mocker: MockerFixture
    ) -> None:
        """Same hazard as every other diagnostic, on the one line markup still runs on."""
        from tests.conftest import narrow_terminal

        endpoint = "http://ml-inference-gateway.internal-hostname.corp.example.com:11434/v1"
        # Through the shared guard, so this asserts the RELATION rather than trusting
        # that the example stays longer than 40 columns. It was the sixth
        # width-sensitive site and the only one still setting COLUMNS by hand.
        narrow_terminal(mocker, endpoint)
        settings = EljaSettings(
            workspace=WorkspaceConfig(root=tmp_path),
            model={"base_url": endpoint},  # type: ignore[arg-type]
        )
        prompts = iter(["exit"])
        await repl(settings, "s", input_fn=lambda _: next(prompts))
        assert endpoint in capsys.readouterr().out
