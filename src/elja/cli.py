"""The elja CLI: a thin REPL over the Python API.

The Python API (:func:`elja.agent.build_agent` et al.) is the product; this
module is a demo shell for driving it interactively against a local model.
"""

import argparse
import asyncio
from collections.abc import Callable
from pathlib import Path

from pydantic_ai import Agent, AgentRunResult, AgentRunResultEvent
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
)
from rich.console import Console

from elja.agent import build_agent, build_usage_limits
from elja.deps import EljaDeps
from elja.mcp import build_mcp_toolsets, preflight_mcp_toolsets
from elja.session import Session
from elja.settings import EljaSettings, load_settings


def build_parser() -> argparse.ArgumentParser:
    """The elja command-line interface definition."""
    parser = argparse.ArgumentParser(prog="elja", description="Run an elja agent.")
    sub = parser.add_subparsers(dest="command", required=True)
    chat = sub.add_parser("chat", help="Chat with the agent (REPL, or one-shot with --once).")
    chat.add_argument("--config", type=Path, default=None, help="Path to elja.toml.")
    chat.add_argument("--session", default="default", help="Session name to resume/save.")
    chat.add_argument("--once", default=None, help="Run a single prompt and exit.")
    return parser


async def run_turn(
    agent: Agent[EljaDeps, str],
    settings: EljaSettings,
    session: Session,
    prompt: str,
    on_delta: Callable[[str], None],
    on_status: Callable[[str], None] | None = None,
) -> str:
    """Run one conversational turn, streaming output and persisting history.

    Streams via agent events rather than ``run_stream``: models like Qwen3.8
    narrate text alongside their tool calls, which ``run_stream`` would
    mistake for the final answer and end the run mid-loop.

    Args:
        agent: The agent to run (build once, reuse across turns).
        settings: Resolved elja settings.
        session: The session whose history to extend.
        prompt: The user's message.
        on_delta: Called with each streamed text fragment (including mid-run
            narration between tool calls).
        on_status: Called with a short label when a tool call starts or the
            model begins thinking, so the CLI never looks hung.

    Returns:
        The final response text.
    """
    deps = EljaDeps.from_settings(settings)
    history = session.load()
    result: AgentRunResult[str] | None = None
    started = False

    def emit(text: str) -> None:
        nonlocal started
        if not started:
            # Drop the blank lines local models leave after their thinking block.
            text = text.lstrip("\n")
            if not text:
                return
            started = True
        on_delta(text)

    async with agent.run_stream_events(
        prompt,
        deps=deps,
        message_history=history,
        usage_limits=build_usage_limits(settings),
    ) as events:
        async for event in events:
            if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
                emit(event.part.content)
            elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                emit(event.delta.content_delta)
            elif isinstance(event, PartStartEvent) and isinstance(event.part, ThinkingPart):
                if on_status is not None:
                    on_status("thinking…")
            elif isinstance(event, FunctionToolCallEvent):
                if on_status is not None:
                    on_status(event.part.tool_name)
            elif isinstance(event, AgentRunResultEvent):
                result = event.result
    if result is None:  # pragma: no cover - failures re-raise from the iterator
        raise RuntimeError("event stream ended without a result")
    session.save(list(result.all_messages()))
    return result.output


async def repl(
    settings: EljaSettings,
    session_name: str,
    once: str | None = None,
    input_fn: Callable[[str], str] = input,
) -> None:
    """Interactive chat loop (or a single turn when ``once`` is given)."""
    console = Console()
    session = Session.for_name(settings, session_name)
    # Connect MCP servers up front: failures are named and dropped so one bad
    # entry can't poison every turn.
    mcp_toolsets = await preflight_mcp_toolsets(
        build_mcp_toolsets(settings),
        lambda name, err: console.print(
            f"warning: MCP server {name!r} unavailable, skipping: {err}",
            style="yellow",
            markup=False,
        ),
    )
    alive = {t.id for t in mcp_toolsets}

    def fresh_agent() -> "Agent[EljaDeps, str]":
        # Fresh toolsets (filtered to servers that passed preflight): a server
        # that dies mid-call wedges its transport, so errors get new ones.
        return build_agent(
            settings,
            mcp_toolsets=[t for t in build_mcp_toolsets(settings) if t.id in alive],
        )

    agent = fresh_agent()

    def show_delta(delta: str) -> None:
        # Model output is data: never let rich interpret [brackets] as markup.
        console.print(delta, end="", markup=False, highlight=False, emoji=False)

    def show_status(label: str) -> None:
        console.print(f"\n⚙ {label}", style="dim", markup=False, highlight=False)

    async def do_turn(prompt: str) -> None:
        # A failed turn must never kill the REPL: report, drop the turn
        # (history is only saved on success, atomically), and keep going.
        nonlocal agent
        try:
            await run_turn(agent, settings, session, prompt, show_delta, show_status)
        except UsageLimitExceeded as exc:
            console.print(
                f"\nerror: {exc} — raise limits.request_limit in elja.toml to allow "
                "longer runs (turn not saved)",
                style="red",
                markup=False,
            )
        except Exception as exc:
            console.print(
                f"\nerror: {str(exc) or exc!r} (turn not saved)", style="red", markup=False
            )
            agent = fresh_agent()
        console.print()

    if once is not None:
        if once.strip():
            await do_turn(once)
        return
    console.print(
        f"[bold]elja[/bold] — model [cyan]{settings.model.name}[/cyan] at "
        f"{settings.model.base_url} (session: {session_name}; exit/quit to leave)"
    )
    while True:
        try:
            prompt = input_fn("> ")
        except (EOFError, KeyboardInterrupt):
            break
        if prompt.strip() in {"exit", "quit"}:
            break
        if not prompt.strip():
            continue
        await do_turn(prompt)


def main() -> None:
    """Console-script entry point."""
    args = build_parser().parse_args()
    settings = load_settings(args.config)
    try:
        asyncio.run(repl(settings, args.session, once=args.once))
    except KeyboardInterrupt:
        print("\ninterrupted")


if __name__ == "__main__":  # pragma: no cover
    main()
