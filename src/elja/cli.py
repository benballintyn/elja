"""The elja CLI: a thin REPL over the Python API.

The Python API (:func:`elja.agent.build_agent` et al.) is the product; this
module is a demo shell for driving it interactively against a local model.
"""

import argparse
import asyncio
from collections.abc import Callable
from pathlib import Path

from pydantic_ai.messages import (
    FunctionToolCallEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
)
from pydantic_ai.run import AgentRunResult, AgentRunResultEvent
from rich.console import Console

from elja.agent import build_agent, build_usage_limits
from elja.deps import EljaDeps
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
    settings: EljaSettings,
    session: Session,
    prompt: str,
    on_delta: Callable[[str], None],
    on_tool: Callable[[str], None] | None = None,
) -> str:
    """Run one conversational turn, streaming output and persisting history.

    Streams via agent events rather than ``run_stream``: models like Qwen3.8
    narrate text alongside their tool calls, which ``run_stream`` would
    mistake for the final answer and end the run mid-loop.

    Args:
        settings: Resolved elja settings.
        session: The session whose history to extend.
        prompt: The user's message.
        on_delta: Called with each streamed text fragment (including mid-run
            narration between tool calls).
        on_tool: Called with the tool name each time a tool call starts.

    Returns:
        The final response text.
    """
    agent = build_agent(settings)
    deps = EljaDeps.from_settings(settings)
    history = session.load()
    result: AgentRunResult[str] | None = None
    async with agent.run_stream_events(
        prompt,
        deps=deps,
        message_history=history,
        usage_limits=build_usage_limits(settings),
    ) as events:
        async for event in events:
            if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
                on_delta(event.part.content)
            elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                on_delta(event.delta.content_delta)
            elif isinstance(event, FunctionToolCallEvent) and on_tool is not None:
                on_tool(event.part.tool_name)
            elif isinstance(event, AgentRunResultEvent):
                result = event.result
    assert result is not None, "event stream ended without a result"
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

    def show_delta(delta: str) -> None:
        console.print(delta, end="")

    def show_tool(name: str) -> None:
        console.print(f"\n[dim]⚙ {name}[/dim]")

    if once is not None:
        await run_turn(settings, session, once, show_delta, on_tool=show_tool)
        console.print()
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
        await run_turn(settings, session, prompt, show_delta, on_tool=show_tool)
        console.print()


def main() -> None:
    """Console-script entry point."""
    args = build_parser().parse_args()
    settings = load_settings(args.config)
    asyncio.run(repl(settings, args.session, once=args.once))


if __name__ == "__main__":  # pragma: no cover
    main()
