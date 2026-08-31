"""The elja CLI: a thin REPL over the Python API.

The Python API (:func:`elja.agent.build_agent` et al.) is the product; this
module is a demo shell for driving it interactively against a local model.
"""

import argparse
import asyncio
import shlex
from collections.abc import Callable, Sequence
from pathlib import Path

from pydantic_ai import Agent, AgentRunResult, AgentRunResultEvent, BinaryContent
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
from elja.mcp import build_mcp_toolsets, preflight_mcp_toolsets, toolset_name
from elja.model import effective_endpoint
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
    chat.add_argument("--image", type=Path, default=None, help="Attach an image to --once.")
    return parser


MAX_IMAGE_BYTES = 10 * 1024 * 1024

# Magic-byte signatures for the raster types LM Studio's OpenAI-compatible
# endpoint actually accepts; extensions lie (.jfif, renamed files, .heic).
_IMAGE_MAGIC: list[tuple[bytes, str]] = [
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF8", "image/gif"),
    (b"RIFF", "image/webp"),  # RIFF container; WEBP tag checked below
]


def _sniff_media_type(header: bytes) -> str | None:
    for magic, media_type in _IMAGE_MAGIC:
        if header.startswith(magic):
            if media_type == "image/webp" and header[8:12] != b"WEBP":
                continue
            return media_type
    return None


def attach_image(prompt: str, image: Path) -> list[str | BinaryContent]:
    """Bundle a prompt and an image file into a multimodal user message.

    Args:
        prompt: The text part of the message.
        image: Path to an image file (png/jpeg/gif/webp).

    Returns:
        The content list to pass as the user prompt.

    Raises:
        ValueError: If the file is missing, too large, or not a supported
            image type (detected by content, not extension).
    """
    image = image.expanduser()
    if not image.is_file():
        raise ValueError(f"image not found: {image}")
    if image.stat().st_size > MAX_IMAGE_BYTES:
        raise ValueError(
            f"image too large: {image} ({image.stat().st_size} bytes > {MAX_IMAGE_BYTES}); "
            "note attached images persist in the session and are re-sent every turn"
        )
    with image.open("rb") as f:
        header = f.read(12)
    media_type = _sniff_media_type(header)
    if media_type is None:
        raise ValueError(f"not a supported image (png/jpeg/gif/webp): {image}")
    return [prompt, BinaryContent(data=image.read_bytes(), media_type=media_type)]


async def run_turn(
    agent: Agent[EljaDeps, str],
    settings: EljaSettings,
    session: Session,
    prompt: str | Sequence[str | BinaryContent],
    on_delta: Callable[[str], None],
    on_status: Callable[[str], None] | None = None,
    confirm: Callable[[str], bool] | None = None,
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
    deps = EljaDeps.from_settings(settings, confirm=confirm)
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
    image: Path | None = None,
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
    alive = {toolset_name(t) for t in mcp_toolsets}

    def fresh_agent() -> "Agent[EljaDeps, str]":
        # Fresh toolsets (filtered to servers that passed preflight): a server
        # that dies mid-call wedges its transport, so errors get new ones.
        return build_agent(
            settings,
            mcp_toolsets=[t for t in build_mcp_toolsets(settings) if toolset_name(t) in alive],
        )

    try:
        # Initial build reuses the preflighted toolsets (their stdio servers
        # are already warm); fresh_agent() is for post-error recovery only.
        agent = build_agent(settings, mcp_toolsets=mcp_toolsets)
    except Exception as exc:
        # A malformed skill file or bad config must not dump a traceback.
        console.print(f"cannot start agent: {str(exc) or exc!r}", style="red", markup=False)
        return

    def show_delta(delta: str) -> None:
        # Model output is data: never let rich interpret [brackets] as markup.
        console.print(delta, end="", markup=False, highlight=False, emoji=False)

    def show_status(label: str) -> None:
        console.print(f"\n⚙ {label}", style="dim", markup=False, highlight=False)

    def confirm(description: str) -> bool:
        # Runs in a worker thread; EOF declines. (A real Ctrl+C is delivered
        # to the main thread and won't interrupt this read — press Enter/EOF
        # to decline.)
        console.print(f"\napprove {description}?", style="yellow", markup=False)
        try:
            return input_fn("[y/N] ").strip().lower() in {"y", "yes"}
        except (EOFError, KeyboardInterrupt):
            return False

    async def do_turn(prompt: str | Sequence[str | BinaryContent]) -> None:
        # A failed turn must never kill the REPL: report, drop the turn
        # (history is only saved on success, atomically), and keep going.
        nonlocal agent
        try:
            await run_turn(agent, settings, session, prompt, show_delta, show_status, confirm)
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
            try:
                agent = fresh_agent()
            except Exception as rebuild_exc:
                console.print(
                    f"warning: agent rebuild failed ({str(rebuild_exc) or rebuild_exc!r}); "
                    "keeping current agent",
                    style="yellow",
                    markup=False,
                )
        console.print()

    if once is not None:
        if once.strip():
            try:
                prompt = attach_image(once, image) if image is not None else once
            except ValueError as exc:
                console.print(str(exc), style="red", markup=False)
                return
            await do_turn(prompt)
        return
    console.print(
        f"[bold]elja[/bold] — model [cyan]{settings.model.name}[/cyan] at "
        f"{effective_endpoint(settings.model)} (session: {session_name}; "
        "/img <path> <prompt> to attach "
        "an image; exit/quit to leave)"
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
        if prompt.split(maxsplit=1)[0] == "/img":
            try:
                tokens = shlex.split(prompt)
            except ValueError:
                tokens = []
            if len(tokens) < 3:
                console.print(
                    "usage: /img <path> <prompt> (quote paths containing spaces)",
                    style="yellow",
                )
                continue
            try:
                await do_turn(attach_image(" ".join(tokens[2:]), Path(tokens[1])))
            except ValueError as exc:
                console.print(str(exc), style="red", markup=False)
            continue
        await do_turn(prompt)


def main() -> None:
    """Console-script entry point."""
    parser = build_parser()
    args = parser.parse_args()
    if args.image is not None and args.once is None:
        parser.error("--image requires --once (in the REPL, use /img <path> <prompt>)")
    settings = load_settings(args.config)
    try:
        asyncio.run(repl(settings, args.session, once=args.once, image=args.image))
    except KeyboardInterrupt:
        print("\ninterrupted")


if __name__ == "__main__":  # pragma: no cover
    main()
