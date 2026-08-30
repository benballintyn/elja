"""Built-in workspace tools.

Each tool is a thin ``RunContext`` wrapper around a ``do_*`` function that
takes :class:`~elja.deps.EljaDeps` directly — the ``do_*`` layer is what unit
tests exercise. Failures a model can act on are raised as :class:`ToolError`,
which the wrappers convert to ``ModelRetry`` so the model sees the message and
can correct itself.
"""

import hashlib
import subprocess
from pathlib import Path

from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.toolsets import FunctionToolset

from elja.deps import EljaDeps
from elja.settings import EljaSettings


class ToolError(Exception):
    """A tool failure whose message should be shown to the model."""


def _resolve(deps: EljaDeps, path: str) -> Path:
    """Resolve ``path`` inside the workspace, rejecting escapes."""
    candidate = (deps.workspace / path).resolve()
    if not candidate.is_relative_to(deps.workspace):
        raise ToolError(f"path {path!r} is outside the workspace")
    return candidate


def _cap_output(deps: EljaDeps, text: str, label: str) -> str:
    """Truncate oversized output, preserving the full text in the spill dir."""
    if len(text) <= deps.max_tool_output_chars:
        return text
    deps.spill_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(text.encode()).hexdigest()[:12]
    spill_file = deps.spill_dir / f"{label}-{digest}.txt"
    spill_file.write_text(text)
    head = text[: deps.max_tool_output_chars]
    return (
        f"{head}\n... [output truncated: {len(text)} chars total; "
        f"full output saved to {spill_file}]"
    )


def do_read_file(deps: EljaDeps, path: str) -> str:
    """Read a text file from the workspace."""
    target = _resolve(deps, path)
    if not target.is_file():
        raise ToolError(f"file {path!r} does not exist")
    return _cap_output(deps, target.read_text(), "read_file")


def do_write_file(deps: EljaDeps, path: str, content: str) -> str:
    """Write a text file inside the workspace, creating parent directories."""
    target = _resolve(deps, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return f"wrote {len(content)} chars to {path}"


def do_list_dir(deps: EljaDeps, path: str = ".") -> str:
    """List a workspace directory (directories get a trailing slash)."""
    target = _resolve(deps, path)
    if not target.is_dir():
        raise ToolError(f"directory {path!r} does not exist")
    entries = sorted(target.iterdir(), key=lambda p: p.name)
    if not entries:
        return f"{path} is empty"
    lines = [
        f"{e.name}/" if e.is_dir() else f"{e.name} ({e.stat().st_size} bytes)" for e in entries
    ]
    return _cap_output(deps, "\n".join(lines), "list_dir")


def do_run_shell(deps: EljaDeps, command: str) -> str:
    """Run a shell command in the workspace and report output + exit code."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=deps.workspace,
            capture_output=True,
            text=True,
            timeout=deps.shell_timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return f"error: command timed out after {deps.shell_timeout_seconds}s"
    output = "\n".join(part for part in (result.stdout.strip(), result.stderr.strip()) if part)
    return _cap_output(deps, f"{output}\nexit code: {result.returncode}".strip(), "run_shell")


def read_file(ctx: RunContext[EljaDeps], path: str) -> str:
    """Read a text file. Paths are relative to the workspace root.

    Args:
        path: File path relative to the workspace.
    """
    try:
        return do_read_file(ctx.deps, path)
    except ToolError as exc:
        raise ModelRetry(str(exc)) from exc


def write_file(ctx: RunContext[EljaDeps], path: str, content: str) -> str:
    """Create or overwrite a text file. Paths are relative to the workspace root.

    Args:
        path: File path relative to the workspace.
        content: Full text content to write.
    """
    try:
        return do_write_file(ctx.deps, path, content)
    except ToolError as exc:
        raise ModelRetry(str(exc)) from exc


def list_dir(ctx: RunContext[EljaDeps], path: str = ".") -> str:
    """List files and directories at a workspace path.

    Args:
        path: Directory path relative to the workspace; defaults to its root.
    """
    try:
        return do_list_dir(ctx.deps, path)
    except ToolError as exc:
        raise ModelRetry(str(exc)) from exc


def run_shell(ctx: RunContext[EljaDeps], command: str) -> str:
    """Run a shell command from the workspace root and get its output.

    Args:
        command: The shell command to execute.
    """
    try:
        return do_run_shell(ctx.deps, command)
    except ToolError as exc:  # pragma: no cover - run_shell reports, not raises
        raise ModelRetry(str(exc)) from exc


def build_toolset(settings: EljaSettings) -> FunctionToolset[EljaDeps]:
    """Assemble the built-in toolset according to the settings' tool toggles.

    Args:
        settings: Resolved elja settings.

    Returns:
        A toolset containing only the enabled built-in tools.
    """
    enabled = [
        tool
        for tool, on in (
            (read_file, settings.tools.read_file),
            (write_file, settings.tools.write_file),
            (list_dir, settings.tools.list_dir),
            (run_shell, settings.tools.run_shell),
        )
        if on
    ]
    return FunctionToolset[EljaDeps](enabled)
