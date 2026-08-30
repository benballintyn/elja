"""Built-in workspace tools.

Each tool is a thin ``RunContext`` wrapper around a ``do_*`` function that
takes :class:`~elja.deps.EljaDeps` directly — the ``do_*`` layer is what unit
tests exercise. Every model-plausible failure (bad paths, binary files,
filesystem errors) is raised as :class:`ToolError`, which the wrappers convert
to ``ModelRetry`` so the model sees the message and can correct itself —
a tool mistake must never kill the agent run.
"""

import hashlib
import os
import signal
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
    try:
        candidate = (deps.workspace / path).resolve()
    except (ValueError, OSError) as exc:
        raise ToolError(f"invalid path {path!r}: {exc}") from exc
    if not candidate.is_relative_to(deps.workspace):
        raise ToolError(f"path {path!r} resolves outside the workspace")
    return candidate


def _cap_output(deps: EljaDeps, text: str, label: str) -> str:
    """Truncate oversized output, preserving the full text in the spill dir."""
    if len(text) <= deps.max_tool_output_chars:
        return text
    deps.spill_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(text.encode()).hexdigest()[:12]
    spill_file = deps.spill_dir / f"{label}-{digest}.txt"
    spill_file.write_text(text, encoding="utf-8")
    head = text[: deps.max_tool_output_chars]
    return (
        f"{head}\n... [output truncated: {len(text)} chars total; full output saved to "
        f"{spill_file}; page through it with run_shell, e.g. sed -n '100,200p']"
    )


def do_read_file(deps: EljaDeps, path: str) -> str:
    """Read a text file from the workspace."""
    target = _resolve(deps, path)
    if not target.is_file():
        raise ToolError(f"file {path!r} does not exist")
    try:
        content = target.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ToolError(f"file {path!r} is not utf-8 text (binary file?)") from exc
    except OSError as exc:
        raise ToolError(f"cannot read {path!r}: {exc}") from exc
    return _cap_output(deps, content, "read_file")


def do_write_file(deps: EljaDeps, path: str, content: str) -> str:
    """Write a text file inside the workspace, creating parent directories."""
    target = _resolve(deps, path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    except OSError as exc:
        raise ToolError(f"cannot write {path!r}: {exc}") from exc
    return f"wrote {len(content)} chars to {path}"


def do_list_dir(deps: EljaDeps, path: str = ".") -> str:
    """List a workspace directory (directories get a trailing slash)."""
    target = _resolve(deps, path)
    if not target.exists():
        raise ToolError(f"directory {path!r} does not exist")
    if not target.is_dir():
        raise ToolError(f"{path!r} is not a directory")
    entries = sorted(target.iterdir(), key=lambda p: p.name)
    if not entries:
        return f"{path} is empty"
    lines = []
    for entry in entries:
        try:
            if entry.is_dir():
                lines.append(f"{entry.name}/")
            else:
                lines.append(f"{entry.name} ({entry.stat().st_size} bytes)")
        except OSError:
            lines.append(f"{entry.name} (broken link)")
    return _cap_output(deps, "\n".join(lines), "list_dir")


def _decode(raw: bytes) -> str:
    return raw.decode("utf-8", errors="replace").strip()


def do_run_shell(deps: EljaDeps, command: str) -> str:
    """Run a shell command in the workspace and report output + exit code.

    The command runs in its own process group so a timeout kills the whole
    tree, not just the direct shell. Output is decoded permissively (binary
    output must not crash the run) and the exit code is appended after
    capping so it survives truncation.
    """
    try:
        proc = subprocess.Popen(
            command,
            shell=True,
            cwd=deps.workspace,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as exc:
        raise ToolError(f"cannot run command: {exc}") from exc
    try:
        raw_out, raw_err = proc.communicate(timeout=deps.shell_timeout_seconds)
        tail = f"exit code: {proc.returncode}"
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        raw_out, raw_err = proc.communicate()
        tail = f"error: command timed out after {deps.shell_timeout_seconds}s"
    stdout, stderr = _decode(raw_out), _decode(raw_err)
    output = (
        stdout
        if not stderr
        else f"{stdout}\nstderr:\n{stderr}"
        if stdout
        else f"stderr:\n{stderr}"
    )
    capped = _cap_output(deps, output, "run_shell")
    return f"{capped}\n{tail}".strip()


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
    except ToolError as exc:
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
    return FunctionToolset[EljaDeps](enabled, max_retries=settings.tools.max_retries)
