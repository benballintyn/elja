"""Run-scoped dependencies injected into every tool via ``RunContext``."""

import contextlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from elja.settings import EljaSettings

# An approver takes the human-readable call description and answers yes/no.
ConfirmCallback = Callable[[str], bool] | Callable[[str], Awaitable[bool]]


def notify(sink: "Callable[[str], None] | None", label: str) -> None:
    """Send a status label, treating the sink as display-only telemetry.

    A status sink is a UI affordance. A broken one must never abort a run — and
    must never take the turn's history with it, which is what an exception here
    did before, since it escaped before the session was saved. Accounting and
    admission belong in a model wrapper, which cannot be skipped; see
    ``docs/EMBEDDING.md``.

    Args:
        sink: The caller's status callback, or ``None``.
        label: A short human-readable label.
    """
    if sink is None:
        return
    with contextlib.suppress(Exception):
        sink(label)


@dataclass
class EljaDeps:
    """Everything a built-in tool needs that the model must not control.

    Attributes:
        workspace: Absolute root directory tools are confined to.
        spill_dir: Where oversized tool outputs are written in full.
        max_tool_output_chars: Truncation threshold for tool output.
        shell_timeout_seconds: Wall-clock cap for ``run_shell`` commands.
    """

    workspace: Path
    spill_dir: Path
    max_tool_output_chars: int
    shell_timeout_seconds: float
    # Interactive approver for [permissions] 'ask' policies; None = fail closed.
    # Sync approvers (e.g. terminal input) run in a worker thread; async
    # approvers (e.g. a web UI round-trip) are awaited on the event loop and
    # must contain their own transport errors (a raise kills the run) and
    # apply their own timeout (an unresolved approval hangs its run).
    confirm: "ConfirmCallback | None" = None
    # Status sink for sub-agent activity (e.g. "researcher → run_shell").
    on_status: Callable[[str], None] | None = None

    @classmethod
    def from_settings(
        cls,
        settings: EljaSettings,
        confirm: "ConfirmCallback | None" = None,
        on_status: Callable[[str], None] | None = None,
    ) -> "EljaDeps":
        """Build deps from resolved settings.

        Args:
            settings: Resolved elja settings.
            confirm: Interactive approver for 'ask' permission policies.
            on_status: Status sink for sub-agent activity.

        Returns:
            Deps with an absolute workspace root and spill directory under it.
        """
        workspace = settings.workspace.root.resolve()
        return cls(
            workspace=workspace,
            spill_dir=workspace / ".elja" / "spill",
            max_tool_output_chars=settings.workspace.max_tool_output_chars,
            shell_timeout_seconds=settings.workspace.shell_timeout_seconds,
            confirm=confirm,
            on_status=on_status,
        )
