"""Run-scoped dependencies injected into every tool via ``RunContext``."""

from dataclasses import dataclass
from pathlib import Path

from elja.settings import EljaSettings


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

    @classmethod
    def from_settings(cls, settings: EljaSettings) -> "EljaDeps":
        """Build deps from resolved settings.

        Args:
            settings: Resolved elja settings.

        Returns:
            Deps with an absolute workspace root and spill directory under it.
        """
        workspace = settings.workspace.root.resolve()
        return cls(
            workspace=workspace,
            spill_dir=workspace / ".elja" / "spill",
            max_tool_output_chars=settings.workspace.max_tool_output_chars,
            shell_timeout_seconds=settings.workspace.shell_timeout_seconds,
        )
