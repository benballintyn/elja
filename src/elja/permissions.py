"""The permission gate: per-tool allow/ask/deny, fail-closed.

A single capability whose ``before_tool_execute`` hook fires for every tool
call on the agent (built-ins, MCP tools, and ``delegate_*`` alike, matched by
tool name). Policies come from ``[permissions]``:

- ``allow`` — execute normally.
- ``deny`` — never execute; the model receives a normal tool result saying
  the call was denied (no retry burn, the run continues).
- ``ask`` — execute only if the run's approver (``EljaDeps.confirm``, wired
  by the CLI to an interactive y/N prompt) says yes. With no approver
  available the call is refused — fail closed.

The same gate instance is attached to sub-agents, and the approver travels in
``EljaDeps``, so a delegated child asking to run a gated tool prompts the
user exactly like the parent would.
"""

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import SkipToolExecution
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import ToolDefinition

from elja.deps import EljaDeps
from elja.settings import EljaSettings


def _describe(call: ToolCallPart) -> str:
    try:
        args = json.dumps(call.args_as_dict())
    except Exception:  # noqa: BLE001 - display only; never block on repr issues
        args = str(call.args)
    if len(args) > 300:
        args = args[:300] + "…"
    return f"{call.tool_name}({args})"


@dataclass
class PermissionGate(AbstractCapability[EljaDeps]):
    """Applies ``[permissions]`` policies to every tool call."""

    settings: EljaSettings = None  # type: ignore[assignment]
    id: str = "permission-gate"

    async def before_tool_execute(
        self,
        ctx: RunContext[EljaDeps],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: Any,  # noqa: ANN401 - upstream hook signature
    ) -> Any:  # noqa: ANN401 - upstream hook signature
        """Enforce the configured policy for this tool call."""
        cfg = self.settings.permissions
        policy = cfg.tools.get(call.tool_name, cfg.default)
        if policy == "allow":
            return args
        if policy == "deny":
            raise SkipToolExecution(
                f"denied: {call.tool_name} is disabled by [permissions] config"
            )
        # policy == "ask"
        confirm = ctx.deps.confirm
        if confirm is None:
            raise SkipToolExecution(
                f"not executed: {call.tool_name} requires interactive approval and no "
                "approver is available (set [permissions] to 'allow' it, or run in the REPL)"
            )
        approved = await asyncio.to_thread(confirm, _describe(call))
        if not approved:
            raise SkipToolExecution(
                f"not executed: the user declined {call.tool_name}; choose a different "
                "approach instead of retrying the same call"
            )
        return args


def build_permission_gate(settings: EljaSettings) -> PermissionGate:
    """The gate capability for these settings (attach to parent and children)."""
    return PermissionGate(settings=settings)
