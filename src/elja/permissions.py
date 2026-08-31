"""The permission gate: per-tool allow/ask/deny, fail-closed.

A single capability whose ``before_tool_execute`` hook fires for every tool
call on the agent (built-ins, MCP tools, and ``delegate_*`` alike, matched by
tool name). Policies come from ``[permissions]``:

- ``allow`` — execute normally.
- ``deny`` — never execute; the model receives a normal tool result saying
  the call was denied (no retry burn, the run continues).
- ``ask`` — execute only if the run's approver (``EljaDeps.confirm``) says
  yes: the CLI wires an interactive y/N prompt; a web UI supplies an async
  callback awaiting the user's click. With no approver available the call
  is refused — fail closed.

The same gate instance is attached to sub-agents, and the approver travels in
``EljaDeps``, so a delegated child asking to run a gated tool prompts the
user exactly like the parent would.
"""

import asyncio
import inspect
import json
from dataclasses import dataclass
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering
from pydantic_ai.exceptions import SkipToolExecution
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import ToolDefinition

from elja.deps import EljaDeps
from elja.settings import EljaSettings

# One approval at a time, process-wide: parallel tool calls would otherwise
# race interleaved prompts on a single stdin.
_APPROVAL_LOCK = asyncio.Lock()

# Generous cap with a LOUD middle elision: the user approves what they see,
# so hiding a command tail is a real attack surface.
_MAX_SHOWN = 2000


def _describe(call: ToolCallPart) -> str:
    try:
        args = json.dumps(call.args_as_dict())
    except Exception:  # noqa: BLE001 - display only; never block on repr issues
        args = str(call.args)
    if len(args) > _MAX_SHOWN:
        half = _MAX_SHOWN // 2
        hidden = len(args) - _MAX_SHOWN
        args = f"{args[:half]} …[{hidden} chars hidden]… {args[-half:]}"
    return f"{call.tool_name}({args})"


@dataclass(kw_only=True)
class PermissionGate(AbstractCapability[EljaDeps]):
    """Applies ``[permissions]`` policies to every tool call."""

    settings: EljaSettings
    # Leading underscore: skill ids must start with a letter, so a skill can
    # never collide with the gate's capability id.
    id: str = "_elja_permission_gate"

    def get_ordering(self) -> CapabilityOrdering:
        """Pin the gate innermost so no later hook mutates approved args."""
        return CapabilityOrdering(position="innermost")

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
                f"not executed: {call.tool_name} requires approval, which is unavailable "
                "in this run"
            )
        if inspect.iscoroutinefunction(confirm):
            approved = await confirm(_describe(call))
        else:
            # Sync approvers may block on stdin — keep them off the loop, and
            # serialize them: parallel prompts would race on a single stdin.
            async with _APPROVAL_LOCK:
                approved = await asyncio.to_thread(confirm, _describe(call))
            if inspect.isawaitable(approved):
                # A sync callable that RETURNS an awaitable (async __call__
                # object, wrapper lambda) — await it; never truth-test it.
                approved = await approved
        if not approved:
            raise SkipToolExecution(
                f"not executed: the user declined {call.tool_name}; choose a different "
                "approach instead of retrying the same call"
            )
        return args


def build_permission_gate(settings: EljaSettings) -> PermissionGate:
    """The gate capability for these settings (attach to parent and children)."""
    return PermissionGate(settings=settings)
