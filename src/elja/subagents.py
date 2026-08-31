"""Sub-agents: config-defined delegates exposed as tools.

Each ``[subagents.<name>]`` entry becomes a ``delegate_<name>`` tool on the
main agent. A delegation runs the child agent in an isolated context — it
sees only the task string, never the parent's history — and only its final
answer returns to the parent (the results-only pattern: the child's tool
traffic never pollutes the parent's context). Child token usage rolls up
into the parent run's usage and limits.
"""

import re
from collections.abc import Awaitable, Callable

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.tools import ToolFuncEither
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.usage import UsageLimits

from elja.deps import EljaDeps
from elja.model import build_model
from elja.settings import EljaSettings, SubagentConfig
from elja.tools import list_dir, read_file, run_shell, web_search, write_file

_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_BUILTIN_TOOLS: dict[str, "ToolFuncEither[EljaDeps, ...]"] = {
    "read_file": read_file,
    "write_file": write_file,
    "list_dir": list_dir,
    "run_shell": run_shell,
    "web_search": web_search,
}
_RESERVED_NAMES = frozenset(_BUILTIN_TOOLS) | {"load_capability"}


def known_tool_names() -> frozenset[str]:
    """Names accepted in a subagent's ``tools`` list."""
    return frozenset(_BUILTIN_TOOLS)


def _child_toolset(settings: EljaSettings, cfg: SubagentConfig) -> FunctionToolset[EljaDeps]:
    names = cfg.tools if cfg.tools is not None else list(_BUILTIN_TOOLS)
    return FunctionToolset[EljaDeps](
        [_BUILTIN_TOOLS[n] for n in names], max_retries=settings.tools.max_retries
    )


def build_subagent_toolset(settings: EljaSettings) -> FunctionToolset[EljaDeps] | None:
    """Build one ``delegate_<name>`` tool per configured subagent.

    Args:
        settings: Resolved elja settings.

    Returns:
        A toolset of delegate tools, or ``None`` when no subagents are
        configured.

    Raises:
        ValueError: If a subagent name is not a valid slug or collides with a
            built-in tool name.
    """
    if not settings.subagents:
        return None
    toolset = FunctionToolset[EljaDeps](max_retries=settings.tools.max_retries)
    for name, cfg in settings.subagents.items():
        if not _NAME_RE.match(name) or name in _RESERVED_NAMES:
            raise ValueError(
                f"invalid subagent name {name!r}: use letters/digits/_/- and avoid "
                f"built-in tool names"
            )
        toolset.add_function(
            _make_delegate(settings, name, cfg),
            name=f"delegate_{name}",
            description=(
                f"Delegate a task to the {name} subagent ({cfg.description}) "
                "and get back its final answer."
            ),
        )
    return toolset


def _make_delegate(
    settings: EljaSettings, name: str, cfg: SubagentConfig
) -> Callable[[RunContext[EljaDeps], str], Awaitable[str]]:
    async def delegate(ctx: RunContext[EljaDeps], task: str) -> str:
        child: Agent[EljaDeps, str] = Agent(
            build_model(settings),
            deps_type=EljaDeps,
            instructions=cfg.instructions,
            toolsets=[_child_toolset(settings, cfg)],
        )
        try:
            result = await child.run(
                task,
                deps=ctx.deps,
                usage=ctx.usage,
                usage_limits=UsageLimits(
                    request_limit=cfg.request_limit or settings.limits.request_limit
                ),
            )
        except Exception as exc:
            raise ModelRetry(f"subagent {name!r} failed: {str(exc) or exc!r}") from exc
        return result.output

    return delegate
