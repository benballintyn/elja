"""Sub-agents: config-defined delegates exposed as tools.

Each ``[subagents.<name>]`` entry becomes a ``delegate_<name>`` tool on the
main agent. A delegation runs the child agent in an isolated context — it
sees only the task string, never the parent's history — and only its final
answer returns to the parent (the results-only pattern: the child's tool
traffic never pollutes the parent's context). Child token usage rolls up
into the parent run's usage and limits, and children get the same compaction
policy as the parent.
"""

import re
from collections.abc import Awaitable, Callable

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.tools import ToolFuncEither
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.usage import UsageLimits

from elja.compaction import build_compaction
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
# Reserved purely to avoid confusing near-collisions in model-facing names.
_RESERVED_NAMES = frozenset(_BUILTIN_TOOLS) | {"load_capability"}


def known_tool_names() -> frozenset[str]:
    """Names accepted in a subagent's ``tools`` list."""
    return frozenset(_BUILTIN_TOOLS)


def _enabled_tool_names(settings: EljaSettings) -> list[str]:
    return [n for n in _BUILTIN_TOOLS if getattr(settings.tools, n)]


def _child_toolset(
    settings: EljaSettings, name: str, cfg: SubagentConfig
) -> FunctionToolset[EljaDeps]:
    enabled = _enabled_tool_names(settings)
    if cfg.tools is None:
        names = enabled
    else:
        disabled = [t for t in cfg.tools if t not in enabled]
        if disabled:
            raise ValueError(
                f"subagent {name!r} requests tool(s) disabled in [tools]: {sorted(disabled)}"
            )
        names = cfg.tools
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
        ValueError: If a subagent name is not a valid slug or shadows a
            built-in tool/capability name, or if it requests a tool that is
            disabled in ``[tools]``.
    """
    if not settings.subagents:
        return None
    toolset = FunctionToolset[EljaDeps](max_retries=settings.tools.max_retries)
    for name, cfg in settings.subagents.items():
        if not _NAME_RE.match(name) or name in _RESERVED_NAMES:
            raise ValueError(
                f"invalid subagent name {name!r}: use letters/digits/_/-, and don't "
                f"reuse a built-in tool name (confusing for the model)"
            )
        toolset.add_function(
            _make_delegate(settings, name, cfg),
            name=f"delegate_{name}",
            description=(
                f"Delegate a task to the {name} subagent ({cfg.description}) "
                "and get back its final answer. The task must be self-contained: "
                "the subagent sees nothing else from this conversation."
            ),
        )
    return toolset


def _make_delegate(
    settings: EljaSettings, name: str, cfg: SubagentConfig
) -> Callable[[RunContext[EljaDeps], str], Awaitable[str]]:
    # Built once per subagent and reused across delegations — a fresh Agent
    # per call would leak one provider HTTP client per delegation.
    child: Agent[EljaDeps, str] = Agent(
        build_model(settings),
        deps_type=EljaDeps,
        instructions=cfg.instructions,
        toolsets=[_child_toolset(settings, name, cfg)],
        capabilities=build_compaction(settings),
    )

    async def delegate(ctx: RunContext[EljaDeps], task: str) -> str:
        """Run the subagent on a self-contained task and return its answer."""
        # cfg.request_limit is a per-delegation budget; usage is shared with
        # the parent, so offset by what's already spent. Without a per-agent
        # limit the parent's overall request limit still applies.
        if cfg.request_limit is not None:
            limit = ctx.usage.requests + cfg.request_limit
        else:
            limit = settings.limits.request_limit
        try:
            result = await child.run(
                task,
                deps=ctx.deps,
                usage=ctx.usage,
                usage_limits=UsageLimits(
                    request_limit=limit,
                    total_tokens_limit=settings.limits.total_tokens_limit,
                ),
            )
        except UsageLimitExceeded as exc:
            # Terminal, not a retry: re-delegating cannot succeed once the
            # budget is spent, and a ModelRetry loop would kill the run.
            return (
                f"subagent {name!r} stopped: budget exhausted ({exc}). "
                "Do not delegate this again; continue with what you have."
            )
        except Exception as exc:
            raise ModelRetry(f"subagent {name!r} failed: {str(exc) or exc!r}") from exc
        return result.output

    return delegate
