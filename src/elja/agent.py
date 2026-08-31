"""Agent factory: settings -> a ready-to-run elja agent."""

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.usage import UsageLimits

from elja.compaction import build_compaction
from elja.deps import EljaDeps
from elja.mcp import build_mcp_toolsets
from elja.model import build_model
from elja.settings import EljaSettings
from elja.skills import load_skills
from elja.subagents import build_subagent_toolset
from elja.tools import build_toolset

# Short and directive on purpose: small local models do better with a tight
# prompt and a small toolset than with Claude-style multi-page instructions.
DEFAULT_INSTRUCTIONS = (
    "You are elja, a relentless, capable assistant working in a local workspace.\n"
    "Prefer your available tools over guessing: read files before describing\n"
    "them, and check results before declaring success.\n"
    "Keep answers concise. When a task is done, state plainly what you did."
)


def build_agent(
    settings: EljaSettings, mcp_toolsets: list[MCPToolset] | None = None
) -> Agent[EljaDeps, str]:
    """Assemble the elja agent from settings.

    Args:
        settings: Resolved elja settings.
        mcp_toolsets: Pre-built (e.g. preflighted) MCP toolsets to use instead
            of building fresh ones from settings.

    Returns:
        An agent wired with the configured model, instructions, and toolset.
        Pass ``deps=EljaDeps.from_settings(settings)`` and
        ``usage_limits=build_usage_limits(settings)`` when running it.
    """
    instructions = settings.agent.instructions
    return Agent(
        build_model(settings),
        deps_type=EljaDeps,
        # An explicit empty string means "no system prompt"; only None gets the default.
        instructions=DEFAULT_INSTRUCTIONS if instructions is None else instructions,
        toolsets=[
            build_toolset(settings),
            *([st] if (st := build_subagent_toolset(settings)) is not None else []),
            *(build_mcp_toolsets(settings) if mcp_toolsets is None else mcp_toolsets),
        ],
        capabilities=[*load_skills(settings), *build_compaction(settings)],
    )


def build_usage_limits(settings: EljaSettings) -> UsageLimits:
    """Per-run caps from settings, bounding runaway tool loops."""
    return UsageLimits(
        request_limit=settings.limits.request_limit,
        total_tokens_limit=settings.limits.total_tokens_limit,
    )
