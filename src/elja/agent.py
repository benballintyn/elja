"""Agent factory: settings -> a ready-to-run elja agent."""

from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits

from elja.deps import EljaDeps
from elja.model import build_model
from elja.settings import EljaSettings
from elja.tools import build_toolset

# Short and directive on purpose: small local models do better with a tight
# prompt and a small toolset than with Claude-style multi-page instructions.
DEFAULT_INSTRUCTIONS = (
    "You are elja, a relentless, capable assistant working in a local workspace.\n"
    "Prefer using tools over guessing: read files before describing them, run\n"
    "commands to verify claims, and check results before declaring success.\n"
    "Keep answers concise. When a task is done, state plainly what you did."
)


def build_agent(settings: EljaSettings) -> Agent[EljaDeps, str]:
    """Assemble the elja agent from settings.

    Args:
        settings: Resolved elja settings.

    Returns:
        An agent wired with the configured model, instructions, and toolset.
        Pass ``deps=EljaDeps.from_settings(settings)`` and
        ``usage_limits=build_usage_limits(settings)`` when running it.
    """
    return Agent(
        build_model(settings),
        deps_type=EljaDeps,
        instructions=settings.agent.instructions or DEFAULT_INSTRUCTIONS,
        toolsets=[build_toolset(settings)],
    )


def build_usage_limits(settings: EljaSettings) -> UsageLimits:
    """Per-run caps from settings, bounding runaway tool loops."""
    return UsageLimits(
        request_limit=settings.limits.request_limit,
        total_tokens_limit=settings.limits.total_tokens_limit,
    )
