"""elja: a relentless, fully-customizable LLM agent harness built on Pydantic AI."""

from elja.agent import DEFAULT_INSTRUCTIONS, build_agent, build_usage_limits
from elja.deps import EljaDeps
from elja.mcp import build_mcp_toolsets
from elja.model import build_model
from elja.session import Session
from elja.settings import EljaSettings, load_settings
from elja.tools import build_toolset

# Version is managed by release-please (kept in sync with pyproject.toml).
__version__ = "0.3.0"

__all__ = [
    "DEFAULT_INSTRUCTIONS",
    "EljaDeps",
    "EljaSettings",
    "Session",
    "__version__",
    "build_agent",
    "build_mcp_toolsets",
    "build_model",
    "build_toolset",
    "build_usage_limits",
    "load_settings",
]
