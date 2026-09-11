"""Configuration for elja agents.

Settings are resolved from three sources, highest precedence first:

1. Constructor arguments (programmatic use).
2. ``ELJA_*`` environment variables, nested with ``__``
   (e.g. ``ELJA_MODEL__BASE_URL``).
3. An ``elja.toml`` file (path configurable via :func:`load_settings`).

Note: programmatic overrides merge per-key only in dict form —
``EljaSettings(model={"name": "x"})`` still lets env/TOML fill the other model
keys, while passing a ``ModelConfig`` instance replaces the whole section.
"""

import importlib
import re
from collections.abc import Iterable
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from pydantic_ai.settings import ModelSettings
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

# Settings keys each provider dialect understands, on top of the portable
# ``ModelSettings`` ones. Imported lazily because anthropic/google live behind
# extras; when the extra is missing we fall back to accepting the dialect's own
# ``<provider>_`` prefix and let build_model raise the real "install the extra"
# error.
_SETTINGS_CLASSES = {
    "openai": ("pydantic_ai.models.openai", "OpenAIChatModelSettings"),
    "anthropic": ("pydantic_ai.models.anthropic", "AnthropicModelSettings"),
    "google": ("pydantic_ai.models.google", "GoogleModelSettings"),
}


def unsupported_settings_keys(provider: str, keys: "Iterable[str]") -> list[str]:
    """Which ``model.settings`` keys this provider dialect does not understand.

    Args:
        provider: A ``ModelConfig.provider`` value.
        keys: The settings keys a config supplies.

    Returns:
        The offending keys, sorted. Empty means every key is valid. The check
        covers the portable ``ModelSettings`` keys plus the provider's own. When
        the provider's optional dependency is not installed its specific keys
        cannot be enumerated, so ``<provider>_``-prefixed keys pass here and the
        missing extra is reported by :func:`build_model` instead.
    """
    portable = frozenset(ModelSettings.__annotations__)
    module_name, class_name = _SETTINGS_CLASSES[provider]
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        prefix = f"{provider}_"
        return sorted(k for k in keys if k not in portable and not k.startswith(prefix))
    allowed = portable | frozenset(getattr(module, class_name).__annotations__)
    return sorted(set(keys) - allowed)


class _Section(BaseModel):
    """Base for config sections: unknown keys are errors, not silent no-ops."""

    model_config = ConfigDict(extra="forbid")


class ModelConfig(_Section):
    """Which LLM to talk to, and how.

    ``provider`` selects the API dialect: ``openai`` (default — any
    OpenAI-compatible endpoint, incl. LM Studio/Ollama/vLLM/OpenRouter),
    ``anthropic``, or ``google``. With ``provider = "openai"``, an unset
    ``base_url``/``api_key`` defaults to a **local LM Studio server**
    (``http://localhost:1234/v1``) — ``provider = "openai"`` alone never means
    api.openai.com, so a hosted application must name the endpoint and model it
    intends. For the native providers an unset ``api_key`` falls back to the
    SDK's standard environment variable (ANTHROPIC_API_KEY / GOOGLE_API_KEY).

    ``temperature``/``max_tokens`` are convenience fields for the two settings
    nearly every config sets; ``None`` omits the parameter entirely, for models
    that reject it. Anything else native goes in ``settings``, which is checked
    against the selected provider's own ``ModelSettings`` keys — so
    provider-specific reasoning controls (``openai_reasoning_effort``,
    ``anthropic_thinking``, ``google_thinking_config``, and the portable
    ``thinking``) pass through without elja defining a dialect of its own.
    """

    provider: Literal["openai", "anthropic", "google"] = "openai"
    name: str = "qwen/qwen3.8-27b"
    base_url: str | None = None
    api_key: SecretStr | None = None

    # ``None`` means "omit this parameter from the request" — some reasoning
    # models reject an explicit temperature. The defaults are unchanged.
    temperature: float | None = 0.2
    max_tokens: int | None = 4096
    # Native ModelSettings passed through verbatim, validated against the
    # selected provider's own keys (see _check_settings). Deliberately not a
    # free-form **kwargs bag: an unsupported key is an error, not a silent no-op.
    settings: dict[str, Any] = {}
    # Most local OpenAI-compatible servers (LM Studio included) don't implement
    # strict tool schemas; flip this on for backends that do. (openai provider only.)
    supports_strict_tool_definition: bool = False

    @field_validator("base_url", "api_key", "temperature", "max_tokens", mode="before")
    @classmethod
    def _empty_env_means_unset(cls, v: object) -> object:
        # ELJA_MODEL__BASE_URL='' is the natural env spelling of "back to
        # default"; an empty string would otherwise reach the SDKs verbatim.
        # For temperature/max_tokens it is the only way env can say "omit".
        return None if v == "" else v

    @model_validator(mode="after")
    def _check_settings(self) -> "ModelConfig":
        unknown = unsupported_settings_keys(self.provider, self.settings)
        if unknown:
            raise ValueError(
                f"unsupported model.settings key(s) for provider {self.provider!r}: {unknown}"
            )
        # Two spellings of one parameter is a config bug, not a precedence
        # puzzle — but only when the convenience field was set on purpose.
        both = ("temperature", "max_tokens")
        clash = sorted(k for k in both if k in self.settings and k in self.model_fields_set)
        if clash:
            raise ValueError(
                f"model.settings duplicates model.{clash[0]}; set it in one place only"
            )
        return self


class LimitsConfig(_Section):
    """Caps on a single agent run, to bound runaway tool loops.

    Every field maps to the same-named ``pydantic_ai.usage.UsageLimits``
    argument. ``request_limit`` keeps elja's own lower default (25 rather than
    upstream's 50); the rest default to ``None``, i.e. uncapped, so an existing
    config resolves exactly as before. ``cost_limit`` is in provider currency
    units and is only enforced for models pydantic-ai can price.
    """

    request_limit: int | None = 25
    total_tokens_limit: int | None = None
    cost_limit: Decimal | None = Field(default=None, ge=0)
    tool_calls_limit: int | None = Field(default=None, ge=1)
    input_tokens_limit: int | None = Field(default=None, ge=1)
    output_tokens_limit: int | None = Field(default=None, ge=1)
    per_request_input_tokens_limit: int | None = Field(default=None, ge=1)
    # Costs an extra count-tokens round trip per request on providers that
    # offer one; lets per_request_input_tokens_limit refuse before dispatch.
    count_tokens_before_request: bool = False


class WorkspaceConfig(_Section):
    """The directory tools operate in, and tool-output policies."""

    root: Path = Path(".")
    max_tool_output_chars: int = 20_000
    shell_timeout_seconds: float = 60.0


class ToolsConfig(_Section):
    """Per-tool enable flags and retry policy for the built-in toolset."""

    read_file: bool = True
    write_file: bool = True
    list_dir: bool = True
    run_shell: bool = True
    web_search: bool = True
    # Consecutive failures allowed per tool before the run aborts. Small local
    # models fumble paths often; request_limit still bounds the overall loop.
    max_retries: int = 3


class MCPServerConfig(_Section):
    """One MCP server to attach: a local stdio subprocess or a remote HTTP endpoint."""

    transport: Literal["stdio", "http"] = "stdio"
    # stdio: the subprocess to launch.
    command: str | None = None
    args: list[str] = []
    env: dict[str, str] = {}
    # http: the streamable-HTTP endpoint.
    url: str | None = None
    # http only: extra request headers (e.g. Authorization); values are secret.
    headers: dict[str, SecretStr] = {}
    # Expose this server's tools as <tool_prefix>_<name> to avoid collisions.
    # NB: [permissions.tools] entries must then use the PREFIXED name.
    tool_prefix: str | None = None
    # Seconds allowed for server startup/handshake (SDK default is 5 — too
    # short for npx/uvx-style servers with cold caches).
    init_timeout: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _check_transport_fields(self) -> "MCPServerConfig":
        if self.transport == "stdio" and not self.command:
            raise ValueError("stdio MCP server requires 'command'")
        if self.transport == "http" and not self.url:
            raise ValueError("http MCP server requires 'url'")
        if self.transport == "stdio" and self.headers:
            raise ValueError("'headers' only applies to http MCP servers")
        if self.tool_prefix is not None and not re.match(
            r"^[A-Za-z][A-Za-z0-9_]*$", self.tool_prefix
        ):
            raise ValueError("'tool_prefix' must be letters/digits/underscores")
        return self


class MCPConfig(_Section):
    """MCP servers whose tools the agent can use, keyed by a short name."""

    servers: dict[str, MCPServerConfig] = {}

    @model_validator(mode="after")
    def _check_unique_prefixes(self) -> "MCPConfig":
        prefixes = [s.tool_prefix for s in self.servers.values() if s.tool_prefix]
        dupes = {p for p in prefixes if prefixes.count(p) > 1}
        if dupes:
            raise ValueError(f"duplicate tool_prefix across MCP servers: {sorted(dupes)}")
        return self


class SubagentConfig(_Section):
    """A delegate agent the main agent can hand tasks to."""

    description: str
    instructions: str
    # Built-in tool names the subagent may use; None = all enabled built-ins.
    tools: list[str] | None = None
    request_limit: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _check_tools(self) -> "SubagentConfig":
        if self.tools is not None:
            from elja.subagents import known_tool_names

            unknown = set(self.tools) - known_tool_names()
            if unknown:
                raise ValueError(f"unknown tool(s) for subagent: {sorted(unknown)}")
        return self


class AgentConfig(_Section):
    """Agent-level behavior."""

    instructions: str | None = None


class PermissionsConfig(_Section):
    """Per-tool execution policy: allow, ask (interactive approval), or deny.

    ``tools`` matches by tool NAME — built-ins, MCP tools, ``delegate_*``.
    Unprefixed MCP tools match their raw server-side name (two servers
    exposing the same name share one policy — set ``tool_prefix`` for
    per-server policies, and entries must then use the PREFIXED name, e.g.
    ``helper_echo``). NB the shipped default gates only the built-in
    ``run_shell``; MCP-provided execution tools follow ``default`` unless
    named here. ``ask`` fails closed when no interactive approver is
    available.
    """

    default: Literal["allow", "ask", "deny"] = "allow"
    tools: dict[str, Literal["allow", "ask", "deny"]] = {"run_shell": "ask"}


class CompactionConfig(_Section):
    """Context compaction policy (see elja.compaction for the strategy rationale)."""

    enabled: bool = True
    # Conservative default for a local 27B: quality degrades and prefill slows
    # well before the model's nominal window (Qwen3.8 advertises 262K).
    target_tokens: int = Field(default=24_000, ge=1000)
    # Recent tool call/result pairs kept verbatim by the masking tier.
    keep_tool_pairs: int = Field(default=10, ge=1)
    # Recent messages kept verbatim if the summarization fallback fires.
    keep_messages: int = Field(default=20, ge=1)


class SkillsConfig(_Section):
    """Where markdown skill files live (relative paths anchor at the workspace root)."""

    dir: Path = Path("skills")


class SessionConfig(_Section):
    """Where conversation history is persisted."""

    dir: Path = Path(".elja/sessions")


class EljaSettings(BaseSettings):
    """Top-level elja configuration."""

    model_config = SettingsConfigDict(
        env_prefix="ELJA_",
        env_nested_delimiter="__",
        toml_file="elja.toml",
        extra="forbid",
    )

    model: ModelConfig = ModelConfig()
    limits: LimitsConfig = LimitsConfig()
    workspace: WorkspaceConfig = WorkspaceConfig()
    tools: ToolsConfig = ToolsConfig()
    permissions: PermissionsConfig = PermissionsConfig()
    mcp: MCPConfig = MCPConfig()
    subagents: dict[str, SubagentConfig] = {}
    agent: AgentConfig = AgentConfig()
    compaction: CompactionConfig = CompactionConfig()
    skills: SkillsConfig = SkillsConfig()
    session: SessionConfig = SessionConfig()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Resolve init > env > TOML file."""
        return (init_settings, env_settings, TomlConfigSettingsSource(settings_cls))


def load_settings(config_file: Path | None = None) -> EljaSettings:
    """Load settings, optionally from an explicit TOML file.

    Args:
        config_file: Path to a TOML config file. When ``None``, ``elja.toml``
            in the current directory is used if present (a missing default
            file is fine — defaults and environment variables still apply).
            An explicitly given path that doesn't exist is an error.

    Returns:
        The resolved settings.

    Raises:
        FileNotFoundError: If ``config_file`` is given but doesn't exist.
    """
    if config_file is None:
        return EljaSettings()
    if not config_file.is_file():
        raise FileNotFoundError(f"config file not found: {config_file}")

    class _Settings(EljaSettings):
        model_config = SettingsConfigDict(
            **{**EljaSettings.model_config, "toml_file": str(config_file)}
        )

    return _Settings()
