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
from collections.abc import Iterable, Mapping
from decimal import Decimal
from functools import cache
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    create_model,
    field_validator,
    model_validator,
)
from pydantic_ai.settings import ModelSettings
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

# Settings each provider dialect understands, on top of the portable
# ``ModelSettings`` ones. Imported lazily because anthropic/google live behind
# extras — and only when a config actually sets ``[model.settings]``, so the
# common case never drags a provider SDK into an ``import elja``.
_SETTINGS_CLASSES = {
    "openai": ("pydantic_ai.models.openai", "OpenAIChatModelSettings"),
    "anthropic": ("pydantic_ai.models.anthropic", "AnthropicModelSettings"),
    "google": ("pydantic_ai.models.google", "GoogleModelSettings"),
}


@cache
def _value_validator(dialect: type) -> type[BaseModel]:
    """A model that validates one dialect's setting VALUES, not just its keys.

    ``TypeAdapter`` cannot be configured for a ``TypedDict`` directly, and these
    dialects annotate ``timeout`` as ``float | httpx.Timeout``, which needs
    ``arbitrary_types_allowed``. A generated wrapper model supplies both.
    """
    return create_model(
        f"_{dialect.__name__}Values",
        __config__=ConfigDict(arbitrary_types_allowed=True),
        settings=(dialect, ...),
    )


def validate_provider_settings(provider: str, settings: Mapping[str, Any]) -> dict[str, Any]:
    """Check ``model.settings`` against one provider dialect, keys and values.

    Args:
        provider: A ``ModelConfig.provider`` value.
        settings: The settings a config supplies.

    Returns:
        The settings with values coerced to their annotated types — which is
        also how ``ELJA_MODEL__SETTINGS__TOP_P=0.9`` stops being the string
        ``"0.9"`` by the time it reaches a provider.

    Raises:
        ValueError: If any key is not part of this dialect, at any depth. A
            value that does not match its annotation raises
            ``pydantic.ValidationError``, which is also what a bad
            ``model.temperature`` raises.
    """
    portable = frozenset(ModelSettings.__annotations__)
    module_name, class_name = _SETTINGS_CLASSES[provider]
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        # Without the extra the dialect's own keys cannot be enumerated, let
        # alone type-checked; accept its prefix and let build_model report the
        # missing extra, which is the error the user actually needs.
        prefix = f"{provider}_"
        _reject(provider, [k for k in settings if k not in portable and not k.startswith(prefix)])
        return dict(settings)
    dialect = getattr(module, class_name)
    validated = _materialize(_value_validator(dialect)(settings=settings).settings)  # type: ignore[attr-defined]
    # Pydantic DROPS keys a TypedDict does not declare rather than complaining,
    # at every depth. One walk of what came back finds all of them, so a typo
    # inside google_thinking_config is refused exactly like one beside it —
    # rather than silently disabling the setting it was meant to configure.
    _reject(provider, _dropped_keys(settings, validated))
    return dict(validated)


def _materialize(value: Any) -> Any:  # noqa: ANN401 - walks arbitrary settings values
    """Force pydantic's lazy ``Iterable[...]`` validators into real containers.

    Three leaves in the shipped dialects are annotated ``Iterable[...]``
    (``anthropic_context_management.edits``, ``anthropic_container.skills``,
    ``openai_prediction.content``), and pydantic validates those lazily into a
    ``ValidatorIterator`` that can be consumed exactly once. The settings dict
    is reused for every request of every run, so without this the first request
    carries the value and every later one carries an empty list — a provider
    feature that silently switches itself off after one turn.
    """
    if isinstance(value, Mapping):
        return {key: _materialize(item) for key, item in value.items()}
    if isinstance(value, str | bytes) or not isinstance(value, Iterable):
        return value
    return [_materialize(item) for item in value]


def _dropped_keys(original: Any, validated: Any, prefix: str = "") -> list[str]:  # noqa: ANN401
    """Dotted paths of mapping keys that validation discarded.

    Walks the two trees together. Sequences are compared positionally and only
    when they are the same length, so a coercion that changes a list's shape is
    left to pydantic's own error rather than reported as a missing key.
    """
    dropped: list[str] = []
    if isinstance(original, Mapping):
        if not isinstance(validated, Mapping):
            return dropped
        for key, value in original.items():
            path = f"{prefix}{key}"
            if key not in validated:
                dropped.append(path)
            else:
                dropped.extend(_dropped_keys(value, validated[key], f"{path}."))
    elif (
        isinstance(original, list)
        and isinstance(validated, list)
        and len(original) == len(validated)
    ):
        for index, (left, right) in enumerate(zip(original, validated, strict=True)):
            dropped.extend(_dropped_keys(left, right, f"{prefix}{index}."))
    return dropped


def _reject(provider: str, unknown: list[str]) -> None:
    """Raise for settings keys this provider dialect does not have."""
    if unknown:
        raise ValueError(
            f"unsupported model.settings key(s) for provider {provider!r}: {sorted(unknown)}"
        )


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
        # The overwhelmingly common case, and the one that runs when this class
        # is constructed for EljaSettings' own default: nothing to check, and no
        # provider SDK imported.
        if not self.settings:
            return self
        # Two spellings of one parameter is a config bug, not a precedence
        # puzzle — but only when the convenience field was set on purpose.
        # "Set on purpose" is a property of the VALUE, not of how the object was
        # built: model_dump() emits every field, so keying this on
        # model_fields_set would make a dump/reload round trip raise on a config
        # that validated. Comparing against the default also subsumes
        # model_fields_set — an unset field always holds its default — so there
        # is one condition here rather than two that mask each other. The cost
        # is that writing the default value explicitly beside a settings entry is
        # no longer flagged.
        both = ("temperature", "max_tokens")
        fields = type(self).model_fields
        clash = sorted(
            k for k in both if k in self.settings and getattr(self, k) != fields[k].default
        )
        if clash:
            named = ", ".join(f"model.{k}" for k in clash)
            raise ValueError(f"model.settings duplicates {named}; set each in one place only")
        self.settings = validate_provider_settings(self.provider, self.settings)
        return self


class LimitsConfig(_Section):
    """Caps on a single agent run, to bound runaway tool loops.

    Every field maps to the same-named ``pydantic_ai.usage.UsageLimits``
    argument. ``request_limit`` keeps elja's own lower default (25 rather than
    upstream's 50); the rest default to ``None``, i.e. uncapped, so an existing
    config resolves exactly as before.

    Two fields come with conditions worth knowing before relying on them:

    - ``cost_limit`` is in **USD** and is only enforced for models pydantic-ai
      can price. On an unpriced model (the local default among them) the run's
      cost is ``None``, the limit does nothing, and pydantic-ai emits a
      ``CostNotFoundWarning`` on every request.
    - ``count_tokens_before_request`` needs a provider that offers a
      count-tokens call. Only ``anthropic`` and ``google`` do, so
      :class:`EljaSettings` refuses it together with ``provider = "openai"``
      rather than letting every request fail with ``NotImplementedError``.
      ``per_request_input_tokens_limit`` works on every provider without it,
      checked against the usage a response reports.
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

    @model_validator(mode="after")
    def _check_cross_section(self) -> "EljaSettings":
        if self.limits.count_tokens_before_request:
            # Deferred import: elja.model imports this module. It also means the
            # provider SDK is only touched by a config that sets this flag.
            from elja.model import provider_implements_count_tokens

            # Asked of the CLASS elja would build, not of a hardcoded provider
            # list, so this stays true if upstream adds the method later.
            if not provider_implements_count_tokens(self.model.provider):
                raise ValueError(
                    "limits.count_tokens_before_request needs a model that implements "
                    f"count_tokens. model.provider {self.model.provider!r} builds "
                    "OpenAIChatModel, which does not, so pydantic-ai would raise "
                    "NotImplementedError before the first request. Use 'anthropic' or "
                    "'google', or drop the flag — per_request_input_tokens_limit is "
                    "enforced without it. A host injecting its own model should check "
                    "elja.model.implements_count_tokens(model) instead: this setting "
                    "describes the model elja would build, not the one you passed."
                )
        return self

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
