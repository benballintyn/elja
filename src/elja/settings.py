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

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


class _Section(BaseModel):
    """Base for config sections: unknown keys are errors, not silent no-ops."""

    model_config = ConfigDict(extra="forbid")


class ModelConfig(_Section):
    """Which LLM to talk to, and how."""

    name: str = "qwen/qwen3.8-27b"
    base_url: str = "http://localhost:1234/v1"
    api_key: SecretStr = SecretStr("lm-studio")
    temperature: float = 0.2
    max_tokens: int = 4096
    # Most local OpenAI-compatible servers (LM Studio included) don't implement
    # strict tool schemas; flip this on for backends that do.
    supports_strict_tool_definition: bool = False


class LimitsConfig(_Section):
    """Caps on a single agent run, to bound runaway tool loops."""

    request_limit: int = 25
    total_tokens_limit: int | None = None


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

    @model_validator(mode="after")
    def _check_transport_fields(self) -> "MCPServerConfig":
        if self.transport == "stdio" and not self.command:
            raise ValueError("stdio MCP server requires 'command'")
        if self.transport == "http" and not self.url:
            raise ValueError("http MCP server requires 'url'")
        return self


class MCPConfig(_Section):
    """MCP servers whose tools the agent can use, keyed by a short name."""

    servers: dict[str, MCPServerConfig] = {}


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
