"""Configuration for elja agents.

Settings are resolved from three sources, highest precedence first:

1. Constructor arguments (programmatic use).
2. ``ELJA_*`` environment variables, nested with ``__``
   (e.g. ``ELJA_MODEL__BASE_URL``).
3. An ``elja.toml`` file (path configurable via :func:`load_settings`).
"""

from pathlib import Path

from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


class ModelConfig(BaseModel):
    """Which LLM to talk to, and how."""

    name: str = "qwen/qwen3.8-27b"
    base_url: str = "http://localhost:1234/v1"
    api_key: str = "lm-studio"
    temperature: float = 0.2
    max_tokens: int = 4096
    # Most local OpenAI-compatible servers (LM Studio included) don't implement
    # strict tool schemas; flip this on for backends that do.
    supports_strict_tool_definition: bool = False


class LimitsConfig(BaseModel):
    """Caps on a single agent run, to bound runaway tool loops."""

    request_limit: int = 25
    total_tokens_limit: int | None = None


class WorkspaceConfig(BaseModel):
    """The directory tools operate in, and tool-output policies."""

    root: Path = Path(".")
    max_tool_output_chars: int = 20_000
    shell_timeout_seconds: float = 60.0


class ToolsConfig(BaseModel):
    """Per-tool enable flags for the built-in toolset."""

    read_file: bool = True
    write_file: bool = True
    list_dir: bool = True
    run_shell: bool = True


class AgentConfig(BaseModel):
    """Agent-level behavior."""

    instructions: str | None = None


class SessionConfig(BaseModel):
    """Where conversation history is persisted."""

    dir: Path = Path(".elja/sessions")


class EljaSettings(BaseSettings):
    """Top-level elja configuration."""

    model_config = SettingsConfigDict(
        env_prefix="ELJA_",
        env_nested_delimiter="__",
        toml_file="elja.toml",
        extra="ignore",
    )

    model: ModelConfig = ModelConfig()
    limits: LimitsConfig = LimitsConfig()
    workspace: WorkspaceConfig = WorkspaceConfig()
    tools: ToolsConfig = ToolsConfig()
    agent: AgentConfig = AgentConfig()
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
            in the current directory is used if present. A missing file is not
            an error — defaults and environment variables still apply.

    Returns:
        The resolved settings.
    """
    if config_file is None:
        return EljaSettings()

    class _Settings(EljaSettings):
        model_config = SettingsConfigDict(
            **{**EljaSettings.model_config, "toml_file": str(config_file)}
        )

    return _Settings()
