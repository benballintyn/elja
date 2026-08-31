"""Tests for elja.settings."""

from pathlib import Path

import pytest
from pydantic import ValidationError
from pytest_mock import MockerFixture

from elja.settings import EljaSettings, load_settings


def test_defaults_target_lm_studio() -> None:
    """With no config file or env, settings default to LM Studio + Qwen3.8-27B."""
    settings = EljaSettings()
    assert settings.model.provider == "openai"
    assert settings.model.name == "qwen/qwen3.8-27b"
    assert settings.model.base_url is None
    assert settings.model.api_key is None
    # Secrets must not leak through repr/logging.
    leaky = EljaSettings(model={"api_key": "super-secret"})  # type: ignore[arg-type]
    assert "super-secret" not in repr(leaky)
    assert settings.limits.request_limit == 25
    assert settings.limits.total_tokens_limit is None
    assert settings.workspace.root == Path(".")
    assert settings.tools.run_shell is True
    assert settings.agent.instructions is None


def test_load_settings_default_path_reads_cwd_toml() -> None:
    """load_settings() with no argument picks up ./elja.toml (cwd is tmp_path)."""
    Path("elja.toml").write_text('[model]\nname = "from-cwd-toml"\n')
    assert load_settings().model.name == "from-cwd-toml"


def test_load_settings_default_path_no_file() -> None:
    """load_settings() without ./elja.toml present just uses defaults."""
    settings = load_settings()
    assert settings.model.name == "qwen/qwen3.8-27b"


def test_load_settings_explicit_missing_file_raises(tmp_path: Path) -> None:
    """An explicitly-passed config path that doesn't exist is a user error."""
    with pytest.raises(FileNotFoundError, match="config file not found"):
        load_settings(tmp_path / "typo.toml")


def test_unknown_keys_rejected(tmp_path: Path) -> None:
    """Config typos fail loudly instead of silently doing nothing."""
    config = tmp_path / "elja.toml"
    config.write_text("[tools]\nrun_shel = false\n")
    with pytest.raises(ValidationError):
        load_settings(config)
    with pytest.raises(ValidationError):
        EljaSettings(modell={"name": "x"})  # type: ignore[call-arg]


def test_init_dict_form_merges_with_env(mocker: MockerFixture) -> None:
    """Init beats env per-key (dict form), while env still fills sibling keys."""
    mocker.patch.dict(
        "os.environ",
        {"ELJA_MODEL__NAME": "env-name", "ELJA_MODEL__TEMPERATURE": "0.9"},
    )
    settings = EljaSettings(model={"name": "init-name"})  # type: ignore[arg-type]
    assert settings.model.name == "init-name"
    assert settings.model.temperature == 0.9


def test_env_parses_paths_and_optional_ints(mocker: MockerFixture) -> None:
    """Non-string field types parse correctly from env strings."""
    mocker.patch.dict(
        "os.environ",
        {"ELJA_WORKSPACE__ROOT": "/some/where", "ELJA_LIMITS__TOTAL_TOKENS_LIMIT": "9000"},
    )
    settings = EljaSettings()
    assert settings.workspace.root == Path("/some/where")
    assert settings.limits.total_tokens_limit == 9000


def test_load_settings_reads_toml(tmp_path: Path) -> None:
    """Values in the TOML file override defaults."""
    config = tmp_path / "elja.toml"
    config.write_text(
        """
[model]
name = "some/other-model"
temperature = 0.7

[limits]
request_limit = 5

[workspace]
root = "/tmp/ws"

[tools]
run_shell = false

[agent]
instructions = "Be terse."
"""
    )
    settings = load_settings(config)
    assert settings.model.name == "some/other-model"
    assert settings.model.temperature == 0.7
    # Unset TOML keys keep their defaults.
    assert settings.model.base_url is None
    assert settings.limits.request_limit == 5
    assert settings.workspace.root == Path("/tmp/ws")
    assert settings.tools.run_shell is False
    assert settings.agent.instructions == "Be terse."


def test_env_overrides_toml(tmp_path: Path, mocker: MockerFixture) -> None:
    """ELJA_* environment variables take precedence over the TOML file."""
    config = tmp_path / "elja.toml"
    config.write_text('[model]\nname = "from-toml"\n')
    mocker.patch.dict(
        "os.environ",
        {"ELJA_MODEL__NAME": "from-env", "ELJA_LIMITS__REQUEST_LIMIT": "3"},
    )
    settings = load_settings(config)
    assert settings.model.name == "from-env"
    assert settings.limits.request_limit == 3


def test_empty_env_string_means_unset(mocker: MockerFixture) -> None:
    """ELJA_MODEL__BASE_URL='' resets to the default rather than sending ''."""
    mocker.patch.dict("os.environ", {"ELJA_MODEL__BASE_URL": "", "ELJA_MODEL__API_KEY": ""})
    settings = EljaSettings()
    assert settings.model.base_url is None
    assert settings.model.api_key is None
