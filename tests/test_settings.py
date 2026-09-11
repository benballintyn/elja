"""Tests for elja.settings."""

from decimal import Decimal
from pathlib import Path
from typing import Literal

import pytest
from pydantic import SecretStr, ValidationError
from pytest_mock import MockerFixture

from elja.settings import (
    EljaSettings,
    LimitsConfig,
    ModelConfig,
    load_settings,
    validate_provider_settings,
)


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


class TestModelSettingsValidation:
    """model.settings is checked against the selected dialect, before any request."""

    def test_unknown_key_is_rejected_by_name(self) -> None:
        with pytest.raises(ValidationError, match=r"unsupported model\.settings key"):
            ModelConfig(settings={"temparature": 0.5})

    def test_the_message_names_the_offending_key_and_provider(self) -> None:
        with pytest.raises(ValidationError) as exc:
            ModelConfig(provider="anthropic", settings={"nope": 1, "also_nope": 2})
        message = str(exc.value)
        assert "'also_nope', 'nope'" in message
        assert "'anthropic'" in message

    def test_another_providers_key_is_rejected(self) -> None:
        """A google key under anthropic is a config bug, not a passthrough."""
        with pytest.raises(ValidationError, match=r"google_thinking_config"):
            ModelConfig(provider="anthropic", settings={"google_thinking_config": {}})

    def test_the_selected_providers_own_key_is_accepted(self) -> None:
        cfg = ModelConfig(
            provider="google", settings={"google_thinking_config": {"thinking_budget": 1024}}
        )
        assert cfg.settings == {"google_thinking_config": {"thinking_budget": 1024}}

    def test_an_unknown_key_nested_inside_a_value_is_dropped_not_refused(self) -> None:
        """A documented limit: pydantic silently drops extras inside a TypedDict.

        Top-level keys are rejected by name because elja checks them itself;
        one level down there is no hook to forbid them, so a typo vanishes
        rather than raising. Pinned so the behavior is known, not discovered.
        """
        cfg = ModelConfig(
            provider="google",
            settings={"google_thinking_config": {"thinking_budget": 8, "typo_here": 1}},
        )
        assert cfg.settings == {"google_thinking_config": {"thinking_budget": 8}}

    def test_portable_keys_are_accepted_for_every_provider(self) -> None:
        for provider in ("openai", "anthropic", "google"):
            cfg = ModelConfig(provider=provider, settings={"thinking": "high"})
            assert cfg.settings["thinking"] == "high"

    def test_a_parameter_set_in_both_places_is_refused(self) -> None:
        with pytest.raises(ValidationError, match=r"duplicates model\.temperature"):
            ModelConfig(temperature=0.5, settings={"temperature": 0.7})

    def test_max_tokens_set_in_both_places_is_refused(self) -> None:
        with pytest.raises(ValidationError, match=r"duplicates model\.max_tokens"):
            ModelConfig(max_tokens=100, settings={"max_tokens": 200})

    def test_an_untouched_default_is_not_a_duplicate(self) -> None:
        cfg = ModelConfig(settings={"temperature": 0.7})
        assert cfg.settings["temperature"] == 0.7
        assert cfg.temperature == 0.2  # still the default; _model_settings drops it


class TestProviderSettingsValidation:
    def test_missing_extra_defers_to_build_model_for_its_own_keys(
        self, mocker: MockerFixture
    ) -> None:
        """Without the extra the dialect's keys can't be listed, so they pass here."""
        mocker.patch(
            "elja.settings.importlib.import_module", side_effect=ImportError("no anthropic")
        )
        assert validate_provider_settings("anthropic", {"anthropic_thinking": {}}) == {
            "anthropic_thinking": {}
        }
        assert validate_provider_settings("anthropic", {"temperature": 0.4}) == {
            "temperature": 0.4
        }
        with pytest.raises(ValueError, match=r"\['bogus', 'google_x'\]"):
            validate_provider_settings("anthropic", {"bogus": 1, "google_x": 2})

    def test_installed_extra_enumerates_real_keys(self) -> None:
        assert "anthropic_thinking" in validate_provider_settings(
            "anthropic", {"anthropic_thinking": {"type": "enabled", "budget_tokens": 2048}}
        )
        with pytest.raises(ValueError, match=r"anthropic_not_a_real_key"):
            validate_provider_settings("anthropic", {"anthropic_not_a_real_key": 1})


class TestSettingsValuesAreValidatedToo:
    """Key names are not enough: a bad value must not reach the provider."""

    def test_a_wrongly_typed_value_is_refused(self) -> None:
        """Exactly as `[model] temperature = "hot"` already was."""
        with pytest.raises(ValidationError, match="temperature"):
            ModelConfig(settings={"temperature": "hot"})

    def test_an_out_of_range_literal_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="thinking"):
            ModelConfig(settings={"thinking": "enormous"})

    def test_an_environment_string_is_coerced_to_its_annotated_type(
        self, mocker: MockerFixture
    ) -> None:
        """Env vars arrive as strings; the provider must not get "0.9"."""
        mocker.patch.dict("os.environ", {"ELJA_MODEL__SETTINGS__TOP_P": "0.9"})
        settings = EljaSettings()
        assert settings.model.settings == {"top_p": 0.9}
        assert isinstance(settings.model.settings["top_p"], float)

    def test_a_valid_nested_value_survives(self) -> None:
        cfg = ModelConfig(
            provider="anthropic",
            settings={"anthropic_thinking": {"type": "enabled", "budget_tokens": 2048}},
        )
        assert cfg.settings["anthropic_thinking"]["budget_tokens"] == 2048


class TestCountTokensBeforeRequestIsRefusedWhereUnsupported:
    """OpenAIChatModel has no count-tokens API; the base method raises."""

    def test_the_default_provider_refuses_the_flag(self) -> None:
        with pytest.raises(ValidationError, match=r"count_tokens_before_request"):
            EljaSettings(limits=LimitsConfig(count_tokens_before_request=True))

    @pytest.mark.parametrize("provider", ["anthropic", "google"])
    def test_providers_that_implement_it_are_allowed(
        self, provider: Literal["anthropic", "google"]
    ) -> None:
        settings = EljaSettings(
            model=ModelConfig(provider=provider, name="m", api_key=SecretStr("k")),
            limits=LimitsConfig(count_tokens_before_request=True),
        )
        assert settings.limits.count_tokens_before_request is True

    def test_the_flag_off_is_fine_everywhere(self) -> None:
        assert EljaSettings().limits.count_tokens_before_request is False

    def test_only_anthropic_and_google_implement_count_tokens(self) -> None:
        """The premise behind the refusal, pinned against the installed SDK."""
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.models.google import GoogleModel
        from pydantic_ai.models.openai import OpenAIChatModel

        assert "count_tokens" not in OpenAIChatModel.__dict__
        assert "count_tokens" in AnthropicModel.__dict__
        assert "count_tokens" in GoogleModel.__dict__


_COUNT_CEILINGS = [
    "tool_calls_limit",
    "input_tokens_limit",
    "output_tokens_limit",
    "per_request_input_tokens_limit",
]


class TestLimitsValidation:
    @pytest.mark.parametrize("field", _COUNT_CEILINGS)
    @pytest.mark.parametrize("bad", [0, -5])
    def test_a_non_positive_count_ceiling_is_refused(self, field: str, bad: int) -> None:
        """A ceiling of zero or less would refuse the first request, not cap it."""
        with pytest.raises(ValidationError):
            LimitsConfig.model_validate({field: bad})

    @pytest.mark.parametrize("field", _COUNT_CEILINGS)
    def test_one_is_accepted_for_every_count_ceiling(self, field: str) -> None:
        assert getattr(LimitsConfig.model_validate({field: 1}), field) == 1

    def test_a_negative_cost_ceiling_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            LimitsConfig(cost_limit=Decimal("-1"))

    def test_a_zero_cost_ceiling_is_allowed(self) -> None:
        """Zero is a meaningful ceiling: refuse every priced request."""
        assert LimitsConfig(cost_limit=Decimal("0")).cost_limit == Decimal("0")


class TestOmittingSamplingParameters:
    def test_toml_empty_string_omits_temperature(self, tmp_path: Path) -> None:
        """TOML can say it too: the validator runs before type coercion."""
        config = tmp_path / "elja.toml"
        config.write_text('[model]\ntemperature = ""\nmax_tokens = ""\n')
        settings = load_settings(config)
        assert settings.model.temperature is None
        assert settings.model.max_tokens is None

    def test_env_empty_string_omits_temperature(self, mocker: MockerFixture) -> None:
        """'' is the only spelling env has for "send no temperature"."""
        mocker.patch.dict("os.environ", {"ELJA_MODEL__TEMPERATURE": ""})
        assert EljaSettings().model.temperature is None

    def test_env_empty_string_omits_max_tokens(self, mocker: MockerFixture) -> None:
        mocker.patch.dict("os.environ", {"ELJA_MODEL__MAX_TOKENS": ""})
        assert EljaSettings().model.max_tokens is None

    def test_env_value_still_parses(self, mocker: MockerFixture) -> None:
        mocker.patch.dict("os.environ", {"ELJA_MODEL__TEMPERATURE": "0.75"})
        assert EljaSettings().model.temperature == 0.75
