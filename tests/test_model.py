"""Tests for elja.model."""

import pytest
from pydantic import SecretStr
from pytest_mock import MockerFixture

from elja.model import build_model
from elja.settings import EljaSettings, ModelConfig


class TestOpenAIPath:
    def test_defaults_target_local_lm_studio(self) -> None:
        """Unset base_url/api_key on the openai provider mean a local server."""
        settings = EljaSettings()
        model = build_model(settings)
        assert model.model_name == "qwen/qwen3.8-27b"
        assert (model.base_url or "").rstrip("/") == "http://localhost:1234/v1"
        assert model.system == "openai"
        assert model.settings is not None
        assert model.settings.get("temperature") == 0.2
        assert model.settings.get("max_tokens") == 4096

    def test_profile_quirks(self) -> None:
        model = build_model(EljaSettings())
        assert dict(model.profile)["openai_supports_strict_tool_definition"] is False

    def test_overrides(self) -> None:
        settings = EljaSettings(
            model=ModelConfig(
                name="org/some-model",
                base_url="http://example.com:8080/v1",
                api_key=SecretStr("secret"),
                temperature=0.9,
                max_tokens=128,
                supports_strict_tool_definition=True,
            )
        )
        model = build_model(settings)
        assert model.model_name == "org/some-model"
        assert (model.base_url or "").rstrip("/") == "http://example.com:8080/v1"
        assert model.settings is not None
        assert model.settings.get("temperature") == 0.9
        assert dict(model.profile)["openai_supports_strict_tool_definition"] is True


class TestAnthropicPath:
    def test_builds_native_anthropic_model(self) -> None:
        settings = EljaSettings(
            model=ModelConfig(provider="anthropic", name="claude-sonnet-5", api_key=SecretStr("k"))
        )
        model = build_model(settings)
        assert model.system == "anthropic"
        assert model.model_name == "claude-sonnet-5"
        assert model.settings is not None
        assert model.settings.get("temperature") == 0.2

    def test_env_key_fallback(self, mocker: MockerFixture) -> None:
        """Unset api_key defers to the SDK's standard environment variable."""
        mocker.patch.dict("os.environ", {"ANTHROPIC_API_KEY": "env-key"})
        settings = EljaSettings(model=ModelConfig(provider="anthropic", name="claude-sonnet-5"))
        assert build_model(settings).system == "anthropic"

    def test_custom_base_url(self) -> None:
        settings = EljaSettings(
            model=ModelConfig(
                provider="anthropic",
                name="claude-sonnet-5",
                api_key=SecretStr("k"),
                base_url="http://proxy.local:9999",
            )
        )
        model = build_model(settings)
        assert model.base_url is not None and "proxy.local" in model.base_url


class TestGooglePath:
    def test_builds_native_google_model(self) -> None:
        settings = EljaSettings(
            model=ModelConfig(
                provider="google", name="gemini-3-flash-preview", api_key=SecretStr("k")
            )
        )
        model = build_model(settings)
        assert model.system == "google"
        assert model.model_name == "gemini-3-flash-preview"


class TestGoogleMissingKey:
    def test_missing_key_is_clear_error(self) -> None:
        from elja.model import ModelProviderError

        settings = EljaSettings(
            model=ModelConfig(provider="google", name="gemini-3-flash-preview")
        )
        with pytest.raises(ModelProviderError, match="GOOGLE_API_KEY"):
            build_model(settings)


class TestValidation:
    def test_unknown_provider_rejected(self) -> None:
        with pytest.raises(Exception, match="provider"):
            ModelConfig(provider="frontier-corp")  # type: ignore[arg-type]
