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


class TestEffectiveEndpoint:
    def test_openai_default_is_local(self) -> None:
        """The banner must show the real local endpoint, not 'openai'."""
        from elja.model import effective_endpoint

        assert effective_endpoint(ModelConfig()) == "http://localhost:1234/v1"
        assert effective_endpoint(ModelConfig(base_url="http://x:1/v1")) == "http://x:1/v1"
        assert "anthropic" in effective_endpoint(
            ModelConfig(provider="anthropic", name="claude-sonnet-5")
        )


class TestOpenAIEnvKey:
    def test_openai_api_key_env_fallback(self, mocker: MockerFixture) -> None:
        """provider=openai + cloud endpoint honors OPENAI_API_KEY (not lm-studio)."""
        mocker.patch.dict("os.environ", {"OPENAI_API_KEY": "cloud-key"})
        settings = EljaSettings(
            model=ModelConfig(name="gpt-5.2", base_url="https://api.openai.com/v1")
        )
        model = build_model(settings)
        assert model.system == "openai"


class TestMissingExtras:
    def test_missing_anthropic_extra_is_clear(self, mocker: MockerFixture) -> None:
        """Selecting a provider without its extra names the install command."""
        import sys

        from elja.model import ModelProviderError

        mocker.patch.dict(sys.modules, {"pydantic_ai.models.anthropic": None})
        settings = EljaSettings(model=ModelConfig(provider="anthropic", name="claude-sonnet-5"))
        with pytest.raises(ModelProviderError, match="elja\\[anthropic\\]"):
            build_model(settings)

    def test_missing_google_extra_is_clear(self, mocker: MockerFixture) -> None:
        """Same for google."""
        import sys

        from elja.model import ModelProviderError

        mocker.patch.dict(sys.modules, {"pydantic_ai.models.google": None})
        settings = EljaSettings(
            model=ModelConfig(
                provider="google", name="gemini-3-flash-preview", api_key=SecretStr("k")
            )
        )
        with pytest.raises(ModelProviderError, match="elja\\[google\\]"):
            build_model(settings)


class TestGoogleEnvPrecedence:
    def test_google_key_beats_gemini_key(self, mocker: MockerFixture) -> None:
        """GOOGLE_API_KEY wins over GEMINI_API_KEY, matching the SDK convention."""
        mocker.patch.dict("os.environ", {"GOOGLE_API_KEY": "g1", "GEMINI_API_KEY": "g2"})
        settings = EljaSettings(
            model=ModelConfig(provider="google", name="gemini-3-flash-preview")
        )
        assert build_model(settings).system == "google"

    def test_gemini_key_alone_works(self, mocker: MockerFixture) -> None:
        """GEMINI_API_KEY is honored when GOOGLE_API_KEY is absent."""
        mocker.patch.dict("os.environ", {"GEMINI_API_KEY": "g2"})
        settings = EljaSettings(
            model=ModelConfig(provider="google", name="gemini-3-flash-preview")
        )
        assert build_model(settings).system == "google"


class TestValidation:
    def test_unknown_provider_rejected(self) -> None:
        """An unsupported provider name fails config validation."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="provider"):
            ModelConfig(provider="frontier-corp")  # type: ignore[arg-type]
