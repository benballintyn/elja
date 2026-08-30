"""Tests for elja.model."""

from pydantic import SecretStr

from elja.model import build_model
from elja.settings import EljaSettings, ModelConfig


def test_build_model_wires_provider_and_settings() -> None:
    """The factory produces an OpenAI-dialect model pointed at the configured endpoint."""
    settings = EljaSettings()
    model = build_model(settings)
    assert model.model_name == "qwen/qwen3.8-27b"
    assert model.base_url.rstrip("/") == "http://localhost:1234/v1"
    assert model.system == "openai"
    assert model.settings is not None
    assert model.settings.get("temperature") == 0.2
    assert model.settings.get("max_tokens") == 4096


def test_build_model_applies_profile_quirks() -> None:
    """Local OpenAI-compatible servers don't get strict tool definitions by default."""
    model = build_model(EljaSettings())
    assert model.profile["openai_supports_strict_tool_definition"] is False


def test_build_model_respects_overrides() -> None:
    """Custom endpoint/model settings flow through to the built model."""
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
    assert model.base_url.rstrip("/") == "http://example.com:8080/v1"
    assert model.settings is not None
    assert model.settings.get("temperature") == 0.9
    assert model.profile["openai_supports_strict_tool_definition"] is True
