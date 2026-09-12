"""Tests for elja.model."""

import asyncio

import pytest
from pydantic import SecretStr
from pydantic_ai.models import ModelRequestParameters
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


class TestNativeModelSettings:
    """model.settings carries native ModelSettings through untouched."""

    def test_portable_setting_reaches_the_model(self) -> None:
        settings = EljaSettings(model=ModelConfig(settings={"top_p": 0.4, "seed": 11}))
        built = build_model(settings)
        assert built.settings is not None
        assert built.settings.get("top_p") == 0.4
        assert built.settings.get("seed") == 11
        # The convenience shortcuts still apply alongside it.
        assert built.settings.get("temperature") == 0.2
        assert built.settings.get("max_tokens") == 4096

    def test_provider_specific_reasoning_setting_reaches_the_model(self) -> None:
        """No elja-side enum: the provider's own key passes through verbatim."""
        settings = EljaSettings(
            model=ModelConfig(provider="openai", settings={"openai_reasoning_effort": "high"})
        )
        built = build_model(settings)
        assert built.settings is not None
        assert built.settings.get("openai_reasoning_effort") == "high"

    def test_google_reasoning_setting_reaches_the_model(self) -> None:
        settings = EljaSettings(
            model=ModelConfig(
                provider="google",
                name="gemini-2.5-pro",
                api_key=SecretStr("k"),
                settings={"google_thinking_config": {"thinking_budget": 2048}},
            )
        )
        built = build_model(settings)
        assert built.settings is not None
        assert built.settings.get("google_thinking_config") == {"thinking_budget": 2048}

    def test_anthropic_thinking_setting_reaches_the_model(self) -> None:
        settings = EljaSettings(
            model=ModelConfig(
                provider="anthropic",
                name="claude-sonnet-5",
                api_key=SecretStr("k"),
                settings={"anthropic_thinking": {"type": "enabled", "budget_tokens": 2048}},
            )
        )
        built = build_model(settings)
        assert built.settings is not None
        assert built.settings.get("anthropic_thinking") == {
            "type": "enabled",
            "budget_tokens": 2048,
        }

    def test_none_temperature_omits_the_parameter_entirely(self) -> None:
        """Reasoning models that reject temperature must not receive one."""
        settings = EljaSettings(model=ModelConfig(temperature=None))
        built = build_model(settings)
        assert built.settings is not None
        assert "temperature" not in built.settings
        assert built.settings.get("max_tokens") == 4096

    def test_none_max_tokens_omits_the_parameter_entirely(self) -> None:
        settings = EljaSettings(model=ModelConfig(max_tokens=None))
        built = build_model(settings)
        assert built.settings is not None
        assert "max_tokens" not in built.settings
        assert built.settings.get("temperature") == 0.2

    def test_settings_wins_over_an_unset_convenience_default(self) -> None:
        """Leaving temperature at its default is not a conflict; settings rules."""
        settings = EljaSettings(model=ModelConfig(settings={"temperature": 0.9}))
        built = build_model(settings)
        assert built.settings is not None
        assert built.settings.get("temperature") == 0.9

    def test_empty_settings_changes_nothing(self) -> None:
        built = build_model(EljaSettings(model=ModelConfig(settings={})))
        assert built.settings is not None
        assert built.settings.get("temperature") == 0.2
        assert built.settings.get("max_tokens") == 4096


class TestSettingsAreNotShared:
    """A built model must not reach back into the config it came from."""

    def test_the_built_settings_are_a_copy_of_the_config(self) -> None:
        cfg = ModelConfig(settings={"top_p": 0.4})
        built = build_model(EljaSettings(model=cfg))
        assert built.settings is not None
        assert built.settings is not cfg.settings

    def test_building_twice_does_not_accumulate_into_the_config(self) -> None:
        """The shortcuts are merged into the request settings, not into config.

        Aliasing instead of copying would write temperature/max_tokens back into
        ModelConfig.settings, which also makes a later re-validation raise the
        duplicate-parameter error.
        """
        cfg = ModelConfig(settings={"top_p": 0.4})
        first = build_model(EljaSettings(model=cfg))
        second = build_model(EljaSettings(model=cfg))
        assert cfg.settings == {"top_p": 0.4}
        assert first.settings is not second.settings
        for built in (first, second):
            assert built.settings is not None
            assert built.settings.get("temperature") == 0.2

    async def test_per_run_settings_do_not_leak_between_concurrent_runs(self) -> None:
        """E3: no setting bleed between simultaneous runs."""
        model = build_model(EljaSettings(model=ModelConfig(settings={"top_p": 0.4})))
        before = dict(model.settings or {})
        params = ModelRequestParameters()
        resolved = await asyncio.gather(
            *(
                asyncio.to_thread(model.prepare_request, {"temperature": t}, params)
                for t in (0.1, 0.5, 0.9)
            )
        )
        seen = [(s or {}).get("temperature") for s, _ in resolved]
        assert sorted(t for t in seen if t is not None) == [0.1, 0.5, 0.9]
        # Each run still saw the agent-level settings, and none of them wrote back.
        for merged, _ in resolved:
            assert (merged or {}).get("top_p") == 0.4
        assert dict(model.settings or {}) == before


class TestRequestConstruction:
    """Assert the settings that reach a REQUEST, not just the model attribute."""

    def test_a_portable_thinking_level_is_resolved_out_of_the_request_settings(self) -> None:
        """prepare_request lifts `thinking` into request params and strips it."""
        settings = EljaSettings(
            model=ModelConfig(
                provider="anthropic",
                name="claude-sonnet-5",
                api_key=SecretStr("k"),
                settings={"thinking": "high"},
            )
        )
        merged, params = build_model(settings).prepare_request(None, ModelRequestParameters())
        assert "thinking" not in (merged or {})
        assert params.thinking == "high"

    def test_provider_specific_settings_survive_request_preparation(self) -> None:
        settings = EljaSettings(
            model=ModelConfig(provider="openai", settings={"openai_reasoning_effort": "high"})
        )
        merged, _ = build_model(settings).prepare_request(None, ModelRequestParameters())
        assert (merged or {}).get("openai_reasoning_effort") == "high"
        assert (merged or {}).get("max_tokens") == 4096

    def test_an_omitted_temperature_is_absent_from_the_request(self) -> None:
        settings = EljaSettings(model=ModelConfig(temperature=None))
        merged, _ = build_model(settings).prepare_request(None, ModelRequestParameters())
        assert "temperature" not in (merged or {})

    def test_a_google_request_carries_its_own_dialect_settings(self) -> None:
        settings = EljaSettings(
            model=ModelConfig(
                provider="google",
                name="gemini-2.5-pro",
                api_key=SecretStr("k"),
                settings={"google_thinking_config": {"thinking_budget": 2048}},
            )
        )
        merged, _ = build_model(settings).prepare_request(None, ModelRequestParameters())
        assert (merged or {}).get("google_thinking_config") == {"thinking_budget": 2048}


class TestImplementsCountTokensSeesThroughWrappers:
    """A wrapper DEFINES count_tokens in order to delegate it.

    So the plain "does the class override the base method" test answers True for
    every wrapper regardless of what it wraps — and `InstrumentedModel` is a
    `WrapperModel`, put around a model whenever instrumentation is on
    (`instrument=True`, `Agent.instrument_all()`, Logfire). A host with Logfire
    enabled would have been told its OpenAI-compatible endpoint supports
    count-tokens, turned the flag on, and failed every request: the exact failure
    this helper exists to prevent. `settings.py` directs hosts here by name.
    """

    def test_an_instrumented_model_reports_what_it_wraps(self) -> None:
        from pydantic_ai.models.instrumented import InstrumentedModel

        from elja.model import build_model, implements_count_tokens

        inner = build_model(EljaSettings())
        assert type(inner).__name__ == "OpenAIChatModel"
        # Both directions: the wrapper must not invent support...
        assert implements_count_tokens(inner) is False
        assert implements_count_tokens(InstrumentedModel(inner)) is False

    def test_a_wrapper_does_not_hide_real_support_either(self) -> None:
        """...nor mask it. Unwrapping that always answered False would be as wrong."""
        from pydantic_ai.models.instrumented import InstrumentedModel

        from elja.model import build_model, implements_count_tokens

        inner = build_model(EljaSettings(model={"provider": "anthropic", "api_key": "x"}))  # type: ignore[arg-type]
        assert implements_count_tokens(inner) is True
        assert implements_count_tokens(InstrumentedModel(inner)) is True

    def test_nested_wrappers_are_unwrapped_to_the_bottom(self) -> None:
        from pydantic_ai.models.instrumented import InstrumentedModel

        from elja.model import build_model, implements_count_tokens

        inner = build_model(EljaSettings())
        assert implements_count_tokens(InstrumentedModel(InstrumentedModel(inner))) is False

    def test_a_wrapper_class_answers_false_rather_than_guessing(self) -> None:
        """There is no instance to look through, so there is nothing to report."""
        from pydantic_ai.models.wrapper import WrapperModel

        from elja.model import implements_count_tokens

        assert implements_count_tokens(WrapperModel) is False

    def test_a_plain_class_is_still_answered_from_the_class(self) -> None:
        from pydantic_ai.models.openai import OpenAIChatModel

        from elja.model import implements_count_tokens

        assert implements_count_tokens(OpenAIChatModel) is False


def test_model_class_name_reads_the_builders_own_table() -> None:
    """A diagnostic that names a class must read it where the class is chosen.

    Only `openai` currently fails the count-tokens check, and its class really is
    `OpenAIChatModel`, so hardcoding that string in the error message is an
    equivalent mutant today — there is no provider for which the table and the
    literal differ. The table lookup is still the right call: it is what keeps the
    message true if a provider ever drops `count_tokens` upstream. Pinned here
    directly, since the message cannot distinguish it.
    """
    from elja.model import model_class_name

    assert model_class_name("openai") == "OpenAIChatModel"
    assert model_class_name("anthropic") == "AnthropicModel"
    assert model_class_name("google") == "GoogleModel"
