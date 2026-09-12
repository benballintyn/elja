"""Tests for elja.settings."""

import json
import os
import subprocess
import sys
from collections import deque
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal, cast

import pytest
from pydantic import BaseModel, SecretStr, ValidationError
from pytest_mock import MockerFixture

import elja
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

    def test_an_unknown_key_nested_inside_a_value_is_refused_by_its_path(self) -> None:
        """A typo one level down must not silently disable the setting.

        Pydantic drops undeclared TypedDict keys at every depth, and pydantic-ai
        reads `if config := model_settings.get(...)`, so a misspelled budget
        leaves an empty dict that reads as "reasoning off". Before this check the
        provider SDK would at least have refused the request by name.
        """
        with pytest.raises(ValidationError, match=r"google_thinking_config\.typo_here"):
            ModelConfig(
                provider="google",
                settings={"google_thinking_config": {"thinking_budget": 8, "typo_here": 1}},
            )

    def test_a_typo_beside_a_valid_key_is_still_refused_at_the_top_level(self) -> None:
        with pytest.raises(ValidationError, match=r"'nonsense'"):
            ModelConfig(settings={"top_p": 0.5, "nonsense": 1})

    def test_a_value_coerced_to_a_different_shape_is_left_to_pydantic(self) -> None:
        """Walking two trees only reports MISSING keys, never a type change.

        ``google_safety_settings`` promotes plain strings into enum objects, so
        the validated value is no longer a mapping; the walk must say nothing
        rather than invent a dropped key.
        """
        from elja.settings import _dropped_keys

        assert _dropped_keys({"a": {"b": 1}}, {"a": "now-a-string"}) == []
        assert _dropped_keys({"a": {"b": 1}}, {"a": {}}) == ["a.b"]
        # Lists of differing length are pydantic's problem, not a missing key.
        assert _dropped_keys({"a": [{"b": 1}]}, {"a": []}) == []
        assert _dropped_keys({"a": [{"b": 1}]}, {"a": [{}]}) == ["a.0.b"]

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

    def test_both_clashing_parameters_are_named_at_once(self) -> None:
        """Naming only the first would send the user round the loop twice."""
        with pytest.raises(ValidationError) as exc:
            ModelConfig(
                temperature=0.5, max_tokens=100, settings={"temperature": 0.1, "max_tokens": 2}
            )
        message = str(exc.value)
        assert "model.max_tokens" in message
        assert "model.temperature" in message

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


class TestImportWeight:
    def test_importing_elja_does_not_load_a_provider_sdk(self) -> None:
        """A host pays for its own provider, not for all three.

        The settings validator must not touch pydantic_ai.models.openai for the
        default empty [model.settings]; doing so pulled the entire OpenAI SDK
        into every `import elja` and defeated model.py's lazy provider imports.
        A subprocess, because this session has already imported plenty.
        """
        # Point the subprocess at the tree under test. Without cwd/PYTHONPATH it
        # imports whatever `elja` happens to be installed, so in a git worktree
        # or against a built wheel the check silently measures a different file.
        root = str(Path(elja.__file__).resolve().parent.parent)
        result = subprocess.run(
            [sys.executable, "-c", "import elja, sys; print('openai' in sys.modules)"],
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
            cwd=root,
            env={**os.environ, "PYTHONPATH": root},
        )
        assert result.stdout.strip() == "False"


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


class TestLazyValidatorsAreMaterialized:
    """A pydantic ``Iterable[...]`` leaf validates into a one-shot iterator.

    The settings dict is reused for every request of every run, so an unforced
    iterator means the first request carries the value and every later one
    carries an empty list — a provider feature that switches itself off after
    one turn, with no error anywhere.
    """

    def test_a_list_valued_nested_setting_can_be_consumed_twice(self) -> None:
        cfg = ModelConfig(
            provider="anthropic",
            name="claude-sonnet-5",
            settings={
                "anthropic_context_management": {"edits": [{"type": "clear_tool_uses_20250919"}]}
            },
        )
        edits = cfg.settings["anthropic_context_management"]["edits"]
        # Equality alone would pass against a fresh iterator; consume it twice.
        assert list(edits) == [{"type": "clear_tool_uses_20250919"}]
        assert list(edits) == [{"type": "clear_tool_uses_20250919"}]
        assert isinstance(edits, list)

    def test_the_resolved_config_stays_json_serializable(self) -> None:
        """A host that logs or persists its resolved config must not break."""
        cfg = ModelConfig(
            provider="anthropic",
            name="claude-sonnet-5",
            settings={
                "anthropic_context_management": {"edits": [{"type": "clear_tool_uses_20250919"}]}
            },
        )
        assert json.loads(json.dumps(cfg.model_dump()))["settings"][
            "anthropic_context_management"
        ]["edits"] == [{"type": "clear_tool_uses_20250919"}]

    def test_the_built_model_carries_the_value_on_every_request(self) -> None:
        from pydantic_ai.models import ModelRequestParameters

        from elja.model import build_model

        settings = EljaSettings(
            model=ModelConfig(
                provider="anthropic",
                name="claude-sonnet-5",
                api_key=SecretStr("k"),
                settings={
                    "anthropic_context_management": {
                        "edits": [{"type": "clear_tool_uses_20250919"}]
                    }
                },
            )
        )
        model = build_model(settings)
        params = ModelRequestParameters()
        first, _ = model.prepare_request(None, params)
        second, _ = model.prepare_request(None, params)
        for merged in (first, second):
            # dict(...) so mypy does not read this as a ModelSettings key lookup;
            # the dialect's own keys are not in the portable TypedDict.
            management = cast("dict[str, Any]", dict(merged or {})["anthropic_context_management"])
            assert list(management["edits"]) == [{"type": "clear_tool_uses_20250919"}]


class TestValidationIsIdempotent:
    """A resolved config must survive its own serialization round trip."""

    def test_a_model_config_round_trip_does_not_raise(self) -> None:
        cfg = ModelConfig(settings={"temperature": 0.7})
        again = ModelConfig.model_validate(cfg.model_dump())
        assert again.settings["temperature"] == 0.7

    def test_a_whole_settings_round_trip_does_not_raise(self) -> None:
        original = EljaSettings(model=ModelConfig(settings={"max_tokens": 100}))
        again = EljaSettings.model_validate(original.model_dump())
        assert again.model.settings["max_tokens"] == 100

    def test_an_explicit_non_default_value_is_still_a_clash(self) -> None:
        """The round-trip fix must not disarm the check it replaced."""
        with pytest.raises(ValidationError, match=r"duplicates model\.temperature"):
            ModelConfig(temperature=0.5, settings={"temperature": 0.7})


class TestCountTokensSupportIsAskedOfTheModel:
    def test_the_capability_is_read_from_the_class_not_a_provider_list(self) -> None:
        from pydantic_ai.models import Model
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.models.google import GoogleModel
        from pydantic_ai.models.openai import OpenAIChatModel

        from elja.model import implements_count_tokens

        # `is Model.count_tokens` rather than a __dict__ check: a shared base
        # class inserted upstream would fool the latter.
        assert OpenAIChatModel.count_tokens is Model.count_tokens
        assert not implements_count_tokens(OpenAIChatModel)
        assert implements_count_tokens(AnthropicModel)
        assert implements_count_tokens(GoogleModel)

    def test_it_accepts_an_instance_as_well_as_a_class(self) -> None:
        from pydantic_ai.models.function import FunctionModel

        from elja.model import implements_count_tokens

        model = FunctionModel(lambda m, i: None)  # type: ignore[arg-type,return-value]
        assert not implements_count_tokens(model)

    def test_a_missing_extra_does_not_block_the_flag(self, mocker: MockerFixture) -> None:
        """build_model's "install the extra" error is the one the user needs."""
        from elja.model import provider_implements_count_tokens

        mocker.patch("elja.model.importlib.import_module", side_effect=ImportError("no google"))
        assert provider_implements_count_tokens("google") is True


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


# Every caller-writable container shape for an `Iterable[...]` leaf. The iterator
# forms are the ones that matter: pydantic consumes them while validating, so the
# dropped-key walk sees an exhausted original unless the caller's structure is
# snapshotted first.
_CONTAINERS: list[Callable[[list[object]], object]] = [
    list,
    tuple,
    deque,
    lambda items: (item for item in items),
    iter,
    lambda items: map(lambda item: item, items),
]
_CONTAINER_IDS = ["list", "tuple", "deque", "generator", "iter", "map"]


class TestATypoBehindALazyLeafIsStillRefused:
    """The one place where the order of two steps decides a silent drop.

    `anthropic_container.skills` is annotated `Iterable[...]`, so pydantic
    validates it into a once-consumable `ValidatorIterator`. Two things then have
    to happen in order: materialize it into a real list, *then* walk the two trees
    for dropped keys. Swap them and the walk sees an iterator where it expects a
    sequence, gives up, and a misspelled key nested inside is accepted in silence —
    with 289 tests green, because nothing else puts a typo behind a lazy leaf.

    Parametrized over the container type because that is a second, independent
    skip, and it has been reached TWICE. The walk first required the original side
    be a `list`, so a tuple literal skipped it; widening that to `Sequence` left
    generators and other iterators skipping it, because pydantic consumes an
    iterator while validating and there is nothing left on the original side to
    compare. The fix is to snapshot the caller's structure first, which is why a
    generator now refuses like a list. TOML and env can only produce lists; this is
    the programmatic surface every test here uses.
    """

    @staticmethod
    def _skill(**extra: object) -> dict[str, object]:
        return {"skill_id": "s", "type": "custom", **extra}

    @pytest.mark.parametrize("container", _CONTAINERS, ids=_CONTAINER_IDS)
    def test_a_nested_typo_is_refused_by_its_dotted_path(
        self, container: Callable[[list[object]], object]
    ) -> None:
        with pytest.raises(ValidationError, match=r"anthropic_container\.skills\.0\.bogus"):
            ModelConfig(
                provider="anthropic",
                settings={
                    "anthropic_container": {
                        "id": "c",
                        "skills": container([self._skill(bogus=1)]),
                    }
                },
            )

    @pytest.mark.parametrize("container", _CONTAINERS, ids=_CONTAINER_IDS)
    def test_the_same_config_spelled_correctly_is_accepted_and_materialized(
        self, container: Callable[[list[object]], object]
    ) -> None:
        """The positive control: the walk must not refuse a correct nested value.

        And the materialization still has to hold — the value survives being read
        twice, which is what the lazy iterator broke.
        """
        cfg = ModelConfig(
            provider="anthropic",
            settings={"anthropic_container": {"id": "c", "skills": container([self._skill()])}},
        )
        skills = cfg.settings["anthropic_container"]["skills"]
        assert isinstance(skills, list)
        assert [dict(s) for s in skills] == [dict(s) for s in skills], "consumed after one read"
        assert len(skills) == 1


class TestEveryUsageLimitsFieldIsReachableFromConfig:
    """E3 asked for the *full* surface, so the surface itself is the assertion.

    Both forwarding tests hand-list the eight fields, and the inheritance test
    compares against `build_usage_limits(settings)` — so a field pydantic-ai adds
    in a 2.x minor is absent from both sides, silently takes the upstream default,
    and every test still passes. `AGENTS.md` says "Pin-watch: 2.x moves fast", and
    this is what makes the next bump announce itself instead of quietly narrowing
    what a host can configure.
    """

    def test_the_config_covers_exactly_the_upstream_dataclass(self) -> None:
        from dataclasses import fields

        from pydantic_ai.usage import UsageLimits

        from elja.settings import LimitsConfig

        assert {f.name for f in fields(UsageLimits)} == set(LimitsConfig.model_fields)


class TestACeilingOfZeroIsAConfigError:
    """Zero refuses the first request rather than capping anything.

    Every other ceiling in this section carries `ge=1`; these two were widened from
    `int` to `int | None` in the same change and missed it, so `request_limit = 0`
    validated and then refused every run. `SubagentConfig.request_limit` already had
    the guard, which is the inconsistency that makes it a bug rather than a choice.
    """

    @pytest.mark.parametrize("field", ["request_limit", "total_tokens_limit"])
    @pytest.mark.parametrize("value", [0, -3])
    def test_zero_or_negative_is_refused(self, field: str, value: int) -> None:
        from elja.settings import LimitsConfig

        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            LimitsConfig(**{field: value})  # type: ignore[arg-type]

    @pytest.mark.parametrize("field", ["request_limit", "total_tokens_limit"])
    def test_one_and_none_are_both_still_accepted(self, field: str) -> None:
        """A ceiling of one is a real choice, and None means unlimited."""
        from elja.settings import LimitsConfig

        assert getattr(LimitsConfig(**{field: 1}), field) == 1  # type: ignore[arg-type]
        assert getattr(LimitsConfig(**{field: None}), field) is None  # type: ignore[arg-type]


class TestExtraBodyIsPassedThroughUntouched:
    """`extra_body` is the one annotation pydantic does not validate, so nothing else sees it.

    Which made `_materialize` the only thing that did — and it rewrote every iterable
    into a list. A host handing `extra_body` a pydantic model for a local server's own
    knobs (vLLM's `chat_template_kwargs`, `guided_json`) got a list of key/value pairs
    in the request body instead of a loud failure at JSON encode: a wrong request that
    looks like a right one. The walk now tests `Iterator`, not `Iterable`.
    """

    @staticmethod
    def _body(value: object) -> object:
        return ModelConfig(settings={"extra_body": value}).settings["extra_body"]

    def test_a_pydantic_model_survives(self) -> None:
        class Extras(BaseModel):
            chat_template_kwargs: dict[str, bool] = {"enable_thinking": False}

        body = self._body(Extras())
        assert isinstance(body, Extras)
        assert body.chat_template_kwargs == {"enable_thinking": False}

    @pytest.mark.parametrize(
        "value",
        [(1, 2), {1, 2}, bytearray(b"hi"), range(3), {"guided_json": {"a": 1}}],
        ids=["tuple", "set", "bytearray", "range", "dict"],
    )
    def test_every_other_iterable_shape_survives(self, value: object) -> None:
        """All of these are `Iterable` and none is a once-consumable `Iterator`."""
        body = self._body(value)
        assert body == value
        assert type(body) is type(value)

    def test_a_real_iterator_is_still_materialized(self) -> None:
        """The narrowing must not lose the fix it was narrowed from.

        `extra_body` is not where the lazy-validator problem lives, but if a caller does
        put an iterator there it has the same read-once hazard, so it is still forced.
        """
        body = self._body(iter([1, 2]))
        assert body == [1, 2]


class TestTheLegacySequenceProtocolIsWalkedToo:
    """The third container shape to skip the dropped-key walk, each one step further out.

    `Iterable`'s subclass hook checks only `__iter__`, but `iter()` falls back to
    `__getitem__` — so pydantic validates a legacy sequence happily while the walk
    gave up and the misspelled key inside vanished. The walk's type test now mirrors
    what pydantic accepts rather than what `isinstance` recognizes.
    """

    class _GetItemOnly:
        """A sequence by the old protocol: indexable, with no `__iter__`."""

        def __init__(self, items: list[object]) -> None:
            self._items = list(items)

        def __getitem__(self, index: int) -> object:
            return self._items[index]

    def test_a_nested_typo_inside_one_is_refused(self) -> None:
        with pytest.raises(ValidationError, match=r"anthropic_container\.skills\.0\.bogus"):
            ModelConfig(
                provider="anthropic",
                settings={
                    "anthropic_container": {
                        "id": "c",
                        "skills": self._GetItemOnly(
                            [{"skill_id": "s", "type": "custom", "bogus": 1}]
                        ),
                    }
                },
            )

    def test_a_correct_one_is_accepted_and_materialized(self) -> None:
        cfg = ModelConfig(
            provider="anthropic",
            settings={
                "anthropic_container": {
                    "id": "c",
                    "skills": self._GetItemOnly([{"skill_id": "s", "type": "custom"}]),
                }
            },
        )
        skills = cfg.settings["anthropic_container"]["skills"]
        assert isinstance(skills, list)
        assert len(skills) == 1

    def test_something_that_is_not_a_sequence_at_all_is_left_alone(self) -> None:
        """The fallback must not turn a failed conversion into a spurious rejection."""
        cfg = ModelConfig(settings={"extra_body": object()})
        assert isinstance(cfg.settings["extra_body"], object)
