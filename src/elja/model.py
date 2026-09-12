"""Model factory: settings -> a configured Pydantic AI model.

Three provider dialects are supported via ``model.provider``:

- ``openai`` (default): the OpenAI Chat Completions dialect — covers LM
  Studio, Ollama, vLLM, OpenRouter, api.openai.com, and any other
  OpenAI-compatible endpoint via ``model.base_url``.
- ``anthropic``: the native Anthropic Messages API.
- ``google``: the native Gemini API.

Anthropic/Google support needs the matching extra (``elja[anthropic]`` /
``elja[google]``). When ``api_key`` is unset, those SDKs fall back to their
standard environment variables (``ANTHROPIC_API_KEY``, ``GOOGLE_API_KEY``).
Everything above this factory operates on pydantic-ai's normalized types, so
tools, skills, compaction, sessions, and sub-agents are provider-independent.
"""

import importlib
import os
from typing import Any, cast

from pydantic_ai.models import Model
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings

from elja.settings import EljaSettings, ModelConfig

# The OpenAI-dialect defaults keep the out-of-the-box experience local-first.
_LOCAL_BASE_URL = "http://localhost:1234/v1"
_LOCAL_API_KEY = "lm-studio"


class ModelProviderError(Exception):
    """The selected provider can't be built — missing extra or credentials."""


def _model_settings(cfg: ModelConfig) -> ModelSettings:
    """Native settings for this config: ``settings`` plus the two shortcuts.

    ``temperature``/``max_tokens`` are omitted entirely when ``None`` (some
    reasoning models reject them). ``ModelConfig`` already refuses a key
    spelled in both places, so the merge order cannot hide a conflict.
    """
    merged: dict[str, Any] = dict(cfg.settings)
    for key, value in (("temperature", cfg.temperature), ("max_tokens", cfg.max_tokens)):
        if value is not None and key not in merged:
            merged[key] = value
    # ModelSettings is a total=False TypedDict; a dict of validated keys is the
    # only way to build one with a dynamic key set.
    return cast(ModelSettings, merged)


def _build_openai(cfg: ModelConfig) -> Model:
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.profiles.openai import OpenAIModelProfile
    from pydantic_ai.providers.openai import OpenAIProvider

    api_key = (
        cfg.api_key.get_secret_value()
        if cfg.api_key is not None
        else os.environ.get("OPENAI_API_KEY") or _LOCAL_API_KEY
    )
    return OpenAIChatModel(
        cfg.name,
        provider=OpenAIProvider(base_url=cfg.base_url or _LOCAL_BASE_URL, api_key=api_key),
        profile=OpenAIModelProfile(
            openai_supports_strict_tool_definition=cfg.supports_strict_tool_definition,
        ),
        settings=_model_settings(cfg),
    )


def _build_anthropic(cfg: ModelConfig) -> Model:
    try:
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider
    except ImportError as exc:
        raise ModelProviderError(
            "provider 'anthropic' needs the anthropic extra: pip install 'elja[anthropic]'"
        ) from exc
    api_key = cfg.api_key.get_secret_value() if cfg.api_key is not None else None
    # Explicit branches: the providers' typed overloads don't uniformly accept
    # base_url=None, and mypy strict holds us to them.
    if cfg.base_url is None:
        provider = AnthropicProvider(api_key=api_key)
    else:
        provider = AnthropicProvider(api_key=api_key, base_url=cfg.base_url)
    return AnthropicModel(cfg.name, provider=provider, settings=_model_settings(cfg))


def _build_google(cfg: ModelConfig) -> Model:
    try:
        from pydantic_ai.models.google import GoogleModel
        from pydantic_ai.providers.google import GoogleProvider
    except ImportError as exc:
        raise ModelProviderError(
            "provider 'google' needs the google extra: pip install 'elja[google]'"
        ) from exc
    key = (
        cfg.api_key.get_secret_value()
        if cfg.api_key is not None
        else os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    )
    if not key:
        raise ModelProviderError(
            "provider 'google' needs model.api_key or the GOOGLE_API_KEY / GEMINI_API_KEY env var"
        )
    provider = (
        GoogleProvider(api_key=key)
        if cfg.base_url is None
        else GoogleProvider(api_key=key, base_url=cfg.base_url)
    )
    return GoogleModel(cfg.name, provider=provider, settings=_model_settings(cfg))


# Which class each provider dialect builds. Used to answer capability
# questions about a model elja has not built yet.
_MODEL_CLASSES = {
    "openai": ("pydantic_ai.models.openai", "OpenAIChatModel"),
    "anthropic": ("pydantic_ai.models.anthropic", "AnthropicModel"),
    "google": ("pydantic_ai.models.google", "GoogleModel"),
}


def implements_count_tokens(model: "Model | type[Model]") -> bool:
    """Whether this model can count a request's tokens before sending it.

    ``UsageLimits.count_tokens_before_request`` makes pydantic-ai call
    ``Model.count_tokens`` before **every** request, and the base implementation
    raises ``NotImplementedError``. So a model that does not override it turns
    that flag from an ineffective setting into a run that dies on its first
    request.

    A host injecting its own model should check it here rather than inferring
    support from a provider name: ``model.provider`` describes the model elja
    *would build*, which on the application path is not the model being used.

    Wrappers are unwrapped to the **bottom** of the chain, because
    ``WrapperModel`` *defines* ``count_tokens`` in order to delegate it — so the
    plain override test answers True for every wrapper regardless of what it wraps.
    ``InstrumentedModel`` is a ``WrapperModel``, and it is what pydantic-ai puts
    around a model whenever instrumentation is on (``instrument=True``,
    ``Agent.instrument_all()``, Logfire). Measured: an instrumented
    ``OpenAIChatModel`` answered True and then raised ``NotImplementedError`` on the
    first request — the exact failure this function exists to prevent.

    Unwrapping to the bottom rather than stopping at the first wrapper that defines
    the method is deliberate: ``ConcurrencyLimitedModel`` is a shipped
    ``WrapperModel`` that defines ``count_tokens`` *solely* to delegate it under a
    semaphore, so a top-down test would report True for
    ``ConcurrencyLimitedModel(OpenAIChatModel)``. The cost is the mirror case — a
    host's own wrapper that supplies a real implementation is reported False even
    though the call would have worked. That is the safe direction: the host declines
    a feature rather than shipping a run that dies.

    A wrapper *class* has no instance to look through, so it answers False rather
    than guessing at what it would wrap.

    Args:
        model: A model instance or class.

    Returns:
        True if the model at the BOTTOM of the wrapper chain overrides
        ``count_tokens``. A wrapper supplying its own implementation reads False.
    """
    if isinstance(model, type):
        return not issubclass(model, WrapperModel) and model.count_tokens is not Model.count_tokens
    while isinstance(model, WrapperModel):
        model = model.wrapped
    return type(model).count_tokens is not Model.count_tokens


def model_class_name(provider: str) -> str:
    """The name of the model class elja builds for this provider.

    Read from the same table the builders use, so a diagnostic naming the class
    cannot drift from the class actually constructed.
    """
    return _MODEL_CLASSES[provider][1]


def provider_implements_count_tokens(provider: str) -> bool:
    """Whether the model elja builds for this provider can count tokens.

    Returns True when the provider's optional dependency is missing, because
    the class cannot be inspected and :func:`build_model` will report the
    missing extra first — a clearer error than a capability complaint.
    """
    module_name, class_name = _MODEL_CLASSES[provider]
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return True
    return implements_count_tokens(getattr(module, class_name))


def effective_endpoint(cfg: ModelConfig) -> str:
    """The endpoint a built model will actually talk to (for display)."""
    if cfg.base_url is not None:
        return cfg.base_url
    return _LOCAL_BASE_URL if cfg.provider == "openai" else f"{cfg.provider} API"


_BUILDERS = {
    "openai": _build_openai,
    "anthropic": _build_anthropic,
    "google": _build_google,
}


def build_model(settings: EljaSettings) -> Model:
    """Build the chat model an elja agent runs on.

    Args:
        settings: Resolved elja settings.

    Returns:
        A configured model ready to pass to an ``Agent``.

    Raises:
        ModelProviderError: If the selected provider's optional dependency is
            not installed, or required credentials are missing.
    """
    return _BUILDERS[settings.model.provider](settings.model)
