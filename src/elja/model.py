"""Model factory: settings -> a configured Pydantic AI model."""

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from elja.settings import EljaSettings


def build_model(settings: EljaSettings) -> OpenAIChatModel:
    """Build the chat model an elja agent runs on.

    Speaks the OpenAI Chat Completions dialect, which covers LM Studio and any
    other OpenAI-compatible endpoint via ``model.base_url``.

    Args:
        settings: Resolved elja settings.

    Returns:
        A configured model ready to pass to an ``Agent``.
    """
    provider = OpenAIProvider(
        base_url=settings.model.base_url,
        api_key=settings.model.api_key,
    )
    profile = OpenAIModelProfile(
        openai_supports_strict_tool_definition=settings.model.supports_strict_tool_definition,
    )
    return OpenAIChatModel(
        settings.model.name,
        provider=provider,
        profile=profile,
        settings=ModelSettings(
            temperature=settings.model.temperature,
            max_tokens=settings.model.max_tokens,
        ),
    )
