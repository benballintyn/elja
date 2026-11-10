"""Tests for elja.session."""

from pathlib import Path

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.test import TestModel

from elja.session import Session, SessionLoadError
from elja.settings import EljaSettings, SessionConfig, WorkspaceConfig


def _some_messages() -> list[ModelMessage]:
    """Produce a realistic message history via a real (test-model) agent run."""
    agent = Agent(TestModel(), instructions="You are a test.")
    result = agent.run_sync("hello")
    return list(result.all_messages())


def _retry_part() -> ModelRequestPart:
    """A retry bound to no tool, as the installed pydantic-ai records one.

    A ``RetryFeedbackPart`` since pydantic-ai 2.60, a ``RetryPromptPart`` before; the
    deprecated class no longer round-trips as itself, so the history uses the current one.
    """
    try:
        from pydantic_ai.messages import RetryFeedbackPart, RetryInfo
    except ImportError:  # pydantic-ai < 2.60
        return RetryPromptPart(content="try again")
    return RetryFeedbackPart(content="try again", retry=RetryInfo(cause="model_retry"))


def _gnarly_messages() -> list[ModelMessage]:
    """A synthetic history covering the part kinds that must round-trip verbatim."""
    return [
        ModelRequest(parts=[UserPromptPart(content="do things")]),
        ModelResponse(
            parts=[
                ThinkingPart(content="pondering...", signature="sig123"),
                TextPart(content="on it"),
                ToolCallPart(tool_name="read_file", args={"path": "a.txt"}, tool_call_id="c1"),
            ],
            provider_name="openai",
            provider_response_id="resp-1",
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name="read_file", content={"data": [1, 2]}, tool_call_id="c1"),
                _retry_part(),
            ]
        ),
        ModelResponse(parts=[TextPart(content="done")]),
    ]


def test_load_missing_returns_empty(tmp_path: Path) -> None:
    session = Session(tmp_path / "nope.json")
    assert session.load() == []


def test_save_load_roundtrip(tmp_path: Path) -> None:
    messages = _some_messages()
    session = Session(tmp_path / "deep" / "dir" / "chat.json")
    session.save(messages)
    assert session.load() == messages


def test_tool_and_thinking_parts_roundtrip_verbatim(tmp_path: Path) -> None:
    """The load-bearing contract: provider state and hairy parts survive persistence."""
    messages = _gnarly_messages()
    session = Session(tmp_path / "gnarly.json")
    session.save(messages)
    loaded = session.load()
    assert loaded == messages
    # Saving what was loaded is byte-identical (nothing lost or reordered).
    session2 = Session(tmp_path / "again.json")
    session2.save(loaded)
    assert session2.path.read_bytes() == session.path.read_bytes()


def test_a_session_saved_before_the_retry_split_still_loads(tmp_path: Path) -> None:
    """A file elja 0.5.0 wrote (pydantic-ai 2.36), with both kinds of retry prompt.

    Since pydantic-ai 2.60 these load as the parts they meant: a ``RetryFeedbackPart``
    for the one bound to no tool, a retried ``ToolReturnPart`` for the other. The
    text is kept either way.
    """
    path = tmp_path / "old.json"
    path.write_text(
        '[{"parts":[{"content":"try again","tool_name":null,"tool_call_id":"r1",'
        '"timestamp":"2026-01-02T03:04:05Z","part_kind":"retry-prompt"},'
        '{"content":"bad args","tool_name":"read_file","tool_call_id":"c1",'
        '"timestamp":"2026-01-02T03:04:05Z","part_kind":"retry-prompt"}],'
        '"timestamp":"2026-01-02T03:04:05Z","instructions":null,"kind":"request",'
        '"run_id":null,"conversation_id":null,"metadata":null,"state":"complete"}]'
    )
    (request,) = Session(path).load()
    assert isinstance(request, ModelRequest)
    assert [getattr(part, "content", None) for part in request.parts] == ["try again", "bad args"]
    assert getattr(request.parts[1], "tool_call_id", None) == "c1"


def test_corrupt_file_raises_actionable_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('[{"kind": "hologram", "parts": []}]')
    with pytest.raises(SessionLoadError, match=str(path)):
        Session(path).load()


def test_save_overwrites(tmp_path: Path) -> None:
    messages = _some_messages()
    session = Session(tmp_path / "chat.json")
    session.save(messages)
    session.save(messages + messages)
    assert len(session.load()) == 2 * len(messages)


def test_for_name_resolves_under_workspace(tmp_path: Path) -> None:
    settings = EljaSettings(
        workspace=WorkspaceConfig(root=tmp_path),
        session=SessionConfig(dir=Path(".elja/sessions")),
    )
    session = Session.for_name(settings, "mychat")
    assert session.path == tmp_path.resolve() / ".elja" / "sessions" / "mychat.json"


def test_for_name_absolute_session_dir(tmp_path: Path) -> None:
    settings = EljaSettings(
        workspace=WorkspaceConfig(root=tmp_path),
        session=SessionConfig(dir=tmp_path / "elsewhere"),
    )
    session = Session.for_name(settings, "mychat")
    assert session.path == tmp_path / "elsewhere" / "mychat.json"


@pytest.mark.parametrize("bad", ["../evil", "/abs/path", "a/b", "", "..", "a b"])
def test_for_name_rejects_path_injection(tmp_path: Path, bad: str) -> None:
    settings = EljaSettings(workspace=WorkspaceConfig(root=tmp_path))
    with pytest.raises(ValueError, match="invalid session name"):
        Session.for_name(settings, bad)


def test_tmp_file_name_is_predictable(tmp_path: Path) -> None:
    """Sessions with dotted or suffix-less names must not share tmp files."""
    messages = _some_messages()
    for name in ("v1.2", "plain"):
        session = Session(tmp_path / name)
        session.save(messages)
        assert session.load() == messages
    assert sorted(p.name for p in tmp_path.iterdir()) == ["plain", "v1.2"]
