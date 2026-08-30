"""Tests for elja.session."""

from pathlib import Path

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
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
                RetryPromptPart(content="try again"),
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
