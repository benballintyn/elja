"""Tests for elja.session."""

from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from elja.session import Session
from elja.settings import EljaSettings, SessionConfig, WorkspaceConfig


def _some_messages() -> list:  # type: ignore[type-arg]
    """Produce a realistic message history via a real (test-model) agent run."""
    agent = Agent(TestModel(), instructions="You are a test.")
    result = agent.run_sync("hello")
    return list(result.all_messages())


def test_load_missing_returns_empty(tmp_path: Path) -> None:
    session = Session(tmp_path / "nope.json")
    assert session.load() == []


def test_save_load_roundtrip(tmp_path: Path) -> None:
    messages = _some_messages()
    session = Session(tmp_path / "deep" / "dir" / "chat.json")
    session.save(messages)
    assert session.load() == messages


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
