"""End-to-end test against a live LM Studio server.

Run with ``pytest -m integration``. Requires LM Studio serving the configured
model at ``http://localhost:1234/v1``.
"""

from pathlib import Path

import pytest

from elja.cli import run_turn
from elja.session import Session
from elja.settings import EljaSettings, WorkspaceConfig

pytestmark = pytest.mark.integration


async def test_e2e_tool_loop(tmp_path: Path) -> None:
    """Phase-1 acceptance: prompt -> tool calls -> verified side effect -> answer."""
    settings = EljaSettings(workspace=WorkspaceConfig(root=tmp_path))
    session = Session.for_name(settings, "it")
    deltas: list[str] = []
    output = await run_turn(
        settings,
        session,
        "Create a file named hello.txt containing exactly: elja was here\n"
        "Then read it back to confirm, and tell me what it contains.",
        deltas.append,
    )
    assert (tmp_path / "hello.txt").is_file()
    assert "elja was here" in (tmp_path / "hello.txt").read_text()
    assert output
    # The conversation (with tool calls) was persisted and can be resumed.
    assert len(session.load()) >= 4


async def test_e2e_session_resume(tmp_path: Path) -> None:
    """A second turn sees context from the first."""
    settings = EljaSettings(workspace=WorkspaceConfig(root=tmp_path))
    session = Session.for_name(settings, "resume")
    await run_turn(settings, session, "Remember this codeword: snowplow77.", lambda d: None)
    output = await run_turn(settings, session, "What was the codeword?", lambda d: None)
    assert "snowplow77" in output
