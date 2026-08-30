"""Tests for the web_search tool."""

from pathlib import Path

import pytest
from pydantic_ai import ModelRetry
from pytest_mock import MockerFixture

from elja.settings import EljaSettings, ToolsConfig, WorkspaceConfig
from elja.tools import ToolError, build_toolset, do_web_search, web_search

FAKE_RESULTS = [
    {"title": "Python", "href": "https://python.org", "body": "The Python language."},
    {"title": "Docs", "href": "https://docs.python.org", "body": "Official docs."},
]


class TestDoWebSearch:
    def test_formats_results(self, mocker: MockerFixture) -> None:
        ddgs = mocker.patch("elja.tools.DDGS")
        ddgs.return_value.text.return_value = FAKE_RESULTS
        out = do_web_search("python")
        assert "Python — https://python.org" in out
        assert "The Python language." in out
        assert "Docs — https://docs.python.org" in out
        ddgs.return_value.text.assert_called_once_with("python", max_results=5)

    def test_no_results(self, mocker: MockerFixture) -> None:
        ddgs = mocker.patch("elja.tools.DDGS")
        ddgs.return_value.text.return_value = []
        assert "no results" in do_web_search("qwzyx-nothing")

    def test_network_failure_is_tool_error(self, mocker: MockerFixture) -> None:
        ddgs = mocker.patch("elja.tools.DDGS")
        ddgs.return_value.text.side_effect = TimeoutError("net down")
        with pytest.raises(ToolError, match="web search failed"):
            do_web_search("python")


class TestWrapper:
    def test_success_passthrough(self, mocker: MockerFixture) -> None:
        ddgs = mocker.patch("elja.tools.DDGS")
        ddgs.return_value.text.return_value = FAKE_RESULTS
        assert "python.org" in web_search("python")

    def test_failure_becomes_model_retry(self, mocker: MockerFixture) -> None:
        ddgs = mocker.patch("elja.tools.DDGS")
        ddgs.return_value.text.side_effect = OSError("net down")
        with pytest.raises(ModelRetry, match="web search failed"):
            web_search("python")


class TestRegistration:
    def test_enabled_by_default(self, tmp_path: Path) -> None:
        settings = EljaSettings(workspace=WorkspaceConfig(root=tmp_path))
        assert "web_search" in build_toolset(settings).tools

    def test_toggle_disables(self, tmp_path: Path) -> None:
        settings = EljaSettings(
            workspace=WorkspaceConfig(root=tmp_path),
            tools=ToolsConfig(web_search=False),
        )
        assert "web_search" not in build_toolset(settings).tools


@pytest.mark.integration
def test_live_search() -> None:
    out = do_web_search("python programming language")
    assert "https://" in out
