"""Tests for the web_search tool."""

from pathlib import Path

import pytest
from ddgs.exceptions import DDGSException
from pytest_mock import MockerFixture

from elja.deps import EljaDeps
from elja.settings import EljaSettings, ToolsConfig, WorkspaceConfig
from elja.tools import ToolError, build_toolset, do_web_search

FAKE_RESULTS = [
    {"title": "Python", "href": "https://python.org", "body": "The Python language."},
    {"title": "Docs", "href": "https://docs.python.org", "body": "Official docs."},
]


@pytest.fixture
def deps(tmp_path: Path) -> EljaDeps:
    return EljaDeps.from_settings(EljaSettings(workspace=WorkspaceConfig(root=tmp_path)))


class TestDoWebSearch:
    def test_formats_results(self, deps: EljaDeps, mocker: MockerFixture) -> None:
        ddgs = mocker.patch("elja.tools.DDGS")
        ddgs.return_value.text.return_value = FAKE_RESULTS
        out = do_web_search(deps, "python")
        assert "Python — https://python.org" in out
        assert "The Python language." in out
        assert "Docs — https://docs.python.org" in out
        ddgs.return_value.text.assert_called_once_with(
            "python", max_results=5, backend="duckduckgo"
        )

    def test_no_results_is_not_an_error(self, deps: EljaDeps, mocker: MockerFixture) -> None:
        """ddgs raises on zero hits; that must read as a normal outcome, not a failure."""
        ddgs = mocker.patch("elja.tools.DDGS")
        ddgs.return_value.text.side_effect = DDGSException("No results found.")
        assert "no results" in do_web_search(deps, "qwzyx-nothing")

    def test_ratelimit_is_tool_error(self, deps: EljaDeps, mocker: MockerFixture) -> None:
        ddgs = mocker.patch("elja.tools.DDGS")
        ddgs.return_value.text.side_effect = DDGSException("rate limited")
        with pytest.raises(ToolError, match="web search failed"):
            do_web_search(deps, "python")

    def test_network_failure_is_tool_error(self, deps: EljaDeps, mocker: MockerFixture) -> None:
        ddgs = mocker.patch("elja.tools.DDGS")
        ddgs.return_value.text.side_effect = OSError("net down")
        with pytest.raises(ToolError, match="web search failed"):
            do_web_search(deps, "python")

    def test_huge_results_are_capped(self, tmp_path: Path, mocker: MockerFixture) -> None:
        settings = EljaSettings(workspace=WorkspaceConfig(root=tmp_path, max_tool_output_chars=50))
        deps = EljaDeps.from_settings(settings)
        ddgs = mocker.patch("elja.tools.DDGS")
        ddgs.return_value.text.return_value = [{"title": "T", "href": "u", "body": "x" * 500}]
        assert "truncated" in do_web_search(deps, "python")


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


class TestWrapperViaAgent:
    async def test_search_failure_surfaces_as_retry(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        from pydantic_ai import Agent
        from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
        from pydantic_ai.models.function import AgentInfo, FunctionModel

        ddgs = mocker.patch("elja.tools.DDGS")
        ddgs.return_value.text.side_effect = [OSError("net down"), FAKE_RESULTS]
        settings = EljaSettings(workspace=WorkspaceConfig(root=tmp_path))
        deps = EljaDeps.from_settings(settings)
        calls: list[int] = []

        def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append(1)
            if len(calls) <= 2:
                return ModelResponse(
                    parts=[ToolCallPart(tool_name="web_search", args={"query": "python"})]
                )
            return ModelResponse(parts=[TextPart(content="found it")])

        agent = Agent(
            FunctionModel(script), deps_type=EljaDeps, toolsets=[build_toolset(settings)]
        )
        result = await agent.run("search", deps=deps)
        assert result.output == "found it"


@pytest.mark.integration
def test_live_search(tmp_path: Path) -> None:
    deps = EljaDeps.from_settings(EljaSettings(workspace=WorkspaceConfig(root=tmp_path)))
    out = do_web_search(deps, "python programming language")
    assert "https://" in out
    assert " — " in out
