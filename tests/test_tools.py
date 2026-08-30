"""Tests for elja.deps and elja.tools."""

from pathlib import Path

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from elja.deps import EljaDeps
from elja.settings import EljaSettings, ToolsConfig, WorkspaceConfig
from elja.tools import (
    ToolError,
    build_toolset,
    do_list_dir,
    do_read_file,
    do_run_shell,
    do_write_file,
)


@pytest.fixture
def deps(tmp_path: Path) -> EljaDeps:
    """Deps rooted in a temporary workspace."""
    settings = EljaSettings(workspace=WorkspaceConfig(root=tmp_path))
    return EljaDeps.from_settings(settings)


class TestPathResolution:
    def test_read_write_roundtrip(self, deps: EljaDeps) -> None:
        do_write_file(deps, "sub/dir/hello.txt", "hi there")
        assert do_read_file(deps, "sub/dir/hello.txt") == "hi there"

    def test_read_missing_file_raises_tool_error(self, deps: EljaDeps) -> None:
        with pytest.raises(ToolError, match="does not exist"):
            do_read_file(deps, "nope.txt")

    def test_traversal_outside_workspace_blocked(self, deps: EljaDeps) -> None:
        with pytest.raises(ToolError, match="outside the workspace"):
            do_read_file(deps, "../../etc/passwd")

    def test_absolute_path_outside_workspace_blocked(self, deps: EljaDeps) -> None:
        with pytest.raises(ToolError, match="outside the workspace"):
            do_write_file(deps, "/etc/evil.txt", "nope")

    def test_absolute_path_inside_workspace_allowed(self, deps: EljaDeps) -> None:
        target = deps.workspace / "ok.txt"
        do_write_file(deps, str(target), "fine")
        assert target.read_text() == "fine"

    def test_symlink_escape_blocked(self, deps: EljaDeps) -> None:
        (deps.workspace / "esc").symlink_to("/etc")
        with pytest.raises(ToolError, match="resolves outside the workspace"):
            do_list_dir(deps, "esc")

    def test_nul_byte_path_is_tool_error(self, deps: EljaDeps) -> None:
        with pytest.raises(ToolError, match="invalid path"):
            do_read_file(deps, "bad\x00name")


class TestFailuresBecomeToolErrors:
    """Ordinary filesystem failures must surface to the model, not kill the run."""

    def test_binary_file_read(self, deps: EljaDeps) -> None:
        (deps.workspace / "img.png").write_bytes(b"\x89PNG\x00\xff\xfe")
        with pytest.raises(ToolError, match="not utf-8"):
            do_read_file(deps, "img.png")

    def test_write_under_existing_file(self, deps: EljaDeps) -> None:
        do_write_file(deps, "blocker.txt", "x")
        with pytest.raises(ToolError, match="cannot write"):
            do_write_file(deps, "blocker.txt/child.txt", "y")

    def test_write_to_directory_path(self, deps: EljaDeps) -> None:
        (deps.workspace / "adir").mkdir()
        with pytest.raises(ToolError, match="cannot write"):
            do_write_file(deps, "adir", "y")

    def test_list_dir_with_broken_symlink(self, deps: EljaDeps) -> None:
        (deps.workspace / "gone-link").symlink_to(deps.workspace / "no-such-target")
        listing = do_list_dir(deps, ".")
        assert "gone-link (broken link)" in listing

    def test_list_dir_on_file(self, deps: EljaDeps) -> None:
        do_write_file(deps, "f.txt", "x")
        with pytest.raises(ToolError, match="is not a directory"):
            do_list_dir(deps, "f.txt")

    def test_run_shell_missing_workspace(self, tmp_path: Path) -> None:
        settings = EljaSettings(workspace=WorkspaceConfig(root=tmp_path / "gone"))
        deps = EljaDeps.from_settings(settings)
        with pytest.raises(ToolError, match="cannot run command"):
            do_run_shell(deps, "echo hi")

    def test_unreadable_file_is_tool_error(self, deps: EljaDeps) -> None:
        target = deps.workspace / "locked.txt"
        target.write_text("secret")
        target.chmod(0o000)
        try:
            with pytest.raises(ToolError, match="cannot read"):
                do_read_file(deps, "locked.txt")
        finally:
            target.chmod(0o644)

    def test_run_shell_error_surfaces_as_retry_via_agent(self, tmp_path: Path) -> None:
        settings = EljaSettings(workspace=WorkspaceConfig(root=tmp_path / "gone"))
        deps = EljaDeps.from_settings(settings)
        calls: list[int] = []

        def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append(1)
            if len(calls) == 1:
                return ModelResponse(
                    parts=[ToolCallPart(tool_name="run_shell", args={"command": "echo hi"})]
                )
            return ModelResponse(parts=[TextPart(content="noted")])

        agent = Agent(
            FunctionModel(script),
            deps_type=EljaDeps,
            toolsets=[build_toolset(settings)],
        )
        result = agent.run_sync("try", deps=deps)
        assert result.output == "noted"


class TestListDir:
    def test_lists_entries_with_kind(self, deps: EljaDeps) -> None:
        do_write_file(deps, "a.txt", "x")
        (deps.workspace / "subdir").mkdir()
        listing = do_list_dir(deps, ".")
        assert "a.txt" in listing
        assert "subdir/" in listing

    def test_empty_dir(self, deps: EljaDeps) -> None:
        assert "empty" in do_list_dir(deps, ".")

    def test_missing_dir_raises(self, deps: EljaDeps) -> None:
        with pytest.raises(ToolError, match="does not exist"):
            do_list_dir(deps, "ghost")


class TestRunShell:
    def test_captures_stdout_and_exit_code(self, deps: EljaDeps) -> None:
        out = do_run_shell(deps, "echo hello")
        assert "hello" in out
        assert "exit code: 0" in out

    def test_captures_stderr_and_nonzero_exit(self, deps: EljaDeps) -> None:
        out = do_run_shell(deps, "echo oops >&2; exit 3")
        assert "oops" in out
        assert "exit code: 3" in out

    def test_runs_in_workspace_cwd(self, deps: EljaDeps) -> None:
        out = do_run_shell(deps, "pwd")
        assert str(deps.workspace.resolve()) in out

    def test_timeout_reported_not_raised(self, tmp_path: Path) -> None:
        settings = EljaSettings(
            workspace=WorkspaceConfig(root=tmp_path, shell_timeout_seconds=0.2)
        )
        deps = EljaDeps.from_settings(settings)
        out = do_run_shell(deps, "sleep 5")
        assert "timed out" in out

    def test_timeout_preserves_partial_output(self, tmp_path: Path) -> None:
        settings = EljaSettings(
            workspace=WorkspaceConfig(root=tmp_path, shell_timeout_seconds=0.3)
        )
        deps = EljaDeps.from_settings(settings)
        out = do_run_shell(deps, "echo partial-data; exec sleep 5")
        assert "timed out" in out
        assert "partial-data" in out

    def test_timeout_kills_process_tree(self, tmp_path: Path) -> None:
        import time

        settings = EljaSettings(
            workspace=WorkspaceConfig(root=tmp_path, shell_timeout_seconds=0.2)
        )
        deps = EljaDeps.from_settings(settings)
        do_run_shell(deps, "sh -c 'sleep 1; touch grandchild-marker'")
        time.sleep(1.3)
        assert not (tmp_path / "grandchild-marker").exists()

    def test_binary_output_does_not_crash(self, deps: EljaDeps) -> None:
        out = do_run_shell(deps, "head -c 16 /dev/urandom")
        assert "exit code: 0" in out

    def test_stderr_labeled_when_both_streams(self, deps: EljaDeps) -> None:
        out = do_run_shell(deps, "echo out; echo err >&2")
        assert "out" in out
        assert "stderr:\nerr" in out


class TestOutputCapping:
    def test_long_output_truncated_and_spilled(self, tmp_path: Path) -> None:
        settings = EljaSettings(
            workspace=WorkspaceConfig(root=tmp_path, max_tool_output_chars=100)
        )
        deps = EljaDeps.from_settings(settings)
        do_write_file(deps, "big.txt", "x" * 5000)
        result = do_read_file(deps, "big.txt")
        assert len(result) < 5000
        assert "truncated" in result
        # The full output is preserved on disk at the path named in the notice.
        spill_files = list(deps.spill_dir.glob("*.txt"))
        assert len(spill_files) == 1
        assert spill_files[0].read_text() == "x" * 5000
        assert str(spill_files[0]) in result

    def test_short_output_untouched(self, deps: EljaDeps) -> None:
        do_write_file(deps, "small.txt", "tiny")
        assert do_read_file(deps, "small.txt") == "tiny"

    def test_exact_boundary_untouched(self, tmp_path: Path) -> None:
        settings = EljaSettings(workspace=WorkspaceConfig(root=tmp_path, max_tool_output_chars=10))
        deps = EljaDeps.from_settings(settings)
        do_write_file(deps, "b.txt", "x" * 10)
        assert do_read_file(deps, "b.txt") == "x" * 10

    def test_exit_code_survives_truncation(self, tmp_path: Path) -> None:
        settings = EljaSettings(workspace=WorkspaceConfig(root=tmp_path, max_tool_output_chars=50))
        deps = EljaDeps.from_settings(settings)
        out = do_run_shell(deps, "yes fill | head -200; exit 7")
        assert "truncated" in out
        assert out.strip().endswith("exit code: 7")


class TestBuildToolset:
    def test_all_tools_registered_by_default(self, tmp_path: Path) -> None:
        settings = EljaSettings(workspace=WorkspaceConfig(root=tmp_path))
        toolset = build_toolset(settings)
        names = set(toolset.tools)
        assert names == {"read_file", "write_file", "list_dir", "run_shell"}

    def test_toggles_disable_tools(self, tmp_path: Path) -> None:
        settings = EljaSettings(
            workspace=WorkspaceConfig(root=tmp_path),
            tools=ToolsConfig(run_shell=False, write_file=False),
        )
        toolset = build_toolset(settings)
        assert set(toolset.tools) == {"read_file", "list_dir"}

    def test_max_retries_from_settings(self, tmp_path: Path) -> None:
        settings = EljaSettings(workspace=WorkspaceConfig(root=tmp_path))
        assert build_toolset(settings).max_retries == 3


class TestToolsViaAgent:
    """End-to-end through the agent loop with a scripted FunctionModel."""

    def test_tool_error_is_surfaced_to_model_for_retry(self, tmp_path: Path) -> None:
        settings = EljaSettings(workspace=WorkspaceConfig(root=tmp_path))
        deps = EljaDeps.from_settings(settings)
        seen: list[str] = []

        def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) == 1:
                return ModelResponse(
                    parts=[ToolCallPart(tool_name="read_file", args={"path": "missing.txt"})]
                )
            # Second call: the tool error should have come back as a retry prompt.
            last = messages[-1]
            seen.append(str(last))
            return ModelResponse(parts=[TextPart(content="done")])

        agent = Agent(
            FunctionModel(script),
            deps_type=EljaDeps,
            toolsets=[build_toolset(settings)],
        )
        result = agent.run_sync("read missing.txt", deps=deps)
        assert result.output == "done"
        assert any("does not exist" in s for s in seen)

    def test_two_consecutive_same_tool_failures_survive(self, tmp_path: Path) -> None:
        """Default max_retries must tolerate a fumbling local model (review blocker)."""
        settings = EljaSettings(workspace=WorkspaceConfig(root=tmp_path))
        deps = EljaDeps.from_settings(settings)
        calls: list[int] = []

        def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append(1)
            if len(calls) == 1:
                return ModelResponse(
                    parts=[ToolCallPart(tool_name="read_file", args={"path": "missing.txt"})]
                )
            if len(calls) == 2:
                return ModelResponse(
                    parts=[ToolCallPart(tool_name="read_file", args={"path": "missing2.txt"})]
                )
            return ModelResponse(parts=[TextPart(content="recovered")])

        agent = Agent(
            FunctionModel(script),
            deps_type=EljaDeps,
            toolsets=[build_toolset(settings)],
        )
        result = agent.run_sync("read stuff", deps=deps)
        assert result.output == "recovered"

    def test_write_and_list_errors_surface_as_retries(self, tmp_path: Path) -> None:
        settings = EljaSettings(workspace=WorkspaceConfig(root=tmp_path))
        deps = EljaDeps.from_settings(settings)

        calls: list[int] = []

        def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append(1)
            if len(calls) == 1:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="write_file",
                            args={"path": "/etc/evil.txt", "content": "x"},
                        )
                    ]
                )
            if len(calls) == 2:
                return ModelResponse(
                    parts=[ToolCallPart(tool_name="list_dir", args={"path": "ghost"})]
                )
            if len(calls) == 3:
                return ModelResponse(
                    parts=[ToolCallPart(tool_name="run_shell", args={"command": "echo ran"})]
                )
            return ModelResponse(parts=[TextPart(content="gave up")])

        agent = Agent(
            FunctionModel(script),
            deps_type=EljaDeps,
            toolsets=[build_toolset(settings)],
        )
        result = agent.run_sync("try bad paths", deps=deps)
        assert result.output == "gave up"

    def test_successful_tool_call_roundtrip(self, tmp_path: Path) -> None:
        settings = EljaSettings(workspace=WorkspaceConfig(root=tmp_path))
        deps = EljaDeps.from_settings(settings)
        (tmp_path / "data.txt").write_text("payload-42")

        def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) == 1:
                return ModelResponse(
                    parts=[ToolCallPart(tool_name="read_file", args={"path": "data.txt"})]
                )
            return ModelResponse(parts=[TextPart(content="finished")])

        agent = Agent(
            FunctionModel(script),
            deps_type=EljaDeps,
            toolsets=[build_toolset(settings)],
        )
        result = agent.run_sync("read data.txt", deps=deps)
        assert result.output == "finished"
