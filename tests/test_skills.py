"""Tests for elja.skills."""

from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from elja.settings import EljaSettings, SkillsConfig, WorkspaceConfig
from elja.skills import SkillError, load_skills
from tests.conftest import narrow_terminal


def write_skill(path: Path, skill_id: str = "greeting", body: str = "Always say hello.") -> None:
    path.write_text(
        f"""---
id: {skill_id}
description: How to greet people.
---
{body}
"""
    )


@pytest.fixture
def settings(tmp_path: Path) -> EljaSettings:
    return EljaSettings(workspace=WorkspaceConfig(root=tmp_path))


class TestLoadSkills:
    def test_no_skills_dir_is_fine(self, settings: EljaSettings) -> None:
        assert load_skills(settings) == []

    def test_loads_deferred_capabilities(self, settings: EljaSettings, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        write_skill(skills_dir / "greeting.md")
        write_skill(skills_dir / "farewell.md", skill_id="farewell", body="Wave goodbye.")
        skills = load_skills(settings)
        assert sorted(str(s.id) for s in skills) == ["farewell", "greeting"]
        greeting = next(s for s in skills if s.id == "greeting")
        assert greeting.description == "How to greet people."
        assert greeting.defer_loading is True

    def test_body_becomes_instructions(self, settings: EljaSettings, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        write_skill(skills_dir / "greeting.md", body="Line one.\n\nLine two.")
        (skill,) = load_skills(settings)
        assert skill.get_instructions() == ["Line one.\n\nLine two."]

    def test_missing_frontmatter_raises_with_filename(
        self, settings: EljaSettings, tmp_path: Path
    ) -> None:
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        (skills_dir / "bad.md").write_text("just a plain markdown file")
        with pytest.raises(SkillError, match="bad.md"):
            load_skills(settings)

    def test_missing_required_key_raises(self, settings: EljaSettings, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        (skills_dir / "noid.md").write_text("---\ndescription: d\n---\nbody\n")
        with pytest.raises(SkillError, match="noid.md"):
            load_skills(settings)

    def test_unterminated_frontmatter_raises(self, settings: EljaSettings, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        (skills_dir / "open.md").write_text("---\nid: x\ndescription: d\nno closing fence")
        with pytest.raises(SkillError, match="open.md"):
            load_skills(settings)

    def test_invalid_yaml_raises(self, settings: EljaSettings, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        (skills_dir / "badyaml.md").write_text("---\nid: [unclosed\n---\nbody\n")
        with pytest.raises(SkillError, match="badyaml.md"):
            load_skills(settings)

    def test_custom_dir_and_absolute_dir(self, tmp_path: Path) -> None:
        custom = tmp_path / "elsewhere"
        custom.mkdir()
        write_skill(custom / "s.md", skill_id="s")
        settings = EljaSettings(
            workspace=WorkspaceConfig(root=tmp_path),
            skills=SkillsConfig(dir=custom),
        )
        assert [s.id for s in load_skills(settings)] == ["s"]

    def test_non_md_files_ignored(self, settings: EljaSettings, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        (skills_dir / "notes.txt").write_text("not a skill")
        assert load_skills(settings) == []

    def test_dotfiles_skipped(self, settings: EljaSettings, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        (skills_dir / "._junk.md").write_bytes(b"\x00\x05binary AppleDouble")
        assert load_skills(settings) == []

    def test_duplicate_ids_name_both_files(self, settings: EljaSettings, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        write_skill(skills_dir / "a.md", skill_id="greet")
        write_skill(skills_dir / "b.md", skill_id="greet")
        with pytest.raises(SkillError) as exc_info:
            load_skills(settings)
        assert "a.md" in str(exc_info.value)
        assert "b.md" in str(exc_info.value)

    def test_invalid_id_is_skill_error(self, settings: EljaSettings, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        (skills_dir / "colon.md").write_text("---\nid: my:skill\ndescription: d\n---\nbody\n")
        with pytest.raises(SkillError, match="colon.md"):
            load_skills(settings)

    def test_underscore_prefixed_id_reserved(self, settings: EljaSettings, tmp_path: Path) -> None:
        """_-prefixed ids are reserved for elja internals (permission gate etc.)."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        (skills_dir / "sneaky.md").write_text(
            "---\nid: _elja_permission_gate\ndescription: d\n---\nbody\n"
        )
        with pytest.raises(SkillError, match="sneaky.md"):
            load_skills(settings)

    def test_non_string_id_is_skill_error(self, settings: EljaSettings, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        (skills_dir / "nullid.md").write_text("---\nid: null\ndescription: d\n---\nbody\n")
        with pytest.raises(SkillError, match="nullid.md"):
            load_skills(settings)

    def test_non_mapping_frontmatter_raises(self, settings: EljaSettings, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        (skills_dir / "list.md").write_text("---\n- just\n- a list\n---\nbody\n")
        with pytest.raises(SkillError, match="list.md"):
            load_skills(settings)

    def test_empty_description_raises(self, settings: EljaSettings, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        (skills_dir / "blank.md").write_text("---\nid: blank\ndescription: '  '\n---\nbody\n")
        with pytest.raises(SkillError, match="blank.md"):
            load_skills(settings)

    def test_dashes_inside_frontmatter_value_survive(
        self, settings: EljaSettings, tmp_path: Path
    ) -> None:
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        (skills_dir / "hr.md").write_text(
            "---\nid: hr\ndescription: alpha --- beta\n---\nbody\n---\nmore body\n"
        )
        (skill,) = load_skills(settings)
        assert skill.description == "alpha --- beta"
        assert skill.get_instructions() == ["body\n---\nmore body"]

    def test_bom_and_crlf_accepted(self, settings: EljaSettings, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        (skills_dir / "bom.md").write_bytes(
            b"\xef\xbb\xbf---\r\nid: bom\r\ndescription: d\r\n---\r\nbody\r\n"
        )
        (skill,) = load_skills(settings)
        assert str(skill.id) == "bom"

    def test_indented_body_preserved(self, settings: EljaSettings, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        (skills_dir / "code.md").write_text(
            "---\nid: code\ndescription: d\n---\n    indented code\nplain\n"
        )
        (skill,) = load_skills(settings)
        assert skill.get_instructions() == ["    indented code\nplain"]


class TestAgentIntegration:
    def test_skill_catalog_reaches_model(self, tmp_path: Path) -> None:
        """Deferred skills appear as a catalog + load_capability tool, costing ~no context."""
        from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
        from pydantic_ai.models.function import AgentInfo, FunctionModel

        from elja.agent import build_agent
        from elja.deps import EljaDeps

        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        write_skill(skills_dir / "greeting.md")
        settings = EljaSettings(workspace=WorkspaceConfig(root=tmp_path))
        seen: dict[str, object] = {}

        def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen["tools"] = [t.name for t in info.function_tools]
            seen["instructions"] = info.instructions or ""
            return ModelResponse(parts=[TextPart(content="ok")])

        agent = build_agent(settings)
        with agent.override(model=FunctionModel(script)):
            agent.run_sync("hello", deps=EljaDeps.from_settings(settings))
        assert "load_capability" in seen["tools"]  # type: ignore[operator]
        assert "greeting: How to greet people." in seen["instructions"]  # type: ignore[operator]
        # The skill BODY is not in context until loaded.
        assert "Always say hello." not in seen["instructions"]  # type: ignore[operator]

    def test_load_capability_roundtrip_delivers_body(self, tmp_path: Path) -> None:
        """The core promise: loading a skill hands the model the instructions body."""
        from pydantic_ai.messages import (
            ModelMessage,
            ModelResponse,
            TextPart,
            ToolCallPart,
        )
        from pydantic_ai.models.function import AgentInfo, FunctionModel

        from elja.agent import build_agent
        from elja.deps import EljaDeps

        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        write_skill(skills_dir / "greeting.md", body="SECRET-HANDSHAKE-INSTRUCTIONS")
        settings = EljaSettings(workspace=WorkspaceConfig(root=tmp_path))
        delivered: list[str] = []

        def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) == 1:
                return ModelResponse(
                    parts=[ToolCallPart(tool_name="load_capability", args={"id": "greeting"})]
                )
            delivered.append(str(messages[-1]))
            return ModelResponse(parts=[TextPart(content="loaded")])

        agent = build_agent(settings)
        with agent.override(model=FunctionModel(script)):
            result = agent.run_sync("greet me", deps=EljaDeps.from_settings(settings))
        assert result.output == "loaded"
        assert any("SECRET-HANDSHAKE-INSTRUCTIONS" in d for d in delivered)


class TestReplSkillErrors:
    async def test_startup_skill_error_is_clean(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], mocker: MockerFixture
    ) -> None:
        from elja.cli import repl

        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        (skills_dir / "broken.md").write_text("no frontmatter at all")
        narrow_terminal(mocker, str(skills_dir / "broken.md"))
        settings = EljaSettings(workspace=WorkspaceConfig(root=tmp_path))
        await repl(settings, "s", once="hi")  # must not raise
        out = capsys.readouterr().out
        assert "cannot start agent" in out
        # The WHOLE path, not just the filename. Asserting "broke\nn.md" not in
        # out can never fail: it is only false when "broken.md" in out is already
        # false, and pytest stops at the first failure. And "broken.md" alone is
        # tuned to one path length — the original flake was $TMPDIR, not terminal
        # width, so a short TMPDIR moved rich's break offset into the filename
        # while a normal one left the bug live and the assertion passing.
        assert str(skills_dir / "broken.md") in out

    async def test_a_midsession_skill_break_names_the_whole_path(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], mocker: MockerFixture
    ) -> None:
        """The likelier scenario: you edit a skill while the REPL is running.

        This goes through the agent-rebuild warning, a different door from the
        startup error, and one the first version of this fix missed.
        """
        from elja.cli import repl

        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        good = "---\nid: greeting\ndescription: says hello\n---\nsay hello\n"
        (skills_dir / "greeting.md").write_text(good)
        mocker.patch.dict("os.environ", {"FORCE_COLOR": "1"})
        narrow_terminal(mocker, str(skills_dir / "greeting.md"))
        settings = EljaSettings(workspace=WorkspaceConfig(root=tmp_path))

        calls: list[int] = []

        def break_the_skill_then_fail(*args: object, **kwargs: object) -> None:
            calls.append(1)
            (skills_dir / "greeting.md").write_text("no frontmatter at all")
            raise RuntimeError("turn blew up")

        mocker.patch("elja.cli.run_turn", side_effect=break_the_skill_then_fail)
        await repl(settings, "s", once="hi")
        out = capsys.readouterr().out
        assert calls, "the turn never ran"
        assert "agent rebuild failed" in out
        assert str(skills_dir / "greeting.md") in out
        # Yellow, not red: the REPL kept going on the current agent. This was the
        # seventh and last unasserted style literal — in the same message pair whose
        # TEXT is asserted two lines up, which is how the colour slipped through.
        warning = next(line for line in out.splitlines() if "agent rebuild failed" in line)
        assert "\x1b[33m" in warning

    async def test_a_diagnostic_does_not_interpret_its_own_filename(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], mocker: MockerFixture
    ) -> None:
        """markup and emoji in a FILENAME must survive to the reader.

        ``[bold]`` would be eaten and applied as styling; ``:100:`` would become
        an emoji, naming a file that does not exist.
        """
        from elja.cli import repl

        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        name = "[bold]notes:100:.md"
        (skills_dir / name).write_text("no frontmatter at all")
        narrow_terminal(mocker, str(skills_dir / name))
        settings = EljaSettings(workspace=WorkspaceConfig(root=tmp_path))
        await repl(settings, "s", once="hi")
        out = capsys.readouterr().out
        assert str(skills_dir / name) in out
        assert "💯" not in out

    async def test_rebuild_failure_keeps_current_agent(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        mocker: MockerFixture,
    ) -> None:
        """A skill file broken mid-session must not kill the REPL on rebuild."""
        from collections.abc import AsyncIterator

        from pydantic_ai import Agent
        from pydantic_ai.messages import ModelMessage
        from pydantic_ai.models.function import AgentInfo, FunctionModel

        from elja.cli import repl
        from elja.deps import EljaDeps

        settings = EljaSettings(workspace=WorkspaceConfig(root=tmp_path))
        calls: list[int] = []

        async def sf(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
            calls.append(1)
            yield "x"
            raise ConnectionError("flaky")

        good: Agent[EljaDeps, str] = Agent(FunctionModel(stream_function=sf), deps_type=EljaDeps)

        def build(s: EljaSettings, mcp_toolsets: object = None) -> Agent[EljaDeps, str]:
            if not calls:  # initial build succeeds
                return good
            raise SkillError("invalid skill file broken.md: boom")  # rebuild fails

        mocker.patch("elja.cli.build_agent", side_effect=build)
        prompts = iter(["boom", "exit"])
        await repl(settings, "s", input_fn=lambda _: next(prompts))
        out = capsys.readouterr().out
        assert "agent rebuild failed" in out
        assert "broken.md" in out
