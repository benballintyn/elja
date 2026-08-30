"""Tests for elja.skills."""

from pathlib import Path

import pytest

from elja.settings import EljaSettings, SkillsConfig, WorkspaceConfig
from elja.skills import SkillError, load_skills


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
