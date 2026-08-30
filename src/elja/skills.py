"""Skills: markdown files that become on-demand capabilities.

A skill is a ``.md`` file with YAML frontmatter (``id``, ``description``)
followed by the instructions body. Skills load deferred: the model sees only
the id + description in its capability catalog and pulls in the full
instructions when it decides the skill is relevant — so a large skill library
costs almost no context until used.
"""

import yaml
from pydantic_ai.capabilities import Capability

from elja.settings import EljaSettings


class SkillError(Exception):
    """A skill file is malformed; the message names the file."""


def _parse_skill(text: str) -> tuple[str, str, str]:
    """Split a skill file into (id, description, body)."""
    if not text.startswith("---"):
        raise ValueError("missing YAML frontmatter (expected leading '---')")
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise ValueError("unterminated YAML frontmatter")
    meta = yaml.safe_load(parts[1])
    if not isinstance(meta, dict) or "id" not in meta or "description" not in meta:
        raise ValueError("frontmatter must define 'id' and 'description'")
    return str(meta["id"]), str(meta["description"]), parts[2].strip()


def load_skills(settings: EljaSettings) -> list[Capability]:
    """Load every ``*.md`` skill under the configured skills directory.

    Args:
        settings: Resolved elja settings. A relative skills dir is anchored at
            the workspace root; a missing directory means no skills.

    Returns:
        One deferred capability per skill file, sorted by filename.

    Raises:
        SkillError: If any skill file is malformed (named in the message).
    """
    base = settings.skills.dir
    if not base.is_absolute():
        base = settings.workspace.root.resolve() / base
    if not base.is_dir():
        return []
    skills: list[Capability] = []
    for path in sorted(base.glob("*.md")):
        try:
            skill_id, description, body = _parse_skill(path.read_text(encoding="utf-8"))
        except (ValueError, yaml.YAMLError) as exc:
            raise SkillError(f"invalid skill file {path}: {exc}") from exc
        skills.append(
            Capability(
                id=skill_id,
                description=description,
                instructions=body,
                defer_loading=True,
            )
        )
    return skills
