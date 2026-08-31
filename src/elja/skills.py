"""Skills: markdown files that become on-demand capabilities.

A skill is a ``.md`` file with YAML frontmatter (``id``, ``description``)
followed by the instructions body. Skills load deferred: the model sees only
the id + description in its capability catalog and pulls in the full
instructions when it decides the skill is relevant — so a large skill library
costs almost no context until used. (Descriptions ARE injected into every
request, so keep them to one line.)

Skills are re-read from disk only when the agent is (re)built — at REPL
startup and after a failed turn — so mid-session edits apply lazily.
"""

import re
from pathlib import Path

import yaml
from pydantic_ai.capabilities import Capability
from pydantic_ai.exceptions import UserError

from elja.settings import EljaSettings

# Closing fence must be its own line — '---' inside a frontmatter value or the
# body must not terminate the frontmatter block.
_FRONTMATTER_RE = re.compile(r"\A---\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)(.*)\Z", re.DOTALL)
# Keep ids typeable by a small local model (and valid capability ids). The
# leading letter also reserves _-prefixed ids for elja internals (e.g. the
# permission gate).
_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")


class SkillError(Exception):
    """A skill file is malformed; the message names the file."""


def _parse_skill(text: str) -> tuple[str, str, str]:
    """Split a skill file into (id, description, body)."""
    match = _FRONTMATTER_RE.match(text)
    if match is None:
        raise ValueError("missing or unterminated YAML frontmatter (--- fenced)")
    meta = yaml.safe_load(match.group(1))
    if not isinstance(meta, dict):
        raise ValueError("frontmatter must be a YAML mapping")
    skill_id = meta.get("id")
    description = meta.get("description")
    if not isinstance(skill_id, str) or not _ID_RE.match(skill_id):
        raise ValueError("'id' must be a string of letters, digits, '_' or '-'")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("'description' must be a non-empty string")
    return skill_id, description.strip(), match.group(2).lstrip("\n").rstrip()


def load_skills(settings: EljaSettings) -> list[Capability]:
    """Load every ``*.md`` skill under the configured skills directory.

    Args:
        settings: Resolved elja settings. A relative skills dir is anchored at
            the workspace root; a missing directory means no skills.

    Returns:
        One deferred capability per skill file, sorted by filename.

    Raises:
        SkillError: If any skill file is malformed or two files share an id
            (the offending file(s) are named in the message).
    """
    base = settings.skills.dir
    if not base.is_absolute():
        base = settings.workspace.root.resolve() / base
    if not base.is_dir():
        return []
    skills: list[Capability] = []
    seen: dict[str, Path] = {}
    for path in sorted(base.glob("*.md")):
        if path.name.startswith("."):  # editor droppings, AppleDouble files
            continue
        try:
            skill_id, description, body = _parse_skill(path.read_text(encoding="utf-8-sig"))
            if skill_id in seen:
                raise ValueError(f"duplicate skill id {skill_id!r} (also in {seen[skill_id]})")
            seen[skill_id] = path
            skills.append(
                Capability(
                    id=skill_id,
                    description=description,
                    instructions=body,
                    defer_loading=True,
                )
            )
        except (ValueError, yaml.YAMLError, UserError) as exc:
            raise SkillError(f"invalid skill file {path}: {exc}") from exc
    return skills
