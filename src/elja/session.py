"""Conversation persistence.

Pydantic AI has no built-in checkpointing; elja persists the full typed
message history as JSON so conversations survive process restarts. History is
saved verbatim — messages must round-trip untouched so provider-specific
state survives.
"""

import re
from pathlib import Path

from pydantic import ValidationError
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter

from elja.settings import EljaSettings

_NAME_RE = re.compile(r"^[\w.-]+$")


class SessionLoadError(Exception):
    """A session file exists but can't be parsed (corrupt, or written by an
    incompatible pydantic-ai version). Deleting or restoring the file named in
    the message is the remedy."""


class Session:
    """A named conversation stored as a JSON file."""

    def __init__(self, path: Path) -> None:
        """Create a session backed by ``path`` (need not exist yet)."""
        self.path = path

    @classmethod
    def for_name(cls, settings: EljaSettings, name: str) -> "Session":
        """The session called ``name`` under the configured session directory.

        Args:
            settings: Resolved elja settings.
            name: Session name; becomes the filename stem. Restricted to
                word characters, dots, and dashes — no path separators.

        Returns:
            A session at ``<session dir>/<name>.json``, where a relative
            session dir is anchored at the workspace root.

        Raises:
            ValueError: If ``name`` is empty or contains path separators
                or other non-slug characters.
        """
        if not _NAME_RE.match(name) or ".." in name:
            raise ValueError(f"invalid session name {name!r} (use letters/digits/._-)")
        base = settings.session.dir
        if not base.is_absolute():
            base = settings.workspace.root.resolve() / base
        return cls(base / f"{name}.json")

    def load(self) -> list[ModelMessage]:
        """Load the stored history, or an empty list if none exists yet.

        Raises:
            SessionLoadError: If the file exists but can't be parsed.
        """
        if not self.path.is_file():
            return []
        try:
            return ModelMessagesTypeAdapter.validate_json(self.path.read_bytes())
        except ValidationError as exc:
            raise SessionLoadError(f"corrupt or incompatible session file: {self.path}") from exc

    def save(self, messages: list[ModelMessage]) -> None:
        """Persist the full history, replacing any previous contents."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_bytes(ModelMessagesTypeAdapter.dump_json(messages))
        tmp.replace(self.path)
