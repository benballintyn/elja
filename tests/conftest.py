"""Shared test fixtures."""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from pytest_mock import MockerFixture


def narrow_terminal(mocker: MockerFixture, longer_than: str, columns: int = 40) -> None:
    """Force a terminal narrower than the text the caller is about to assert on.

    A "this path survives unbroken" assertion only constrains anything when the
    path does not fit: with a wide terminal it passes against a `soft_wrap=False`
    that is actively broken. Asserting the width equals 40 does not catch that —
    raising the constant satisfies it too — so the precondition asserted here is
    the *relation* between the two.

    Shared from `conftest` because five call sites across three modules need it,
    and the two newest were relying on `$TMPDIR`'s length by accident, which is
    the original flake this replaced.
    """
    from rich.console import Console

    mocker.patch.dict("os.environ", {"COLUMNS": str(columns)})
    width = Console().width
    assert width < len(longer_than), (
        f"terminal is {width} columns and the asserted text is {len(longer_than)}: "
        "it fits, so wrapping cannot break it and the test constrains nothing"
    )


@pytest.fixture(autouse=True)
def _hermetic(tmp_path: Path, mocker: MockerFixture) -> Iterator[None]:
    """Isolate every test from the developer's environment.

    Settings read ``ELJA_*`` env vars and ``./elja.toml``, and the model
    factory reads provider API keys; without this, a stray env var or a config
    file in the repo root would change test results.

    The terminal variables matter too, now that diagnostics are asserted by the
    colour they emit. Forcing a terminal is what exposes a test to ``TERM``:
    rich treats ``TERM`` in ``("dumb", "unknown")`` as a dumb terminal, and then
    reports no colour system at all AND short-circuits its width to 80 columns
    before it reads ``COLUMNS`` — so both the style and the narrow-terminal
    preconditions silently evaporate. Measured on a correct tree: ``TERM=dumb``,
    ``TERM=unknown`` or ``NO_COLOR=1`` each turn six passing tests red. CI's runner
    sets none of them, so this is a local-only trap, and ``AGENTS.md`` tells
    contributors to run pytest locally.
    """
    provider_keys = {"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"}
    colour_keys = {"NO_COLOR", "CLICOLOR", "CLICOLOR_FORCE", "FORCE_COLOR"}
    clean = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("ELJA_") and k not in provider_keys and k not in colour_keys
    }
    clean["TERM"] = "xterm-256color"
    mocker.patch.dict("os.environ", clean, clear=True)
    original = Path.cwd()
    os.chdir(tmp_path)
    try:
        yield
    finally:
        os.chdir(original)
