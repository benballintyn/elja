"""Shared test fixtures."""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from pytest_mock import MockerFixture


@pytest.fixture(autouse=True)
def _hermetic(tmp_path: Path, mocker: MockerFixture) -> Iterator[None]:
    """Isolate every test from the developer's environment.

    Settings read ``ELJA_*`` env vars and ``./elja.toml``, and the model
    factory reads provider API keys; without this, a stray env var or a config
    file in the repo root would change test results.
    """
    provider_keys = {"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"}
    clean = {
        k: v for k, v in os.environ.items() if not k.startswith("ELJA_") and k not in provider_keys
    }
    mocker.patch.dict("os.environ", clean, clear=True)
    original = Path.cwd()
    os.chdir(tmp_path)
    try:
        yield
    finally:
        os.chdir(original)
