"""Tests for package metadata."""

import elja


def test_version_is_set() -> None:
    """The package exposes a semver __version__ string."""
    major, minor, patch = elja.__version__.split(".")
    assert all(part.isdigit() for part in (major, minor, patch))
