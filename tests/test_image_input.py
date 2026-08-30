"""Tests for image attachments in the CLI."""

import struct
import zlib
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pytest_mock import MockerFixture

from elja.cli import attach_image, main, repl, run_turn
from elja.deps import EljaDeps
from elja.session import Session
from elja.settings import EljaSettings, WorkspaceConfig


def make_png(path: Path, rgb: tuple[int, int, int] = (255, 0, 0), size: int = 8) -> None:
    """Write a tiny solid-color PNG using only the stdlib."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    raw = b"".join(b"\x00" + bytes(rgb) * size for _ in range(size))
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


@pytest.fixture
def settings(tmp_path: Path) -> EljaSettings:
    return EljaSettings(workspace=WorkspaceConfig(root=tmp_path))


def _capture_agent(seen: list[object]) -> Agent[EljaDeps, str]:
    async def sf(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        first = messages[0]
        assert isinstance(first, ModelRequest)
        for part in first.parts:
            if isinstance(part, UserPromptPart):
                seen.append(part.content)
        yield "saw it"

    return Agent(FunctionModel(stream_function=sf), deps_type=EljaDeps)


class TestAttachImage:
    def test_builds_binary_content(self, tmp_path: Path) -> None:
        img = tmp_path / "pic.png"
        make_png(img)
        prompt = attach_image("what is this?", img)
        assert prompt[0] == "what is this?"
        content = prompt[1]
        assert isinstance(content, BinaryContent)
        assert content.media_type == "image/png"
        assert content.data == img.read_bytes()

    def test_jpeg_detected_by_magic_bytes_despite_extension(self, tmp_path: Path) -> None:
        img = tmp_path / "photo.jfif"  # extension mimetypes can't map
        img.write_bytes(b"\xff\xd8\xff\xe0 fake")
        content = attach_image("x", img)[1]
        assert isinstance(content, BinaryContent)
        assert content.media_type == "image/jpeg"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not found"):
            attach_image("x", tmp_path / "nope.png")

    def test_non_image_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "notes.txt"
        f.write_text("hi")
        with pytest.raises(ValueError, match="not a supported image"):
            attach_image("x", f)

    def test_renamed_html_rejected(self, tmp_path: Path) -> None:
        fake = tmp_path / "page.png"
        fake.write_text("<html><body>gotcha</body></html>")
        with pytest.raises(ValueError, match="not a supported image"):
            attach_image("x", fake)

    def test_riff_without_webp_tag_rejected(self, tmp_path: Path) -> None:
        fake = tmp_path / "clip.webp"
        fake.write_bytes(b"RIFF\x00\x00\x00\x00WAVE fake audio")
        with pytest.raises(ValueError, match="not a supported image"):
            attach_image("x", fake)

    def test_oversized_image_rejected(self, tmp_path: Path) -> None:
        from elja.cli import MAX_IMAGE_BYTES

        big = tmp_path / "huge.png"
        with big.open("wb") as f:
            f.seek(MAX_IMAGE_BYTES)
            f.write(b"x")
        with pytest.raises(ValueError, match="too large"):
            attach_image("x", big)


class TestRunTurnMultimodal:
    async def test_image_reaches_model(self, settings: EljaSettings, tmp_path: Path) -> None:
        img = tmp_path / "pic.png"
        make_png(img)
        seen: list[object] = []
        session = Session.for_name(settings, "img")
        output = await run_turn(
            _capture_agent(seen),
            settings,
            session,
            attach_image("describe", img),
            lambda d: None,
        )
        assert output == "saw it"
        content = seen[0]
        assert isinstance(content, list)
        assert content[0] == "describe"
        assert isinstance(content[1], BinaryContent)
        # The multimodal turn persists and reloads.
        assert len(session.load()) == 2
        assert session.load() == session.load()


class TestReplImageCommand:
    async def test_img_command_attaches(
        self, settings: EljaSettings, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        img = tmp_path / "pic.png"
        make_png(img)
        seen: list[object] = []
        mocker.patch("elja.cli.build_agent", return_value=_capture_agent(seen))
        prompts = iter([f"/img {img} what color?", "exit"])
        await repl(settings, "s", input_fn=lambda _: next(prompts))
        assert isinstance(seen[0], list)
        assert seen[0][0] == "what color?"
        assert isinstance(seen[0][1], BinaryContent)

    async def test_img_command_missing_file_reports(
        self,
        settings: EljaSettings,
        mocker: MockerFixture,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        seen: list[object] = []
        mocker.patch("elja.cli.build_agent", return_value=_capture_agent(seen))
        prompts = iter(["/img /no/such.png describe", "exit"])
        await repl(settings, "s", input_fn=lambda _: next(prompts))
        assert "not found" in capsys.readouterr().out
        assert seen == []

    async def test_img_command_without_prompt_reports_usage(
        self,
        settings: EljaSettings,
        mocker: MockerFixture,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        seen: list[object] = []
        mocker.patch("elja.cli.build_agent", return_value=_capture_agent(seen))
        prompts = iter(["/img", "exit"])
        await repl(settings, "s", input_fn=lambda _: next(prompts))
        assert "usage" in capsys.readouterr().out.lower()
        assert seen == []


class TestReplImageEdgeCases:
    async def test_quoted_path_with_spaces(
        self, settings: EljaSettings, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        img = tmp_path / "Screenshot 2026-08-30 at noon.png"
        make_png(img)
        seen: list[object] = []
        mocker.patch("elja.cli.build_agent", return_value=_capture_agent(seen))
        prompts = iter([f'/img "{img}" what is it?', "exit"])
        await repl(settings, "s", input_fn=lambda _: next(prompts))
        assert isinstance(seen[0], list)
        assert seen[0][0] == "what is it?"

    async def test_unclosed_quote_shows_usage(
        self,
        settings: EljaSettings,
        mocker: MockerFixture,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        seen: list[object] = []
        mocker.patch("elja.cli.build_agent", return_value=_capture_agent(seen))
        prompts = iter(['/img "unclosed quote.png describe', "exit"])
        await repl(settings, "s", input_fn=lambda _: next(prompts))
        assert "usage" in capsys.readouterr().out.lower()
        assert seen == []

    async def test_img_prefix_word_is_not_the_command(
        self, settings: EljaSettings, mocker: MockerFixture
    ) -> None:
        seen: list[object] = []
        mocker.patch("elja.cli.build_agent", return_value=_capture_agent(seen))
        prompts = iter(["/imgfoo is not a command", "exit"])
        await repl(settings, "s", input_fn=lambda _: next(prompts))
        assert seen == ["/imgfoo is not a command"]


def test_main_image_flag(tmp_path: Path, mocker: MockerFixture) -> None:
    img = tmp_path / "pic.png"
    make_png(img)
    config = tmp_path / "elja.toml"
    config.write_text(f'[workspace]\nroot = "{tmp_path}"\n')
    seen: list[object] = []
    mocker.patch("elja.cli.build_agent", return_value=_capture_agent(seen))
    mocker.patch(
        "sys.argv",
        [
            "elja",
            "chat",
            "--config",
            str(config),
            "--once",
            "look",
            "--image",
            str(img),
        ],
    )
    main()
    assert isinstance(seen[0], list)
    assert isinstance(seen[0][1], BinaryContent)


def test_once_with_bad_image_reports_cleanly(
    tmp_path: Path, mocker: MockerFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "elja.toml"
    config.write_text(f'[workspace]\nroot = "{tmp_path}"\n')
    mocker.patch("elja.cli.build_agent", return_value=_capture_agent([]))
    mocker.patch(
        "sys.argv",
        ["elja", "chat", "--config", str(config), "--once", "x", "--image", "/no/such.png"],
    )
    main()  # must not raise
    assert "not found" in capsys.readouterr().out


def test_image_without_once_is_an_error(mocker: MockerFixture) -> None:
    mocker.patch("sys.argv", ["elja", "chat", "--image", "x.png"])
    with pytest.raises(SystemExit):
        main()


@pytest.mark.integration
async def test_live_vlm_sees_color(tmp_path: Path) -> None:
    """Qwen3.8-27B is a VLM: it should identify the color of a solid image."""
    from elja.agent import build_agent

    img = tmp_path / "red.png"
    make_png(img, rgb=(255, 0, 0), size=64)
    settings = EljaSettings(workspace=WorkspaceConfig(root=tmp_path))
    session = Session.for_name(settings, "vlm")
    output = await run_turn(
        build_agent(settings),
        settings,
        session,
        attach_image("What is the dominant color of this image? Answer with one word.", img),
        lambda d: None,
    )
    assert "red" in output.lower()
