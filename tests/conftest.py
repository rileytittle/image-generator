"""Shared fixtures. Tests never load a real model: generation uses FakeGenerator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from cardgen.config import build_settings
from cardgen.generator import GenerationRequest, GenerationResult, GeneratorError

STYLE = "Flat cartoon style, bold outlines. No text."

CARDS = [
    {"id": "village_archer", "name": "Village Archer", "prompt": "An archer.", "seed": 1001},
    {"id": "brave_peasant", "name": "Brave Peasant", "prompt": "A peasant.", "seed": 1002},
    {"id": "court_jester", "name": "Court Jester", "prompt": "A jester."},
]

BASE_SETTINGS = {
    "model_id": "fake/model",
    "presets": {"test": {"width": 256, "height": 384, "steps": 2, "guidance_scale": 5.0}},
    "preset": "test",
    "negative_prompt": "text, watermark",
    "style_file": "config/style.txt",
    "output_dir": "output",
}


class FakeGenerator:
    """Stands in for DiffusersGenerator. ``fail_ids`` maps a prompt substring to an exception."""

    def __init__(self, fail_on: dict[str, BaseException] | None = None, size_override=None):
        self.fail_on = fail_on or {}
        self.size_override = size_override
        self.loaded = 0
        self.closed = 0
        self.requests: list[GenerationRequest] = []

    def load(self) -> None:
        self.loaded += 1

    def describe(self) -> dict[str, Any]:
        return {"model_id": "fake/model", "device": "cpu"}

    def close(self) -> None:
        self.closed += 1

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        for needle, exc in self.fail_on.items():
            if needle in request.prompt:
                raise exc
        size = self.size_override or (request.width, request.height)
        image = Image.new("RGB", size, (request.seed % 256, 120, 60))
        image.putpixel((0, 0), (255, 255, 255))  # not blank
        return GenerationResult(image=image, info={"prompt_mode": "fake"})


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A temporary project folder (cwd) with settings, style and three cards."""
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "settings.json").write_text(json.dumps(BASE_SETTINGS))
    (tmp_path / "config" / "style.txt").write_text(STYLE)
    (tmp_path / "cards.json").write_text(json.dumps(CARDS))
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def settings(project: Path):
    return build_settings(BASE_SETTINGS)


@pytest.fixture
def fake_generator(monkeypatch: pytest.MonkeyPatch):
    """Patch the CLI's generator factory; returns the FakeGenerator instance."""
    fake = FakeGenerator()
    monkeypatch.setattr("cardgen.generator.create_generator", lambda settings: fake)
    return fake


__all__ = ["FakeGenerator", "GeneratorError", "STYLE", "CARDS", "BASE_SETTINGS"]
