"""Prompt construction: combines each card's character description with the shared style.

Everything that decides *what text the model sees* lives here, so future style
mechanisms (reference images, IP-Adapter, a LoRA, per-card style overrides) have
one place to plug in. ``PromptSpec`` already carries an (unused) slot for
reference images so callers don't need to change when that lands.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import Settings
from .models import Card

# CLIP text encoders (used by SD 1.x/2.x and both SDXL encoders) read at most
# 77 tokens including start/end markers. Anything past that is silently cut off.
CLIP_TOKEN_LIMIT = 77


class StyleError(ValueError):
    pass


@dataclass(frozen=True)
class PromptSpec:
    prompt: str
    negative_prompt: str
    # Reserved for future style-reference conditioning (e.g. IP-Adapter). Not used
    # by the current generator; a reference image guides style, it never enforces it.
    reference_images: tuple[Path, ...] = field(default_factory=tuple)


def load_style(path: Path) -> str:
    """Read the shared style text. Lines starting with '#' are comments."""
    if not path.exists():
        raise StyleError(f"style file not found: {path}")
    lines = [line for line in path.read_text(encoding="utf-8").splitlines()
             if not line.lstrip().startswith("#")]
    style = " ".join(" ".join(lines).split())
    if not style:
        raise StyleError(f"style file is empty: {path}")
    return style


def compose_prompt(character: str, style: str, template: str) -> str:
    """Fill the prompt template. The character comes first by default because CLIP
    truncates from the end, and the character description matters most per card."""
    text = template.format(character=character.strip(), style=style.strip())
    return re.sub(r"[ \t]+", " ", text).strip()


def build_prompt(card: Card, style: str, settings: Settings) -> PromptSpec:
    negative = card.overrides.get("negative_prompt", settings.negative_prompt)
    return PromptSpec(
        prompt=compose_prompt(card.prompt, style, settings.prompt_template),
        negative_prompt=" ".join(negative.split()),
    )


def estimate_tokens(text: str) -> int:
    """Rough CLIP token estimate without loading a tokenizer: words plus punctuation,
    plus start/end markers. The generator logs the exact count from the real tokenizer."""
    return len(re.findall(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]", text)) + 2
