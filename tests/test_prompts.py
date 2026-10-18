import pytest

from cardgen.config import build_settings
from cardgen.models import Card
from cardgen.prompts import StyleError, build_prompt, compose_prompt, estimate_tokens, load_style


def test_load_style_ignores_comments_and_whitespace(tmp_path):
    path = tmp_path / "style.txt"
    path.write_text("# comment line\nFlat cartoon,\n   bold outlines.\n\n# another\n")
    assert load_style(path) == "Flat cartoon, bold outlines."


def test_project_style_file_contains_required_art_direction():
    from pathlib import Path
    style = load_style(Path(__file__).resolve().parents[1] / "config" / "style.txt")
    for phrase in ("bold dark-brown outlines", "top 12%", "bottom 15%", "No text"):
        assert phrase in style


def test_missing_or_empty_style(tmp_path):
    with pytest.raises(StyleError, match="not found"):
        load_style(tmp_path / "missing.txt")
    (tmp_path / "empty.txt").write_text("# only a comment\n\n")
    with pytest.raises(StyleError, match="empty"):
        load_style(tmp_path / "empty.txt")


def test_compose_puts_character_before_style():
    text = compose_prompt(" An archer. ", "Cartoon style.", "{character}\n\n{style}")
    assert text == "An archer.\n\nCartoon style."
    assert text.index("archer") < text.index("Cartoon")


def test_custom_template():
    assert compose_prompt("Knight", "flat", "{style}, featuring {character}") == "flat, featuring Knight"


def test_build_prompt_uses_shared_style_and_negative():
    settings = build_settings({"negative_prompt": "text,  watermark"})
    card = Card(id="a", name="A", prompt="A jester.")
    spec = build_prompt(card, "Shared style.", settings)
    assert spec.prompt.startswith("A jester.") and spec.prompt.endswith("Shared style.")
    assert spec.negative_prompt == "text, watermark"
    assert spec.reference_images == ()


def test_card_negative_prompt_override():
    settings = build_settings({"negative_prompt": "global"})
    card = Card(id="a", name="A", prompt="x", overrides={"negative_prompt": "only this"})
    assert build_prompt(card, "s", settings).negative_prompt == "only this"


def test_token_estimate_flags_long_prompts():
    assert estimate_tokens("one two three") == 5
    from pathlib import Path
    style = load_style(Path(__file__).resolve().parents[1] / "config" / "style.txt")
    assert estimate_tokens(style) > 77  # documented reason long_prompts exists
