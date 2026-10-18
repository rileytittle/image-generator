import json

import pytest

from cardgen.config import DEFAULTS, ConfigError, build_settings, load_settings


def test_defaults_are_valid_and_portrait():
    s = build_settings({})
    assert s.model_id == DEFAULTS["model_id"]
    assert (s.width, s.height) == (512, 768)
    assert s.width / s.height == pytest.approx(2 / 3)
    assert s.device == "auto" and s.allow_cpu_fallback is False
    assert s.existing_outputs == "skip"
    assert s.manifest_path.as_posix() == "output/manifest.json"
    assert s.warnings == ()


def test_project_settings_file_is_valid():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    s = load_settings(root / "config" / "settings.json")
    assert (s.width, s.height) == (512, 768)
    q = load_settings(root / "config" / "settings.json", preset="quality")
    assert (q.width, q.height) == (832, 1248)
    assert q.width / q.height == pytest.approx(2 / 3)


def test_preset_then_overrides_take_precedence():
    raw = {"width": 640, "presets": {"big": {"width": 832, "height": 1248}}, "preset": "big"}
    s = build_settings(raw)
    assert (s.width, s.height) == (832, 1248)
    s = build_settings(raw, overrides={"width": 512, "height": 768, "steps": None})
    assert (s.width, s.height) == (512, 768)
    assert s.steps == DEFAULTS["steps"]


def test_unknown_preset_is_an_error():
    with pytest.raises(ConfigError, match="unknown preset 'nope'"):
        build_settings({}, preset="nope")


@pytest.mark.parametrize("raw, message", [
    ({"width": 500}, "multiple of 8"),
    ({"height": 0}, "positive integer"),
    ({"width": 4096}, "between 256"),
    ({"steps": 0}, "'steps'"),
    ({"steps": True}, "'steps'"),
    ({"guidance_scale": "high"}, "guidance_scale"),
    ({"device": "cuda"}, "'device'"),
    ({"default_seed": -1}, "seed"),
    ({"image_format": "jpg"}, "image_format"),
    ({"existing_outputs": "replace"}, "existing_outputs"),
    ({"prompt_template": "{style}"}, "{character}"),
    ({"prompt_template": "{character} {oops}"}, "prompt_template"),
    ({"postprocess": {"fit": "stretch"}}, "postprocess.fit"),
    ({"postprocess": {"pad_color": "not-a-color"}}, "pad_color"),
    ({"safe_zones": {"top": 0.9}}, "safe_zones.top"),
    ({"allow_download": "yes"}, "allow_download"),
])
def test_invalid_values_are_rejected(raw, message):
    with pytest.raises(ConfigError) as info:
        build_settings(raw)
    assert any(message in e for e in info.value.errors), info.value.errors


def test_all_errors_reported_together():
    with pytest.raises(ConfigError) as info:
        build_settings({"width": 500, "steps": 0, "device": "gpu"})
    assert len(info.value.errors) == 3


def test_non_portrait_ratio_warns_but_loads():
    s = build_settings({"width": 768, "height": 768})
    assert any("not a 2:3" in w for w in s.warnings)


def test_unknown_key_warns():
    s = build_settings({"widht": 512})
    assert any("widht" in w for w in s.warnings)


def test_malformed_settings_json(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{ not json")
    with pytest.raises(ConfigError, match="invalid JSON at line 1"):
        load_settings(path)


def test_missing_settings_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_settings(tmp_path / "nope.json")


def test_settings_file_must_be_object(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps([1, 2]))
    with pytest.raises(ConfigError, match="JSON object"):
        load_settings(path)
