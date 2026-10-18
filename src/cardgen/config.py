"""Loading and validating generation settings (``config/settings.json``).

Resolution order for every value (later wins):

1. Built-in defaults (``DEFAULTS`` below).
2. Top-level keys in the settings file.
3. The selected preset (``"preset"`` key or ``--preset``), if any.
4. Command-line overrides (``--width``, ``--steps``, ...).

Relative paths in the settings file are resolved against the current working
directory, so run commands from the project root.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

DEFAULT_SETTINGS_PATH = Path("config/settings.json")

VALID_DEVICES = ("auto", "mps", "cpu")
VALID_DTYPES = ("auto", "float16", "bfloat16", "float32")
VALID_EXISTING = ("skip", "overwrite")
VALID_FITS = ("crop", "pad")

# Keys that a preset (or a per-card override) may set.
GENERATION_KEYS = ("width", "height", "steps", "guidance_scale")

DEFAULTS: dict[str, Any] = {
    "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
    "model_variant": "fp16",
    "cache_dir": None,
    "allow_download": False,
    "device": "auto",
    "allow_cpu_fallback": False,
    "dtype": "auto",
    "attention_slicing": "auto",
    "long_prompts": True,
    "preset": None,
    "presets": {},
    "width": 512,
    "height": 768,
    "steps": 25,
    "guidance_scale": 6.5,
    "default_seed": 12345,
    "style_file": "config/style.txt",
    "prompt_template": "{character}\n\n{style}",
    "negative_prompt": "",
    "output_dir": "output",
    "image_format": "png",
    "existing_outputs": "skip",
    "retry_failed": False,
    "max_consecutive_failures": 3,
    "safe_zones": {"top": 0.12, "bottom": 0.15},
    "postprocess": {
        "enabled": False,
        "width": 1024,
        "height": 1536,
        "fit": "crop",
        "pad_color": "#000000",
        "output_subdir": "final",
    },
}

MAX_SEED = 2**32 - 1
MAX_SIDE = 2048


class ConfigError(ValueError):
    """Raised when settings are missing, malformed or out of range."""

    def __init__(self, errors: list[str] | str):
        self.errors = [errors] if isinstance(errors, str) else list(errors)
        super().__init__("; ".join(self.errors))


@dataclass(frozen=True)
class PostprocessSettings:
    enabled: bool = False
    width: int = 1024
    height: int = 1536
    fit: str = "crop"
    pad_color: str = "#000000"
    output_subdir: str = "final"


@dataclass(frozen=True)
class Settings:
    model_id: str
    model_variant: str | None
    cache_dir: Path | None
    allow_download: bool
    device: str
    allow_cpu_fallback: bool
    dtype: str
    attention_slicing: str | bool
    long_prompts: bool
    preset: str | None
    width: int
    height: int
    steps: int
    guidance_scale: float
    default_seed: int
    style_file: Path
    prompt_template: str
    negative_prompt: str
    output_dir: Path
    image_format: str
    existing_outputs: str
    retry_failed: bool
    max_consecutive_failures: int
    safe_top: float
    safe_bottom: float
    postprocess: PostprocessSettings = field(default_factory=PostprocessSettings)
    warnings: tuple[str, ...] = ()

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"

    @property
    def images_dir(self) -> Path:
        return self.output_dir / "images"

    @property
    def final_dir(self) -> Path:
        return self.output_dir / self.postprocess.output_subdir

    @property
    def logs_dir(self) -> Path:
        return self.output_dir / "logs"

    def summary(self) -> dict[str, Any]:
        data = asdict(self)
        for key, value in list(data.items()):
            if isinstance(value, Path):
                data[key] = str(value)
        data.pop("warnings", None)
        return data


# --------------------------------------------------------------------------- #
# Validation helpers (shared with per-card overrides in models.py)
# --------------------------------------------------------------------------- #


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def is_number(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def validate_generation_values(values: Mapping[str, Any], where: str) -> list[str]:
    """Validate width/height/steps/guidance_scale values present in ``values``."""
    errors: list[str] = []
    for key in ("width", "height"):
        if key in values:
            v = values[key]
            if not is_int(v) or v <= 0:
                errors.append(f"{where}: '{key}' must be a positive integer (got {v!r})")
            elif v % 8 != 0:
                errors.append(f"{where}: '{key}' must be a multiple of 8 (got {v})")
            elif v < 256 or v > MAX_SIDE:
                errors.append(f"{where}: '{key}' must be between 256 and {MAX_SIDE} (got {v})")
    if "steps" in values:
        v = values["steps"]
        if not is_int(v) or not 1 <= v <= 200:
            errors.append(f"{where}: 'steps' must be an integer from 1 to 200 (got {v!r})")
    if "guidance_scale" in values:
        v = values["guidance_scale"]
        if not is_number(v) or not 0 <= v <= 30:
            errors.append(f"{where}: 'guidance_scale' must be a number from 0 to 30 (got {v!r})")
    return errors


def validate_seed(value: Any, where: str) -> list[str]:
    if not is_int(value) or not 0 <= value <= MAX_SEED:
        return [f"{where}: seed must be an integer from 0 to {MAX_SEED} (got {value!r})"]
    return []


def aspect_warning(width: int, height: int, where: str) -> str | None:
    if abs(width / height - 2 / 3) > 0.01:
        return (f"{where}: {width}x{height} is not a 2:3 portrait "
                f"(ratio {width / height:.3f}, expected 0.667)")
    return None


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def read_settings_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"settings file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"{path}: invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a JSON object")
    return data


def load_settings(
    path: Path | str = DEFAULT_SETTINGS_PATH,
    *,
    preset: str | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> Settings:
    """Load, merge and validate settings. Raises ConfigError listing every problem."""
    path = Path(path)
    raw = read_settings_file(path)
    return build_settings(raw, preset=preset, overrides=overrides, source=str(path))


def build_settings(
    raw: Mapping[str, Any],
    *,
    preset: str | None = None,
    overrides: Mapping[str, Any] | None = None,
    source: str = "settings",
) -> Settings:
    errors: list[str] = []
    warnings: list[str] = []

    unknown = sorted(set(raw) - set(DEFAULTS))
    for key in unknown:
        warnings.append(f"{source}: unknown key '{key}' is ignored")

    merged: dict[str, Any] = {**DEFAULTS, **{k: v for k, v in raw.items() if k in DEFAULTS}}
    merged["postprocess"] = {**DEFAULTS["postprocess"], **(raw.get("postprocess") or {})}
    merged["safe_zones"] = {**DEFAULTS["safe_zones"], **(raw.get("safe_zones") or {})}

    presets = merged.get("presets") or {}
    if not isinstance(presets, dict):
        errors.append(f"{source}: 'presets' must be an object")
        presets = {}
    selected = preset if preset is not None else merged.get("preset")
    if selected:
        if selected not in presets:
            available = ", ".join(sorted(presets)) or "none defined"
            errors.append(f"{source}: unknown preset '{selected}' (available: {available})")
        elif not isinstance(presets[selected], dict):
            errors.append(f"{source}: preset '{selected}' must be an object")
        else:
            bad = sorted(set(presets[selected]) - set(GENERATION_KEYS))
            if bad:
                errors.append(f"{source}: preset '{selected}' has unsupported keys: {', '.join(bad)}")
            merged.update({k: v for k, v in presets[selected].items() if k in GENERATION_KEYS})
    merged["preset"] = selected or None

    for key, value in (overrides or {}).items():
        if value is not None:
            merged[key] = value

    # ---- field validation ----
    if not isinstance(merged["model_id"], str) or not merged["model_id"].strip():
        errors.append(f"{source}: 'model_id' must be a non-empty string")
    if merged["model_variant"] is not None and not isinstance(merged["model_variant"], str):
        errors.append(f"{source}: 'model_variant' must be a string or null")
    if merged["device"] not in VALID_DEVICES:
        errors.append(f"{source}: 'device' must be one of {', '.join(VALID_DEVICES)} "
                      f"(got {merged['device']!r})")
    if merged["dtype"] not in VALID_DTYPES:
        errors.append(f"{source}: 'dtype' must be one of {', '.join(VALID_DTYPES)}")
    if merged["attention_slicing"] not in ("auto", True, False):
        errors.append(f"{source}: 'attention_slicing' must be \"auto\", true or false")
    for key in ("allow_download", "allow_cpu_fallback", "retry_failed", "long_prompts"):
        if not isinstance(merged[key], bool):
            errors.append(f"{source}: '{key}' must be true or false")

    errors += validate_generation_values(merged, source)
    errors += validate_seed(merged["default_seed"], f"{source}: 'default_seed'")

    if merged["image_format"] != "png":
        errors.append(f"{source}: 'image_format' must be \"png\" (the only supported format)")
    if merged["existing_outputs"] not in VALID_EXISTING:
        errors.append(f"{source}: 'existing_outputs' must be one of {', '.join(VALID_EXISTING)}")
    mcf = merged["max_consecutive_failures"]
    if not is_int(mcf) or mcf < 0:
        errors.append(f"{source}: 'max_consecutive_failures' must be an integer >= 0 (0 = no limit)")

    template = merged["prompt_template"]
    if not isinstance(template, str) or "{character}" not in template:
        errors.append(f"{source}: 'prompt_template' must contain '{{character}}'")
    else:
        try:
            template.format(character="", style="")
        except (KeyError, IndexError, ValueError) as exc:
            errors.append(f"{source}: 'prompt_template' may only use {{character}} and {{style}} ({exc})")
    if not isinstance(merged["negative_prompt"], str):
        errors.append(f"{source}: 'negative_prompt' must be a string")
    for key in ("style_file", "output_dir"):
        if not isinstance(merged[key], str) or not merged[key].strip():
            errors.append(f"{source}: '{key}' must be a non-empty path string")
    if merged["cache_dir"] is not None and not isinstance(merged["cache_dir"], str):
        errors.append(f"{source}: 'cache_dir' must be a path string or null")

    zones = merged["safe_zones"]
    for key in ("top", "bottom"):
        v = zones.get(key)
        if not is_number(v) or not 0 <= v < 0.5:
            errors.append(f"{source}: 'safe_zones.{key}' must be a fraction from 0 to 0.5")

    pp = merged["postprocess"]
    pp_errors = validate_generation_values(
        {"width": pp.get("width"), "height": pp.get("height")}, f"{source}: postprocess")
    errors += pp_errors
    if pp.get("fit") not in VALID_FITS:
        errors.append(f"{source}: 'postprocess.fit' must be one of {', '.join(VALID_FITS)}")
    if not isinstance(pp.get("enabled"), bool):
        errors.append(f"{source}: 'postprocess.enabled' must be true or false")
    if not isinstance(pp.get("pad_color"), str):
        errors.append(f"{source}: 'postprocess.pad_color' must be a color string like \"#3b2a1a\"")
    else:
        try:
            from PIL import ImageColor
            ImageColor.getrgb(pp["pad_color"])
        except (ValueError, ImportError):
            errors.append(f"{source}: 'postprocess.pad_color' is not a valid color: {pp['pad_color']!r}")
    sub = pp.get("output_subdir")
    if not isinstance(sub, str) or not sub or "/" in sub or "\\" in sub or sub in (".", "..", "images", "logs"):
        errors.append(f"{source}: 'postprocess.output_subdir' must be a simple folder name")

    if errors:
        raise ConfigError(errors)

    w = aspect_warning(merged["width"], merged["height"], source)
    if w:
        warnings.append(w)
    if not pp_errors:
        w = aspect_warning(pp["width"], pp["height"], f"{source}: postprocess")
        if w:
            warnings.append(w)

    return Settings(
        model_id=merged["model_id"].strip(),
        model_variant=merged["model_variant"] or None,
        cache_dir=Path(merged["cache_dir"]).expanduser() if merged["cache_dir"] else None,
        allow_download=merged["allow_download"],
        device=merged["device"],
        allow_cpu_fallback=merged["allow_cpu_fallback"],
        dtype=merged["dtype"],
        attention_slicing=merged["attention_slicing"],
        long_prompts=merged["long_prompts"],
        preset=merged["preset"],
        width=merged["width"],
        height=merged["height"],
        steps=merged["steps"],
        guidance_scale=float(merged["guidance_scale"]),
        default_seed=merged["default_seed"],
        style_file=Path(merged["style_file"]).expanduser(),
        prompt_template=template,
        negative_prompt=merged["negative_prompt"].strip(),
        output_dir=Path(merged["output_dir"]).expanduser(),
        image_format=merged["image_format"],
        existing_outputs=merged["existing_outputs"],
        retry_failed=merged["retry_failed"],
        max_consecutive_failures=mcf,
        safe_top=float(zones["top"]),
        safe_bottom=float(zones["bottom"]),
        postprocess=PostprocessSettings(
            enabled=pp["enabled"], width=pp["width"], height=pp["height"],
            fit=pp["fit"], pad_color=pp["pad_color"], output_subdir=sub,
        ),
        warnings=tuple(warnings),
    )
