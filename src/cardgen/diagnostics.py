"""``cardgen doctor``: environment checks. Never downloads a model."""

from __future__ import annotations

import importlib
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ConfigError, Settings, load_settings
from .prompts import StyleError, load_style

OK, WARN, FAIL = "ok", "warn", "fail"

# Rough sizes for planning; SDXL base fp16 weights are ~7 GB, fp32 ~14 GB.
SDXL_FP16_GB = 7
MIN_FREE_GB_WITHOUT_MODEL = SDXL_FP16_GB + 5
MIN_FREE_GB_WITH_MODEL = 2
RECOMMENDED_RAM_GB = 16


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str


def _sysctl(name: str) -> str | None:
    try:
        out = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def total_ram_gb() -> float | None:
    if sys.platform == "darwin":
        value = _sysctl("hw.memsize")
        if value and value.isdigit():
            return int(value) / 1024**3
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    except (ValueError, OSError, AttributeError):
        return None


def free_disk_gb(path: Path) -> float:
    probe = Path(path).resolve()
    while not probe.exists():
        probe = probe.parent
    return shutil.disk_usage(probe).free / 1024**3


def hf_cache_dir(settings: Settings) -> Path:
    if settings.cache_dir:
        return settings.cache_dir
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def model_cached(settings: Settings) -> bool | None:
    """Check the local Hugging Face cache for the model's index file (no network)."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    if Path(settings.model_id).is_dir():
        return (Path(settings.model_id) / "model_index.json").exists()
    found = try_to_load_from_cache(settings.model_id, "model_index.json",
                                   cache_dir=str(settings.cache_dir) if settings.cache_dir else None)
    return isinstance(found, str)


def _version(module: str) -> str | None:
    try:
        return getattr(importlib.import_module(module), "__version__", "installed")
    except ImportError:
        return None
    except Exception as exc:  # broken install
        return f"error: {exc}"


def run_checks(settings_path: Path, preset: str | None = None) -> list[Check]:
    checks: list[Check] = []
    add = lambda name, status, detail: checks.append(Check(name, status, detail))  # noqa: E731

    # ---- platform ----
    py = sys.version_info
    add("Python", OK if py >= (3, 10) else FAIL,
        f"{platform.python_version()} ({platform.python_implementation()}, {sys.executable})"
        + ("" if py >= (3, 10) else " - Python 3.10+ is required; 3.11 is recommended"))
    machine = platform.machine()
    system = platform.system()
    if system == "Darwin":
        translated = _sysctl("sysctl.proc_translated") == "1"
        mac = platform.mac_ver()[0]
        if translated:
            add("Platform", FAIL, f"macOS {mac}, Python is running under Rosetta (x86_64 emulation). "
                "Install an arm64 Python so PyTorch can use MPS.")
        elif machine == "arm64":
            add("Platform", OK, f"macOS {mac} on Apple Silicon (arm64)")
        else:
            add("Platform", WARN, f"macOS {mac} on {machine}: no MPS; SDXL on CPU will be very slow")
    else:
        add("Platform", WARN, f"{system} {machine}: not macOS, so no MPS. Only CPU generation is "
            "supported by this tool (very slow for SDXL)")
    ram = total_ram_gb()
    if ram is None:
        add("Memory", WARN, "could not determine total RAM")
    else:
        enough = ram >= RECOMMENDED_RAM_GB - 0.5  # OSes report slightly under the nominal size
        status = OK if enough else WARN
        detail = f"{ram:.0f} GB total"
        if not enough:
            detail += (f"; SDXL wants {RECOMMENDED_RAM_GB} GB+ of unified memory. Use the 'test' preset, "
                       "close other apps, or choose a smaller model")
        add("Memory", status, detail)

    # ---- dependencies ----
    for module, required in (("PIL", True), ("torch", True), ("diffusers", True),
                             ("transformers", True), ("accelerate", False), ("safetensors", True),
                             ("huggingface_hub", True)):
        version = _version(module)
        name = "Pillow" if module == "PIL" else module
        if version is None:
            add(name, FAIL if required else WARN, "not installed - run: pip install -e '.[gen]'")
        elif version.startswith("error"):
            add(name, FAIL, version)
        else:
            add(name, OK, version)

    torch_mod: Any = None
    try:
        import torch as torch_mod  # noqa: F811
    except Exception:
        pass
    if torch_mod is not None:
        from .generator import mps_status
        ok, why = mps_status(torch_mod)
        add("MPS", OK if ok else WARN, why)

    # ---- configuration ----
    try:
        settings = load_settings(settings_path, preset=preset)
    except ConfigError as exc:
        for error in exc.errors:
            add("Settings", FAIL, error)
        return checks
    add("Settings", OK, f"{settings_path} (preset: {settings.preset or 'none'}, "
        f"{settings.width}x{settings.height}, {settings.steps} steps)")
    for warning in settings.warnings:
        add("Settings", WARN, warning)

    if torch_mod is not None:
        from .generator import DeviceError, resolve_device
        try:
            device, why = resolve_device(settings.device, settings.allow_cpu_fallback, torch_mod)
            add("Device", OK if device == "mps" else WARN, f"{device}: {why}")
        except DeviceError as exc:
            add("Device", FAIL, str(exc))

    try:
        style = load_style(settings.style_file)
        add("Style", OK, f"{settings.style_file} ({len(style.split())} words)")
    except StyleError as exc:
        add("Style", FAIL, str(exc))

    # ---- model + disk ----
    cached = model_cached(settings)
    cache = hf_cache_dir(settings)
    if cached is None:
        add("Model", WARN, f"{settings.model_id}: cannot check the cache (huggingface_hub missing)")
    elif cached:
        add("Model", OK, f"{settings.model_id} found in local cache ({cache})")
    else:
        add("Model", WARN, f"{settings.model_id} is not in the local cache ({cache}). "
            "Download it once with: python -m cardgen generate --limit 1 --allow-download")
    need = MIN_FREE_GB_WITH_MODEL if cached else MIN_FREE_GB_WITHOUT_MODEL
    free_cache = free_disk_gb(cache)
    add("Disk (model cache)", OK if free_cache >= need else FAIL,
        f"{free_cache:.1f} GB free at {cache} (need about {need} GB)")

    out = settings.output_dir
    try:
        out.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=out, prefix=".write-test-"):
            pass
        free_out = free_disk_gb(out)
        add("Output folder", OK if free_out >= 1 else WARN, f"{out.resolve()} is writable, {free_out:.1f} GB free")
    except OSError as exc:
        add("Output folder", FAIL, f"{out}: not writable ({exc})")
    return checks
