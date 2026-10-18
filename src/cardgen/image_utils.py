"""Image saving, validation, resizing and a heuristic safe-zone check (Pillow only)."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from PIL import Image, ImageColor, ImageFilter, ImageStat, PngImagePlugin, UnidentifiedImageError


@dataclass(frozen=True)
class ImageCheck:
    ok: bool
    reason: str
    size: tuple[int, int] | None = None

    @property
    def exists(self) -> bool:
        return self.reason != "missing"


def check_image(path: Path, expected_size: tuple[int, int] | None = None) -> ImageCheck:
    """Verify that ``path`` is a complete, decodable PNG (and optionally its size)."""
    path = Path(path)
    if not path.exists():
        return ImageCheck(False, "missing")
    if path.stat().st_size == 0:
        return ImageCheck(False, "empty file")
    try:
        with Image.open(path) as img:
            fmt = img.format
            img.verify()  # structure/CRC check
        with Image.open(path) as img:
            img.load()  # full decode catches truncated data
            size = img.size
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        return ImageCheck(False, f"unreadable image ({exc})")
    if fmt != "PNG":
        return ImageCheck(False, f"not a PNG (format {fmt})", size)
    if expected_size and size != tuple(expected_size):
        return ImageCheck(False, f"size {size[0]}x{size[1]} != expected "
                                 f"{expected_size[0]}x{expected_size[1]}", size)
    return ImageCheck(True, "ok", size)


def save_png_atomic(image: Image.Image, path: Path, metadata: Mapping[str, str] | None = None) -> None:
    """Write a PNG to a temp file in the same folder, verify it, then atomically
    rename it into place, so a crash never leaves a half-written image at ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    info = PngImagePlugin.PngInfo()
    for key, value in (metadata or {}).items():
        info.add_text(key, str(value))
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".png.tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            image.save(handle, format="PNG", pnginfo=info)
            handle.flush()
            os.fsync(handle.fileno())
        check = check_image(tmp, image.size)
        if not check.ok:
            raise OSError(f"written PNG failed verification: {check.reason}")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def is_blank(image: Image.Image) -> bool:
    """True if every pixel has the same colour (e.g. the all-black output of a NaN failure)."""
    extrema = image.convert("RGB").getextrema()
    return all(low == high for low, high in extrema)


def fit_image(image: Image.Image, width: int, height: int, fit: str = "crop",
              pad_color: str = "#000000") -> Image.Image:
    """Resize to exactly ``width`` x ``height`` without distortion.

    - ``crop``: scale to cover the target, then centre-crop the overflow.
    - ``pad``:  scale to fit inside the target, then centre on a ``pad_color`` canvas.

    When the aspect ratios already match, both give a plain high-quality resize.
    """
    src_w, src_h = image.size
    if fit == "crop":
        scale = max(width / src_w, height / src_h)
    elif fit == "pad":
        scale = min(width / src_w, height / src_h)
    else:
        raise ValueError(f"unknown fit mode: {fit}")
    new_w = max(1, round(src_w * scale))
    new_h = max(1, round(src_h * scale))
    resized = image.convert("RGB").resize((new_w, new_h), Image.Resampling.LANCZOS)
    if fit == "crop":
        left = (new_w - width) // 2
        top = (new_h - height) // 2
        return resized.crop((left, top, left + width, top + height))
    canvas = Image.new("RGB", (width, height), ImageColor.getrgb(pad_color))
    canvas.paste(resized, ((width - new_w) // 2, (height - new_h) // 2))
    return canvas


def safe_zone_report(image: Image.Image, top: float, bottom: float) -> dict[str, float]:
    """Heuristic only: compares edge density (a proxy for visual detail) in the
    top/bottom banner zones against the middle of the image. A ratio well above
    1.0 suggests busy detail where the banner or stats bar will go. It does not
    know what a character, face or weapon is, so treat it as a hint for review."""
    gray = image.convert("L")
    w, h = gray.size
    edges = gray.filter(ImageFilter.FIND_EDGES)
    top_px, bottom_px = int(h * top), int(h * bottom)

    def density(box: tuple[int, int, int, int]) -> float:
        if box[3] <= box[1]:
            return 0.0
        return ImageStat.Stat(edges.crop(box)).mean[0]

    # Ignore a 2px margin: FIND_EDGES marks the image border itself.
    top_d = density((2, 2, w - 2, top_px))
    mid_d = density((2, top_px, w - 2, h - bottom_px))
    bottom_d = density((2, h - bottom_px, w - 2, h - 2))
    base = max(mid_d, 1e-6)
    return {
        "top_edge_density": round(top_d, 2),
        "middle_edge_density": round(mid_d, 2),
        "bottom_edge_density": round(bottom_d, 2),
        "top_ratio": round(top_d / base, 2),
        "bottom_ratio": round(bottom_d / base, 2),
    }
