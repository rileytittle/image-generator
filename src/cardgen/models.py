"""Card definitions: loading, validation, filename safety and seed assignment."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .config import (
    GENERATION_KEYS,
    aspect_warning,
    is_int,
    is_number,
    validate_generation_values,
    validate_seed,
)

# IDs become filenames, so keep them to a portable, case-insensitively unique set.
ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
CARD_KEYS = {"id", "name", "prompt", "seed", "rarity", "overrides"}
OVERRIDE_KEYS = set(GENERATION_KEYS) | {"negative_prompt"}
MAX_PROMPT_CHARS = 2000


class CardValidationError(ValueError):
    """Raised when the card file has one or more problems. ``errors`` lists them all."""

    def __init__(self, errors: list[str]):
        self.errors = list(errors)
        super().__init__(f"{len(self.errors)} problem(s) in card file")


@dataclass(frozen=True)
class Card:
    id: str
    name: str
    prompt: str
    seed: int | None = None
    rarity: str | None = None  # informational only; recorded in the manifest
    overrides: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class CardSet:
    cards: list[Card]
    warnings: list[str]


def slugify(text: str, max_length: int = 64) -> str:
    """Turn arbitrary text into a filesystem-safe card ID (``"Sir Gö!"`` -> ``"sir_go"``)."""
    normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "_", normalized.lower()).strip("_-")
    return slug[:max_length].rstrip("_-")


def is_safe_id(card_id: Any) -> bool:
    return isinstance(card_id, str) and bool(ID_PATTERN.match(card_id))


def output_filename(card_id: str) -> str:
    """Predictable output filename for a card. Refuses unsafe IDs as a last line of defence."""
    if not is_safe_id(card_id):
        raise ValueError(f"unsafe card id for a filename: {card_id!r}")
    return f"{card_id}.png"


def derive_seed(global_seed: int, card_id: str) -> int:
    """Stable per-card seed: depends only on the global seed and the card ID,
    so adding, removing or reordering other cards never changes it."""
    digest = hashlib.sha256(f"{global_seed}:{card_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def resolve_seed(card: Card, global_seed: int) -> int:
    return card.seed if card.seed is not None else derive_seed(global_seed, card.id)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def _validate_card(raw: Any, where: str) -> tuple[Card | None, list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(raw, dict):
        return None, [f"{where}: each card must be a JSON object (got {type(raw).__name__})"], []

    for key in sorted(set(raw) - CARD_KEYS):
        errors.append(f"{where}: unknown field '{key}' (allowed: {', '.join(sorted(CARD_KEYS))})")

    card_id = raw.get("id")
    label = where
    if card_id is None or (isinstance(card_id, str) and not card_id.strip()):
        hint = f" (suggestion: \"{slugify(raw['name'])}\")" if isinstance(raw.get("name"), str) and slugify(raw["name"]) else ""
        errors.append(f"{where}: missing 'id'{hint}")
    elif not is_safe_id(card_id):
        suggestion = slugify(card_id) if isinstance(card_id, str) else ""
        hint = f" (suggestion: \"{suggestion}\")" if suggestion else ""
        errors.append(f"{where}: id {card_id!r} is not filesystem-safe; use 1-64 lowercase "
                      f"letters, digits, '_' or '-', starting with a letter or digit{hint}")
    else:
        label = f"{where} ('{card_id}')"

    name = raw.get("name")
    if name is None:
        name = card_id if isinstance(card_id, str) else ""
    elif not isinstance(name, str) or not name.strip():
        errors.append(f"{label}: 'name' must be a non-empty string when present")

    prompt = raw.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        errors.append(f"{label}: missing or empty 'prompt'")
    elif len(prompt) > MAX_PROMPT_CHARS:
        errors.append(f"{label}: 'prompt' is longer than {MAX_PROMPT_CHARS} characters")

    seed = raw.get("seed")
    if seed is not None:
        errors += validate_seed(seed, label)

    rarity = raw.get("rarity")
    if rarity is not None and (not isinstance(rarity, str) or not rarity.strip() or len(rarity) > 32):
        errors.append(f"{label}: 'rarity' must be a short non-empty string when present")

    overrides = raw.get("overrides") or {}
    if not isinstance(overrides, dict):
        errors.append(f"{label}: 'overrides' must be an object")
        overrides = {}
    else:
        bad = sorted(set(overrides) - OVERRIDE_KEYS)
        if bad:
            errors.append(f"{label}: unsupported override(s) {', '.join(bad)} "
                          f"(allowed: {', '.join(sorted(OVERRIDE_KEYS))})")
        errors += validate_generation_values(overrides, f"{label} overrides")
        if "negative_prompt" in overrides and not isinstance(overrides["negative_prompt"], str):
            errors.append(f"{label}: override 'negative_prompt' must be a string")
        if ("width" in overrides) != ("height" in overrides):
            errors.append(f"{label}: overrides must set both 'width' and 'height' or neither")
        elif is_int(overrides.get("width")) and is_int(overrides.get("height")):
            w = aspect_warning(overrides["width"], overrides["height"], label)
            if w:
                warnings.append(w)

    if errors:
        return None, errors, warnings
    return Card(id=card_id, name=name.strip(), prompt=" ".join(prompt.split()),
                seed=seed, rarity=rarity.strip() if rarity else None,
                overrides=dict(overrides)), [], warnings


def validate_cards(raw_cards: Any, source: str = "cards") -> CardSet:
    """Validate a parsed list of card dicts. Raises CardValidationError listing every problem;
    never drops an invalid card silently."""
    if not isinstance(raw_cards, list):
        raise CardValidationError([f"{source}: top level must be a JSON list of cards"])
    if not raw_cards:
        raise CardValidationError([f"{source}: the card list is empty"])

    errors: list[str] = []
    warnings: list[str] = []
    cards: list[Card] = []
    seen: dict[str, int] = {}
    for index, raw in enumerate(raw_cards):
        where = f"{source} card #{index + 1}"
        card, card_errors, card_warnings = _validate_card(raw, where)
        errors += card_errors
        warnings += card_warnings
        if isinstance(raw, dict) and isinstance(raw.get("id"), str):
            key = raw["id"].strip().lower()
            if key in seen:
                errors.append(f"{where}: duplicate id '{raw['id']}' (first used by card #{seen[key]})")
            else:
                seen[key] = index + 1
        if card:
            cards.append(card)
    if errors:
        raise CardValidationError(errors)
    return CardSet(cards=cards, warnings=warnings)


def load_cards(path: Path | str) -> CardSet:
    """Load cards from ``.json`` (primary) or ``.csv`` (columns: id,name,prompt[,seed,...])."""
    path = Path(path)
    if not path.exists():
        raise CardValidationError([f"card file not found: {path}"])
    if path.suffix.lower() == ".csv":
        return validate_cards(_read_csv(path), source=str(path))
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CardValidationError(
            [f"{path}: malformed JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"]) from exc
    except UnicodeDecodeError as exc:
        raise CardValidationError([f"{path}: file is not valid UTF-8 ({exc})"]) from exc
    return validate_cards(data, source=str(path))


def _read_csv(path: Path) -> list[dict[str, Any]]:
    """Read cards from CSV. Numeric columns are converted; empty cells mean "not set".
    Conversion failures are left as strings so validation reports them."""
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "prompt" not in reader.fieldnames:
                raise CardValidationError([f"{path}: CSV needs a header row with at least 'id' and 'prompt'"])
            rows = list(reader)
    except UnicodeDecodeError as exc:
        raise CardValidationError([f"{path}: file is not valid UTF-8 ({exc})"]) from exc

    cards: list[dict[str, Any]] = []
    for row in rows:
        if None in row:  # more cells than headers
            row.pop(None)
            row["__extra_cells__"] = True
        card: dict[str, Any] = {}
        overrides: dict[str, Any] = {}
        for key, value in row.items():
            if key == "__extra_cells__":
                card["extra_cells"] = "row has more cells than the header"
                continue
            value = (value or "").strip() if isinstance(value, str) else value
            if value == "":
                continue
            if key == "seed":
                card["seed"] = _to_number(value, int)
            elif key in ("width", "height", "steps"):
                overrides[key] = _to_number(value, int)
            elif key == "guidance_scale":
                overrides[key] = _to_number(value, float)
            elif key == "negative_prompt":
                overrides[key] = value
            else:
                card[key] = value
        if overrides:
            card["overrides"] = overrides
        cards.append(card)
    return cards


def _to_number(value: str, kind: type) -> Any:
    try:
        number = kind(value)
    except ValueError:
        return value
    return number if is_number(number) else value
