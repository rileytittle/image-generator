"""Persistent run manifest (``output/manifest.json``) and a simple run lock.

The manifest is rewritten atomically after every state change (write a temp
file, fsync, ``os.replace``), and the previous version is kept as
``manifest.json.bak``, so an interrupted run never leaves a corrupt file.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MANIFEST_VERSION = 1

PENDING = "pending"
RUNNING = "running"
SUCCESS = "success"
FAILED = "failed"
INTERRUPTED = "interrupted"
STATUSES = (PENDING, RUNNING, SUCCESS, FAILED, INTERRUPTED)


class ManifestError(RuntimeError):
    pass


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


class Manifest:
    def __init__(self, path: Path, data: dict[str, Any] | None = None):
        self.path = Path(path)
        self.data = data or {"version": MANIFEST_VERSION, "created_at": now_iso(),
                             "updated_at": None, "cards": {}}

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        path = Path(path)
        if not path.exists():
            return cls(path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            backup = path.with_name(path.name + ".bak")
            hint = f" A previous copy may be at {backup}." if backup.exists() else ""
            raise ManifestError(
                f"manifest {path} is unreadable ({exc}). It was not modified.{hint} "
                "Restore it, or move it aside to start a fresh manifest (existing valid images "
                "will still be detected and skipped).") from exc
        if not isinstance(data, dict) or not isinstance(data.get("cards"), dict):
            raise ManifestError(f"manifest {path} has an unexpected structure; move it aside to start fresh")
        if data.get("version", 0) > MANIFEST_VERSION:
            raise ManifestError(f"manifest {path} was written by a newer version of cardgen")
        return cls(path, data)

    @property
    def cards(self) -> dict[str, dict[str, Any]]:
        return self.data["cards"]

    def get(self, card_id: str) -> dict[str, Any] | None:
        return self.cards.get(card_id)

    def status(self, card_id: str) -> str:
        entry = self.get(card_id)
        return entry["status"] if entry else PENDING

    def update(self, card_id: str, **fields: Any) -> dict[str, Any]:
        entry = self.cards.setdefault(card_id, {"id": card_id, "attempts": 0})
        entry.update(fields)
        return entry

    def save(self) -> None:
        self.data["updated_at"] = now_iso()
        text = json.dumps(self.data, indent=2, ensure_ascii=False, sort_keys=False) + "\n"
        if self.path.exists():
            try:
                shutil.copy2(self.path, self.path.with_name(self.path.name + ".bak"))
            except OSError:
                pass
        atomic_write_text(self.path, text)


class RunLock:
    """Prevents two generate runs from writing the same output folder at once."""

    def __init__(self, output_dir: Path):
        self.path = Path(output_dir) / ".cardgen.lock"
        self.acquired = False

    def __enter__(self) -> "RunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if self._stale():
                    self.path.unlink(missing_ok=True)
                    continue
                raise ManifestError(
                    f"another cardgen run appears to be using this output folder ({self.path}). "
                    "If no other run is active, delete that lock file.") from None
            with os.fdopen(fd, "w") as handle:
                handle.write(str(os.getpid()))
            self.acquired = True
            return self
        raise ManifestError(f"could not acquire lock {self.path}")

    def _stale(self) -> bool:
        try:
            pid = int(self.path.read_text().strip())
        except (OSError, ValueError):
            return True
        if pid == os.getpid():
            return True
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        return False

    def __exit__(self, *exc: Any) -> None:
        if self.acquired:
            self.path.unlink(missing_ok=True)
            self.acquired = False
