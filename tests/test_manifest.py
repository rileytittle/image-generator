import json
import os

import pytest

from cardgen.manifest import FAILED, SUCCESS, Manifest, ManifestError, RunLock, atomic_write_text


def test_new_manifest_and_roundtrip(tmp_path):
    path = tmp_path / "out" / "manifest.json"
    m = Manifest.load(path)
    assert m.cards == {} and not path.exists()
    m.update("a", status=SUCCESS, seed=1)
    m.save()
    again = Manifest.load(path)
    assert again.get("a")["status"] == SUCCESS
    assert again.status("missing") == "pending"
    assert again.data["updated_at"]


def test_save_keeps_backup_and_leaves_no_temp_files(tmp_path):
    path = tmp_path / "manifest.json"
    m = Manifest(path)
    m.update("a", status=FAILED)
    m.save()
    m.update("a", status=SUCCESS)
    m.save()
    backup = json.loads((tmp_path / "manifest.json.bak").read_text())
    assert backup["cards"]["a"]["status"] == FAILED
    assert sorted(os.listdir(tmp_path)) == ["manifest.json", "manifest.json.bak"]


def test_corrupt_manifest_is_reported_not_overwritten(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text('{"cards": {"a": ')
    with pytest.raises(ManifestError, match="unreadable"):
        Manifest.load(path)
    assert path.read_text() == '{"cards": {"a": '


def test_wrong_structure(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text("[]")
    with pytest.raises(ManifestError, match="unexpected structure"):
        Manifest.load(path)


def test_atomic_write_failure_keeps_old_file(tmp_path, monkeypatch):
    path = tmp_path / "manifest.json"
    path.write_text("old")

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        atomic_write_text(path, "new")
    assert path.read_text() == "old"
    assert os.listdir(tmp_path) == ["manifest.json"]


def test_run_lock(tmp_path):
    with RunLock(tmp_path):
        assert (tmp_path / ".cardgen.lock").exists()
        # A live process (pid 1 always exists) holding the lock blocks a second run.
    (tmp_path / ".cardgen.lock").write_text("1")
    if os.getpid() != 1:
        with pytest.raises(ManifestError, match="another cardgen run"):
            RunLock(tmp_path).__enter__()
    # Stale lock (dead pid / garbage) is cleaned up.
    (tmp_path / ".cardgen.lock").write_text("not-a-pid")
    with RunLock(tmp_path):
        pass
    assert not (tmp_path / ".cardgen.lock").exists()
