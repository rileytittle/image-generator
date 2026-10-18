"""Device/precision selection with a fake torch module (no real torch needed)."""

from types import SimpleNamespace

import pytest

from cardgen.generator import DeviceError, is_oom, mps_status, resolve_device, resolve_dtype


def fake_torch(built=True, available=True):
    mps = SimpleNamespace(is_built=lambda: built, is_available=lambda: available)
    return SimpleNamespace(backends=SimpleNamespace(mps=mps), float16="f16", float32="f32",
                           bfloat16="bf16")


def test_auto_prefers_mps():
    assert resolve_device("auto", False, fake_torch())[0] == "mps"
    assert resolve_device("mps", False, fake_torch())[0] == "mps"


def test_no_silent_cpu_fallback():
    with pytest.raises(DeviceError, match="allow_cpu_fallback"):
        resolve_device("auto", False, fake_torch(available=False))
    with pytest.raises(DeviceError, match="not built with MPS"):
        resolve_device("mps", True, fake_torch(built=False))
    device, why = resolve_device("auto", True, fake_torch(available=False))
    assert device == "cpu" and "falling back" in why
    assert resolve_device("cpu", False, fake_torch())[0] == "cpu"


def test_mps_status_without_backend():
    assert mps_status(SimpleNamespace(backends=SimpleNamespace()))[0] is False


def test_dtype_selection():
    t = fake_torch()
    assert resolve_dtype("auto", "mps", t) == "f16"
    assert resolve_dtype("auto", "cpu", t) == "f32"
    assert resolve_dtype("float32", "mps", t) == "f32"
    with pytest.raises(DeviceError):
        resolve_dtype("float16", "cpu", t)


def test_is_oom():
    assert is_oom(RuntimeError("MPS backend out of memory (MPS allocated: 18 GB)"))
    assert not is_oom(RuntimeError("shape mismatch"))
