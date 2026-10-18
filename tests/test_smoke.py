"""Opt-in smoke test: generates ONE low-resolution image with an already downloaded model.

    CARDGEN_SMOKE=1 python -m pytest tests/test_smoke.py -s

Uses the model from config/settings.json (override with CARDGEN_SMOKE_MODEL) and
never downloads: it fails if the model is not already in the local cache.
"""

import os
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.smoke,
    pytest.mark.skipif(os.environ.get("CARDGEN_SMOKE") != "1", reason="set CARDGEN_SMOKE=1 to run"),
]

ROOT = Path(__file__).resolve().parents[1]


def test_generate_one_image(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    from cardgen.batch import PlanOptions, plan_batch, run_batch
    from cardgen.config import load_settings
    from cardgen.generator import DiffusersGenerator
    from cardgen.image_utils import check_image
    from cardgen.manifest import Manifest
    from cardgen.models import load_cards
    from cardgen.prompts import load_style

    overrides = {"output_dir": str(tmp_path), "width": 512, "height": 768,
                 "steps": int(os.environ.get("CARDGEN_SMOKE_STEPS", "20")), "allow_download": False}
    if os.environ.get("CARDGEN_SMOKE_MODEL"):
        overrides["model_id"] = os.environ["CARDGEN_SMOKE_MODEL"]
    if "CARDGEN_SMOKE_VARIANT" in os.environ:  # "" means no weight variant
        overrides["model_variant"] = os.environ["CARDGEN_SMOKE_VARIANT"]
    if os.environ.get("CARDGEN_SMOKE_DEVICE"):
        overrides["device"] = os.environ["CARDGEN_SMOKE_DEVICE"]
    settings = load_settings(ROOT / "config" / "settings.json", overrides=overrides)
    settings = settings.__class__(**{**settings.__dict__, "style_file": ROOT / "config" / "style.txt"})
    cards = load_cards(ROOT / "cards.json").cards[:1]
    manifest = Manifest.load(settings.manifest_path)
    jobs = plan_batch(cards, settings, load_style(settings.style_file), manifest, PlanOptions())
    summary = run_batch(jobs, settings, manifest, DiffusersGenerator(settings), progress=print)

    assert summary.succeeded == [cards[0].id], summary.aborted or Manifest.load(settings.manifest_path).cards
    check = check_image(jobs[0].output_path, (512, 768))
    assert check.ok, check.reason
    print(f"\nSmoke image: {jobs[0].output_path}")
    print(Manifest.load(settings.manifest_path).get(cards[0].id)["generation_info"])
