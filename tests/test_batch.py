import pytest
from PIL import Image

from conftest import STYLE, FakeGenerator
from cardgen.batch import GENERATE, SKIP, PlanOptions, plan_batch, qc_rows, run_batch
from cardgen.config import build_settings
from cardgen.generator import BlankImageError, ModelLoadError, OutOfMemoryError
from cardgen.manifest import FAILED, INTERRUPTED, RUNNING, SUCCESS, Manifest
from cardgen.models import load_cards


@pytest.fixture
def cards(project):
    return load_cards(project / "cards.json").cards


def plan(cards, settings, manifest=None, **options):
    manifest = manifest or Manifest.load(settings.manifest_path)
    return plan_batch(cards, settings, STYLE, manifest, PlanOptions(**options))


def run(cards, settings, generator=None, **options):
    manifest = Manifest.load(settings.manifest_path)
    jobs = plan(cards, settings, manifest, **options)
    generator = generator or FakeGenerator()
    summary = run_batch(jobs, settings, manifest, generator, progress=lambda m: None)
    return summary, Manifest.load(settings.manifest_path), generator


def actions(jobs):
    return {j.card.id: j.action for j in jobs}


def test_plan_fresh_batch(cards, settings):
    jobs = plan(cards, settings)
    assert all(j.action == GENERATE and j.reason == "pending" for j in jobs)
    archer = jobs[0]
    assert archer.request.seed == 1001 and archer.seed_source == "card"
    assert jobs[2].seed_source == "derived"
    assert archer.request.prompt == f"An archer.\n\n{STYLE}"
    assert (archer.request.width, archer.request.height) == (256, 384)
    assert archer.output_path.as_posix() == "output/images/village_archer.png"


def test_limit_and_only(cards, settings):
    jobs = plan(cards, settings, limit=2)
    assert [j.action for j in jobs] == [GENERATE, GENERATE, SKIP]
    assert "limit" in jobs[2].reason
    jobs = plan(cards, settings, only=frozenset({"court_jester"}))
    assert [j.card.id for j in jobs] == ["court_jester"]


def test_successful_run_writes_images_and_manifest(cards, settings):
    summary, manifest, gen = run(cards, settings)
    assert summary.succeeded == ["village_archer", "brave_peasant", "court_jester"]
    assert gen.loaded == 1 and gen.closed == 1  # model loaded once per run
    entry = manifest.get("village_archer")
    assert entry["status"] == SUCCESS
    assert entry["seed"] == 1001 and entry["model_id"] == "fake/model"
    assert entry["settings"]["width"] == 256 and entry["settings"]["steps"] == 2
    assert entry["started_at"] and entry["completed_at"] and entry["error"] is None
    assert entry["full_prompt"].endswith(STYLE)
    with Image.open(entry["output_path"]) as img:
        assert img.size == (256, 384)
    assert manifest.data["runs"][-1]["succeeded"] == 3


def test_resume_skips_completed_and_never_overwrites(cards, settings):
    run(cards, settings, limit=1)
    path = settings.images_dir / "village_archer.png"
    before = path.read_bytes()
    summary, manifest, gen = run(cards, settings)
    assert summary.succeeded == ["brave_peasant", "court_jester"]
    assert [r.seed for r in gen.requests][0] == 1002
    assert path.read_bytes() == before
    # Nothing left to do: the generator is not even loaded.
    summary, _, gen = run(cards, settings)
    assert summary.planned == 0 and gen.loaded == 0


def test_existing_untracked_valid_image_is_not_overwritten(cards, settings):
    settings.images_dir.mkdir(parents=True)
    Image.new("RGB", (256, 384), (1, 2, 3)).save(settings.images_dir / "court_jester.png")
    jobs = plan(cards, settings)
    assert actions(jobs)["court_jester"] == SKIP
    assert "not overwritten" in jobs[2].reason


def test_existing_image_with_other_size_is_kept(cards, settings):
    settings.images_dir.mkdir(parents=True)
    Image.new("RGB", (512, 768)).save(settings.images_dir / "court_jester.png")
    jobs = plan(cards, settings)
    assert actions(jobs)["court_jester"] == SKIP and "512x768" in jobs[2].reason


def test_corrupt_or_missing_outputs_are_regenerated(cards, settings):
    run(cards, settings)
    (settings.images_dir / "village_archer.png").write_bytes(b"garbage")
    (settings.images_dir / "brave_peasant.png").unlink()
    jobs = plan(cards, settings)
    assert actions(jobs) == {"village_archer": GENERATE, "brave_peasant": GENERATE, "court_jester": SKIP}
    assert "invalid" in jobs[0].reason and "missing" in jobs[1].reason
    summary, manifest, _ = run(cards, settings)
    assert summary.succeeded == ["village_archer", "brave_peasant"]


def test_failed_card_does_not_stop_batch_and_is_retried_on_request(cards, settings):
    gen = FakeGenerator(fail_on={"A peasant": RuntimeError("boom")})
    summary, manifest, _ = run(cards, settings, gen)
    assert summary.failed == ["brave_peasant"]
    assert summary.succeeded == ["village_archer", "court_jester"]
    entry = manifest.get("brave_peasant")
    assert entry["status"] == FAILED and "boom" in entry["error"]["message"]
    assert not (settings.images_dir / "brave_peasant.png").exists()

    jobs = plan(cards, settings)
    assert actions(jobs)["brave_peasant"] == SKIP and "--retry-failed" in jobs[1].reason
    summary, manifest, gen = run(cards, settings, retry_failed=True)
    assert summary.succeeded == ["brave_peasant"]
    assert manifest.get("brave_peasant")["status"] == SUCCESS
    assert manifest.get("brave_peasant")["attempts"] == 2


def test_blank_image_error_fails_only_that_card(cards, settings):
    gen = FakeGenerator(fail_on={"An archer": BlankImageError("blank")})
    summary, _, _ = run(cards, settings, gen)
    assert summary.failed == ["village_archer"] and len(summary.succeeded) == 2


def test_fatal_error_stops_batch(cards, settings):
    gen = FakeGenerator(fail_on={"A peasant": OutOfMemoryError("oom")})
    summary, manifest, _ = run(cards, settings, gen)
    assert summary.fatal and "oom" in summary.aborted
    assert summary.succeeded == ["village_archer"] and summary.not_reached == 1
    assert manifest.status("court_jester") == "pending"


def test_model_load_failure_marks_nothing(cards, settings):
    class BadLoad(FakeGenerator):
        def load(self):
            raise ModelLoadError("not cached")
    summary, manifest, gen = run(cards, settings, BadLoad())
    assert summary.fatal and "not cached" in summary.aborted
    assert manifest.cards == {} and gen.closed == 1


def test_consecutive_failure_limit(cards, project):
    settings = build_settings({**__import__("conftest").BASE_SETTINGS, "max_consecutive_failures": 2})
    gen = FakeGenerator(fail_on={"A": RuntimeError("bad")})  # every prompt starts with "A"/"An"
    summary, _, _ = run(cards, settings, gen)
    assert len(summary.failed) == 2 and "in a row" in summary.aborted and not summary.fatal


def test_interrupt_marks_card_and_resume_continues(cards, settings):
    gen = FakeGenerator(fail_on={"A peasant": KeyboardInterrupt()})
    summary, manifest, _ = run(cards, settings, gen)
    assert summary.interrupted and summary.succeeded == ["village_archer"]
    assert manifest.get("brave_peasant")["status"] == INTERRUPTED
    jobs = plan(cards, settings)
    assert actions(jobs) == {"village_archer": SKIP, "brave_peasant": GENERATE, "court_jester": GENERATE}


def test_crash_mid_card_leaves_running_status_which_resumes(cards, settings):
    manifest = Manifest.load(settings.manifest_path)
    manifest.update("village_archer", status=RUNNING)
    manifest.save()
    jobs = plan(cards, settings)
    assert jobs[0].action == GENERATE and "interrupted" in jobs[0].reason


def test_wrong_size_from_model_is_a_failure(cards, settings):
    summary, _, _ = run(cards, settings, FakeGenerator(size_override=(100, 100)), limit=1)
    assert summary.failed == ["village_archer"]


def test_overwrite_archives_previous_image(cards, settings):
    run(cards, settings, limit=1)
    jobs = plan(cards, settings, overwrite=True, only=frozenset({"village_archer"}))
    assert jobs[0].action == GENERATE and jobs[0].replaces_existing
    summary, manifest, _ = run(cards, settings, overwrite=True, only=frozenset({"village_archer"}))
    assert summary.succeeded == ["village_archer"]
    archived = list((settings.output_dir / "replaced").glob("village_archer-*.png"))
    assert len(archived) == 1
    assert manifest.get("village_archer")["replaced_previous"] == str(archived[0])


def test_changed_prompt_is_reported_but_not_regenerated(cards, settings):
    run(cards, settings, limit=1)
    manifest = Manifest.load(settings.manifest_path)
    jobs = plan_batch(cards, settings, "A different style", manifest, PlanOptions())
    assert jobs[0].action == SKIP and "changed" in jobs[0].reason


def test_postprocess_during_run(cards, project):
    from conftest import BASE_SETTINGS
    settings = build_settings({**BASE_SETTINGS, "postprocess": {"enabled": True, "width": 512, "height": 768}})
    summary, manifest, _ = run(cards, settings, limit=1)
    entry = manifest.get("village_archer")
    with Image.open(entry["final_path"]) as img:
        assert img.size == (512, 768)
    with Image.open(entry["output_path"]) as img:
        assert img.size == (256, 384)  # original preserved


def test_qc_rows(cards, settings):
    run(cards, settings, limit=2)
    (settings.images_dir / "brave_peasant.png").write_bytes(b"bad")
    rows = {r["id"]: r for r in qc_rows(Manifest.load(settings.manifest_path), settings)}
    assert rows["village_archer"]["valid"] and rows["village_archer"]["aspect_2_3"]
    assert rows["brave_peasant"]["review"] and not rows["brave_peasant"]["valid"]
