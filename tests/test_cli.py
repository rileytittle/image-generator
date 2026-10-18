import json

import pytest

from cardgen.cli import EXIT_CARD_FAILURES, EXIT_ENVIRONMENT, EXIT_INTERRUPTED, EXIT_INVALID, EXIT_OK, main
from cardgen.generator import ModelLoadError
from cardgen.manifest import Manifest


def test_validate_ok(project, capsys):
    assert main(["validate"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "OK: 3 cards" in out and "2 with explicit seeds" in out


def test_validate_reports_all_problems(project, capsys):
    (project / "cards.json").write_text(json.dumps([{"id": "a", "prompt": "x"}, {"id": "a"}]))
    assert main(["validate"]) == EXIT_INVALID
    err = capsys.readouterr().err
    assert "duplicate id 'a'" in err and "'prompt'" in err


def test_validate_bad_settings(project, capsys):
    assert main(["validate", "--width", "500"]) == EXIT_INVALID
    assert "multiple of 8" in capsys.readouterr().err


def test_validate_missing_style(project, capsys):
    (project / "config" / "style.txt").unlink()
    assert main(["validate"]) == EXIT_INVALID
    assert "style file not found" in capsys.readouterr().err


def test_dry_run_loads_nothing_and_writes_nothing(project, capsys, monkeypatch):
    def explode(settings):
        raise AssertionError("model must not be created in a dry run")
    monkeypatch.setattr("cardgen.generator.create_generator", explode)
    assert main(["generate", "--dry-run", "--limit", "2"]) == EXIT_OK
    out = capsys.readouterr().out
    assert out.count("GENERATE") == 2 and "beyond --limit 2" in out
    assert "Example full prompt" in out and "Dry run" in out
    assert not (project / "output").exists()


def test_generate_requires_confirmation_when_not_interactive(project, fake_generator, capsys):
    assert main(["generate"]) == EXIT_INVALID
    assert "--yes" in capsys.readouterr().err
    assert fake_generator.loaded == 0


def test_generate_with_yes_then_resume(project, fake_generator, capsys):
    assert main(["generate", "--yes", "--limit", "1"]) == EXIT_OK
    assert (project / "output" / "images" / "village_archer.png").exists()
    assert list((project / "output" / "logs").glob("run-*.log"))
    assert main(["generate", "--yes"]) == EXIT_OK
    assert [r.seed for r in fake_generator.requests][:2] == [1001, 1002]
    capsys.readouterr()
    assert main(["generate", "--yes"]) == EXIT_OK
    assert "Nothing to do" in capsys.readouterr().out
    assert not (project / "output" / ".cardgen.lock").exists()


def test_generate_interactive_confirmation(project, fake_generator, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    assert main(["generate"]) == EXIT_OK
    assert fake_generator.loaded == 0
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    assert main(["generate", "--limit", "1"]) == EXIT_OK
    assert fake_generator.loaded == 1


def test_overwrite_needs_typed_confirmation(project, fake_generator, monkeypatch, capsys):
    main(["generate", "--yes", "--limit", "1"])
    image = project / "output" / "images" / "village_archer.png"
    before = image.read_bytes()
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "y")  # "y" is not enough
    assert main(["generate", "--overwrite", "--only", "village_archer"]) == EXIT_OK
    assert "Cancelled" in capsys.readouterr().out
    assert image.read_bytes() == before
    monkeypatch.setattr("builtins.input", lambda prompt: "overwrite")
    assert main(["generate", "--overwrite", "--only", "village_archer"]) == EXIT_OK
    assert list((project / "output" / "replaced").glob("village_archer-*.png"))


def test_unknown_only_id(project, fake_generator, capsys):
    assert main(["generate", "--yes", "--only", "nobody"]) == EXIT_INVALID
    assert "unknown card id" in capsys.readouterr().err


def test_failures_give_exit_code_1_and_retry(project, monkeypatch):
    from conftest import FakeGenerator
    failing = FakeGenerator(fail_on={"A jester": RuntimeError("nope")})
    monkeypatch.setattr("cardgen.generator.create_generator", lambda s: failing)
    assert main(["generate", "--yes"]) == EXIT_CARD_FAILURES
    fixed = FakeGenerator()
    monkeypatch.setattr("cardgen.generator.create_generator", lambda s: fixed)
    assert main(["generate", "--yes"]) == EXIT_OK and fixed.loaded == 0
    assert main(["generate", "--yes", "--retry-failed"]) == EXIT_OK
    assert Manifest.load(project / "output" / "manifest.json").status("court_jester") == "success"


def test_environment_failure_exit_code(project, monkeypatch, capsys):
    from conftest import FakeGenerator

    class NoModel(FakeGenerator):
        def load(self):
            raise ModelLoadError("model not in cache; use --allow-download")
    monkeypatch.setattr("cardgen.generator.create_generator", lambda s: NoModel())
    assert main(["generate", "--yes"]) == EXIT_ENVIRONMENT
    assert "--allow-download" in capsys.readouterr().out


def test_interrupt_exit_code(project, monkeypatch):
    from conftest import FakeGenerator
    gen = FakeGenerator(fail_on={"A peasant": KeyboardInterrupt()})
    monkeypatch.setattr("cardgen.generator.create_generator", lambda s: gen)
    assert main(["generate", "--yes"]) == EXIT_INTERRUPTED


def test_cli_overrides_reach_generator(project, fake_generator):
    main(["generate", "--yes", "--limit", "1", "--width", "512", "--height", "768", "--steps", "3",
          "--guidance-scale", "4.5"])
    r = fake_generator.requests[0]
    assert (r.width, r.height, r.steps, r.guidance_scale) == (512, 768, 3, 4.5)


def test_status(project, monkeypatch, capsys):
    from conftest import FakeGenerator
    gen = FakeGenerator(fail_on={"A peasant": RuntimeError("kaput")})
    monkeypatch.setattr("cardgen.generator.create_generator", lambda s: gen)
    main(["generate", "--yes", "--limit", "2"])
    capsys.readouterr()
    assert main(["status"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "failed: 1" in out and "pending: 1" in out and "success: 1" in out and "kaput" in out
    (project / "output" / "images" / "village_archer.png").unlink()
    main(["status"])
    assert "missing" in capsys.readouterr().out


def test_status_empty(project, capsys):
    assert main(["status", "--input", "nope.json"]) == EXIT_OK
    assert "No cards yet" in capsys.readouterr().out


def test_corrupt_manifest_blocks_generate(project, fake_generator, capsys):
    (project / "output").mkdir()
    (project / "output" / "manifest.json").write_text("{broken")
    assert main(["generate", "--yes"]) == EXIT_INVALID
    assert "unreadable" in capsys.readouterr().err
    assert (project / "output" / "manifest.json").read_text() == "{broken"


def test_postprocess_and_qc_commands(project, fake_generator, capsys):
    main(["generate", "--yes", "--limit", "1"])
    assert main(["postprocess"]) == EXIT_OK
    final = project / "output" / "final" / "village_archer.png"
    from PIL import Image
    with Image.open(final) as img:
        assert img.size == (1024, 1536)
    assert main(["qc"]) == EXIT_OK
    assert (project / "output" / "qc_report.csv").exists()
    assert "heuristic" in capsys.readouterr().out


def test_doctor_runs_without_downloading(project, capsys):
    code = main(["doctor"])
    out = capsys.readouterr().out
    assert code in (EXIT_OK, EXIT_ENVIRONMENT)
    assert "Python" in out and "Settings" in out and "Output folder" in out


@pytest.mark.parametrize("command", ["validate", "generate", "status", "doctor", "postprocess", "qc"])
def test_help(command, capsys):
    with pytest.raises(SystemExit) as info:
        main([command, "--help"])
    assert info.value.code == 0
    assert "usage" in capsys.readouterr().out
