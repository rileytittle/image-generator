# CLAUDE.md

Local batch text-to-image CLI (`cardgen`) for medieval trading-card character art.
Python 3.11, Hugging Face Diffusers + PyTorch (MPS on Apple Silicon), Pillow, pytest.

## Commands

```bash
pip install -e ".[gen,dev]"        # [gen] = torch/diffusers stack; core + tests need only Pillow
python -m pytest                   # mocked, fast, never downloads models
CARDGEN_SMOKE=1 python -m pytest tests/test_smoke.py -s   # one real image, cached model only
python -m cardgen validate | generate [--dry-run|--limit N|--retry-failed|--overwrite|--only ID] | status | doctor | postprocess | qc
```

Run from the project root: relative paths in `config/settings.json` resolve against the cwd.

## Architecture (src/cardgen)

- `config.py` – settings: defaults → file → preset → CLI overrides; `ConfigError` lists all problems.
- `models.py` – `Card`, JSON/CSV loading, ID rules (`^[a-z0-9][a-z0-9_-]{0,63}$`), `derive_seed` (sha256 of `global_seed:id`).
- `prompts.py` – style file + template composition. Future style hooks (IP-Adapter, LoRA) belong here / in `PromptSpec`.
- `generator.py` – `ImageGenerator` protocol; `DiffusersGenerator` (lazy torch import, `AutoPipelineForText2Image`,
  device/dtype resolution, chunked long-prompt encoding). `create_generator` is the factory tests monkeypatch.
  Errors with `fatal = True` stop the batch; others fail one card.
- `batch.py` – `plan_batch` (pure skip/generate decisions) and `run_batch` (sequential, saves manifest after each card).
- `manifest.py` – atomic JSON writes + `.bak`, `RunLock`.
- `image_utils.py` – atomic PNG save + verification, crop/pad resize, safe-zone heuristic.
- `cli.py` – argparse, confirmation, exit codes (0 ok, 1 card failures, 2 invalid input, 3 environment, 130 interrupt).

## Rules

- Tests must never load torch models or download anything; use `FakeGenerator` from `tests/conftest.py`.
- Never overwrite a valid image without `--overwrite` (old file is archived to `output/replaced/`).
- Never download models unless `allow_download`/`--allow-download`; never call remote inference APIs.
- Never silently drop invalid cards; never claim safe zones or style are enforced (they are prompt goals).
- Keep README commands in sync with the CLI.
