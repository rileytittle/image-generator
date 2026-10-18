# Medieval Trading Card Image Generator (`cardgen`)

A local, command-line batch generator for character illustrations for a medieval
trading card game. It reads a list of character prompts (`cards.json`), adds a
shared art direction (`config/style.txt`), and generates one PNG per card with a
local Stable Diffusion XL model via Hugging Face Diffusers and PyTorch.

- **Fully local.** No paid API, no API key. Prompts and images never leave your Mac.
  The network is only used to install packages and, when you explicitly allow it,
  to download the model once.
- **Resumable.** Each image is saved as soon as it finishes. A manifest records
  every card's settings and status. Re-running continues where it stopped and never
  overwrites a good image unless you ask it to.
- **Predictable.** One seed per card, logged with the model, prompt, size, steps and
  guidance scale.

---

## Contents

1. [Hardware requirements](#hardware-requirements)
2. [Installation](#installation)
3. [First run and model download](#first-run-and-model-download)
4. [Everyday use](#everyday-use)
5. [Cards file](#cards-file)
6. [Shared style and negative prompt](#shared-style-and-negative-prompt)
7. [Settings](#settings)
8. [Output files and the manifest](#output-files-and-the-manifest)
9. [Final 1024 × 1536 images](#final-1024--1536-images)
10. [Quality-control report](#quality-control-report)
11. [Troubleshooting](#troubleshooting)
12. [Licensing and commercial use](#licensing-and-commercial-use)
13. [Limitations](#limitations)
14. [Development and tests](#development-and-tests)

---

## Hardware requirements

| | Minimum | Recommended |
|---|---|---|
| Mac | Apple Silicon (M1 or later) | M1 Pro/Max, M2 or later |
| macOS | 12.3 (needed for PyTorch MPS) | 14 or later |
| Unified memory | 16 GB (use the `test` preset) | 24 GB or more for the `quality` preset |
| Free disk | ~12 GB (≈7 GB model + packages + images) | 20 GB |
| Python | 3.10 | 3.11 (arm64 build) |

- **Apple Silicon** uses the GPU through PyTorch's MPS backend. `cardgen doctor`
  reports whether MPS is available.
- **Intel Macs** have no MPS. Recent PyTorch releases also no longer publish
  wheels for Intel macOS, so this setup is not supported there.
- **CPU generation** works only when you ask for it (`--device cpu` or
  `"allow_cpu_fallback": true`). SDXL on a CPU takes several minutes or more per image.
- Python must be a **native arm64** build, not one running under Rosetta. Check with
  `python3 -c "import platform; print(platform.machine())"`, which should print `arm64`.
  `cardgen doctor` also detects Rosetta.

Rough timings on Apple Silicon at the `test` preset (512×768, 25 steps) are tens
of seconds per image on an M1/M2 Pro class machine. Larger sizes take
proportionally longer. Your numbers will vary; the manifest records the real
duration of each card.

---

## Installation

```bash
git clone https://github.com/rileytittle/image-generator.git
cd image-generator

# Use an arm64 Python 3.11 (python.org installer, Homebrew `python@3.11`, or pyenv).
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Core + local generation (PyTorch, Diffusers, Transformers, ...) + test tools
pip install -e ".[gen,dev]"
```

Package groups:

- core (`pip install -e .`) installs Pillow only. That is enough for `validate`,
  `generate --dry-run`, `status`, `postprocess`, `qc` and the automated tests.
- `[gen]` adds `torch`, `diffusers`, `transformers`, `accelerate`, `safetensors` and
  `huggingface_hub`. This download is roughly 1 GB.
- `[dev]` adds `pytest`.

On macOS, the default `pip install torch` already includes MPS support.

Check the environment (this never downloads a model):

```bash
python -m cardgen doctor
```

> All commands are run from the project root, because relative paths in
> `config/settings.json` (style file, output folder) are resolved against the current
> directory.

---

## First run and model download

The default model is
[`stabilityai/stable-diffusion-xl-base-1.0`](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0).
Its fp16 weights are about **7 GB**.

`cardgen` **never downloads a model unless you allow it.** By default it only loads
from the local Hugging Face cache (`~/.cache/huggingface/hub`, or the `cache_dir`
setting). To download it once:

```bash
python -m cardgen generate --limit 1 --allow-download
```

This downloads the model, then generates one test image. Later runs load from the
cache, and no flag is needed. If Hugging Face asks you to log in or accept terms, run
`hf auth login` (never put tokens in this project's files).

You can also pre-download it without generating anything:

```bash
hf download stabilityai/stable-diffusion-xl-base-1.0 \
  --include "*.json" "*.txt" "*fp16.safetensors"
```

---

## Everyday use

```bash
# 1. Check cards, settings and style (no model loaded)
python -m cardgen validate --input cards.json

# 2. See exactly what would run, including the full prompt (no model loaded, nothing written)
python -m cardgen generate --input cards.json --dry-run
python -m cardgen generate --input cards.json --dry-run --show-prompts

# 3. Generate a three-card test batch
python -m cardgen generate --input cards.json --limit 3

# 4. Generate everything that is still missing
python -m cardgen generate --input cards.json

# 5. Check progress
python -m cardgen status
```

`--input cards.json` is the default and can be omitted.

`generate` prints a plan and asks `Generate N image(s)? [y/N]` before it loads the
model. Add `--yes` (or `-y`) to skip the question, for example in scripts. Without
`--yes`, a non-interactive run refuses to start.

### Resuming an interrupted run

Press **Ctrl-C** at any time, or let the run crash. Then run the same command again:

```bash
python -m cardgen generate --input cards.json
```

Finished images are kept and skipped. Cards that were being generated when the run
stopped (status `interrupted` or `running`) are generated again. The manifest is
written atomically after every card, so an interruption cannot corrupt it.

### Retrying failures

Cards that failed are **not** retried automatically. This stops a broken prompt from
looping forever. To retry them:

```bash
python -m cardgen status                  # shows each failed card and its error
python -m cardgen generate --retry-failed
```

### Regenerating existing images

```bash
# Redo one card (e.g. after editing its prompt or seed)
python -m cardgen generate --overwrite --only court_jester

# Redo everything
python -m cardgen generate --overwrite
```

`--overwrite` asks you to type `overwrite` to confirm, unless you also pass `--yes`.
The previous image is copied to `output/replaced/<id>-<timestamp>.png` first.

If you change a card's prompt, its seed or the settings after an image exists,
`generate` reports the card as `done, but prompt/settings changed since`. It does
**not** regenerate it automatically; use `--overwrite --only <id>`.

### Other useful options

```bash
python -m cardgen generate --only village_archer,court_jester   # just these cards
python -m cardgen generate --preset quality                     # 832x1248, more steps
python -m cardgen generate --width 832 --height 1248 --steps 30 --guidance-scale 6
python -m cardgen generate --seed 777          # new global seed (cards without "seed")
python -m cardgen generate --device cpu        # explicit CPU run (slow)
python -m cardgen generate --model some/other-model
python -m cardgen generate --output-dir output-quality
python -m cardgen <command> --help
```

### Exit codes

| Code | Meaning |
|---|---|
| 0 | Success (or nothing to do / cancelled) |
| 1 | The run finished, but some cards failed (see `status`) |
| 2 | Invalid input: card file, settings, style file, arguments or manifest |
| 3 | Environment problem: missing dependency, no device, model not loadable, out of memory (`doctor` also returns 3 when a check fails) |
| 130 | Interrupted with Ctrl-C |

---

## Cards file

`cards.json` is a JSON list. Each card has:

| Field | Required | Notes |
|---|---|---|
| `id` | yes | 1–64 characters: lowercase letters, digits, `_`, `-`. Used as the filename (`<id>.png`). Must be unique. |
| `name` | no | Display name (defaults to `id`). Not drawn on the image. |
| `prompt` | yes | The character and scene description. |
| `rarity` | no | Free text such as `Common` or `Legendary`. For your own organisation only; it does not change the image. Recorded in the manifest. |
| `seed` | no | Integer 0 – 4294967295. Without it, a seed is derived from the global `default_seed` and the card's `id`. |
| `overrides` | no | Per-card `width`, `height`, `steps`, `guidance_scale`, `negative_prompt`. |

```json
[
  {
    "id": "village_archer",
    "name": "Village Archer",
    "prompt": "A young archer in a green hood squinting while drawing a bow. ...",
    "seed": 1001
  },
  {
    "id": "royal_cook",
    "name": "Royal Cook",
    "prompt": "A round, red-faced castle cook chasing an escaped roast chicken ...",
    "overrides": { "steps": 35, "negative_prompt": "photorealistic, text" }
  }
]
```

**Adding cards:** append entries, then run `python -m cardgen validate`. Validation
reports *every* problem at once, including malformed JSON (with line and column),
missing fields, duplicate or unsafe IDs (with a suggested safe ID), bad seeds and
unknown fields. Invalid cards are never dropped silently; nothing is generated until
the file is valid.

**Seeds:** a card with an explicit `seed` always keeps it. A derived seed depends
only on `default_seed` and the card's `id`. Adding, removing or reordering other
cards therefore never changes any card's seed. The actual seed is written to the
manifest and to the PNG's metadata.

**CSV:** you can also pass a `.csv` file with a header row. Required columns are
`id` and `prompt`. Optional columns are `name`, `seed`, `width`, `height`, `steps`,
`guidance_scale` and `negative_prompt`. Empty cells mean "not set".

```bash
python -m cardgen validate --input cards.csv
python -m cardgen generate --input cards.csv --dry-run
```

---

## Shared style and negative prompt

The art direction lives in **`config/style.txt`**. Edit it freely. Lines starting with
`#` are comments, and line breaks are joined into one paragraph.

Each card's final prompt is built from `prompt_template` in `config/settings.json`.
The default is:

```
{character}

{style}
```

The character comes first on purpose (see the token note below). See the exact
result with `generate --dry-run --show-prompts`.

The **negative prompt** (`negative_prompt` in settings, or per card in `overrides`)
discourages unwanted output: photorealism, text, watermarks, card frames, borders,
cropped characters, blur, extra limbs. Edit it to taste.

### The 77-token limit and `long_prompts`

SDXL's text encoders (CLIP) read at most **77 tokens** (roughly 50–60 words) at a
time and silently ignore the rest. The required style text alone is roughly 115–130
tokens. With a character description added, prompts reach roughly 150–190 tokens.

With `"long_prompts": true` (the default), `cardgen` encodes long prompts in
75-token chunks and joins them, the approach used by popular Stable Diffusion
front ends. The whole prompt then reaches the model. The manifest records the token
count and the mode used for each card (`generation_info.prompt_mode`).

- Text in later chunks still tends to have **less influence** than text near the
  start. Put the most important visual cues first in `style.txt`.
- With `"long_prompts": false`, everything after token 77 is cut off, and the log
  warns you. In that mode, keep `style.txt` short.
- Chunked encoding works for SDXL and SD 1.x/2.x pipelines. Other model families
  fall back to the pipeline's normal prompt handling.

---

## Settings

`config/settings.json` (validated on every command; all problems are listed together):

| Key | Default | Meaning |
|---|---|---|
| `model_id` | `stabilityai/stable-diffusion-xl-base-1.0` | Hugging Face model ID or a local folder. Loaded with `AutoPipelineForText2Image`, so other Diffusers text-to-image models work too. |
| `model_variant` | `"fp16"` | Weight variant to load (smaller download). Set to `null` for models without an fp16 variant. |
| `cache_dir` | `null` | Model cache folder (`null` = Hugging Face default). |
| `allow_download` | `false` | Allow downloading the model (`--allow-download` sets it for one run). |
| `device` | `"auto"` | `auto` (MPS if available), `mps`, or `cpu`. |
| `allow_cpu_fallback` | `false` | With `auto`, use the CPU when MPS is unavailable instead of stopping. |
| `dtype` | `"auto"` | `auto` = float16 on MPS, float32 on CPU. Also `float32`, `bfloat16`, `float16`. |
| `attention_slicing` | `"auto"` | Lower peak memory, slightly slower. `auto` = on for MPS. |
| `long_prompts` | `true` | Chunked encoding for prompts over 77 tokens (see above). |
| `preset` / `presets` | `"test"` | Named size/steps/guidance sets. `test` is 512×768 at 25 steps; `quality` is 832×1248 at 35 steps. |
| `default_seed` | `12345` | Global seed used to derive seeds for cards without one. |
| `style_file` | `config/style.txt` | Shared style text. |
| `prompt_template` | `"{character}\n\n{style}"` | How card prompt and style are combined. |
| `negative_prompt` | (see file) | Default negative prompt. |
| `output_dir` | `output` | Where images, manifest and logs go. |
| `image_format` | `png` | Only PNG is supported. |
| `existing_outputs` | `"skip"` | `skip` keeps valid images. `overwrite` behaves like `--overwrite` (still asks to confirm). |
| `retry_failed` | `false` | Behave as if `--retry-failed` were always given. |
| `max_consecutive_failures` | `3` | Stop after this many failures in a row (probably a systemic problem). `0` = never. |
| `safe_zones` | top `0.12`, bottom `0.15` | Banner areas used by the QC report. |
| `postprocess` | disabled, 1024×1536 | Final-size resize (see below). |

Precedence: built-in defaults → top-level keys → selected preset → command-line flags.

**Changing output dimensions:** pick a preset (`--preset quality`), edit or add a preset
in `settings.json`, or pass `--width/--height`. Width and height must be multiples of
8, between 256 and 2048. A non-2:3 size is allowed, with a warning.

SDXL was trained around 1024×1024 pixels (about 1 megapixel):

- 512×768 is fast but noticeably less detailed, so use it for testing prompts.
- 832×1248 (≈1.04 MP, exactly 2:3) is the recommended quality size.
- Native 1024×1536 is beyond SDXL's training sizes and more likely to produce
  duplicated or stretched anatomy. Use post-processing instead.

---

## Output files and the manifest

```
output/
  images/<id>.png          # original generated image (never modified)
  final/<id>.png           # optional post-processed 1024x1536 version
  replaced/<id>-<time>.png # previous images saved by --overwrite
  manifest.json            # status + settings for every card (manifest.json.bak = previous copy)
  logs/run-<time>.log      # full log of each generate run
  qc_report.csv            # from `cardgen qc`
```

- Every image is written to a temporary file and checked: it must open completely
  and have the expected dimensions. Only then is it renamed into place, so you never
  get a half-written `<id>.png`. The prompt, seed, model, steps and guidance are also
  embedded as PNG text metadata.
- Blank (single-colour) results are treated as failures. They usually mean
  numerical overflow in half precision.
- Each `manifest.json` card entry contains: `id`, `name`, `prompt`, `full_prompt`,
  `negative_prompt`, `seed`, `seed_source`, `model_id`, `model_variant`, `settings`,
  `status`, `output_path`, `final_path`, `started_at`, `completed_at`, `duration_s`,
  `attempts`, `error` (type and message), `fingerprint`, and `generation_info`
  (token count, prompt mode, time). Each run also adds an entry under `runs` with
  the device, dtype and library versions.
- Card status is one of `running`, `success`, `failed` or `interrupted`. Cards not
  yet attempted show as `pending` in `status`.
- If the manifest is unreadable, `cardgen` stops and does **not** overwrite it.
  Restore `manifest.json.bak`, or move the broken file aside. Existing valid images
  are still detected and skipped without a manifest.
- A lock file (`output/.cardgen.lock`) prevents two runs from writing the same
  folder. A lock left by a crashed run is removed automatically.

**Reproducibility:** the same seed and settings usually give the same image on the
same machine and software. Identical seeds are **not** guaranteed to match across
different model revisions, Diffusers/PyTorch versions, or hardware backends (MPS vs
CPU). The manifest records the library versions of each run for this reason.

---

## Final 1024 × 1536 images

Generate at a size SDXL handles well (e.g. `--preset quality`, 832×1248), then resize:

```bash
python -m cardgen postprocess            # writes output/final/<id>.png
python -m cardgen postprocess --force    # redo ones already processed
```

Or set `"postprocess": {"enabled": true, ...}` to do it automatically after each card.

- Resizing uses high-quality Lanczos resampling. It is **not** an AI upscaler: it
  enlarges the image but adds no new detail. From 832×1248 the result looks good;
  from 512×768 it looks soft.
- Images are **never stretched**. If the aspect ratios differ, `"fit": "crop"` scales
  to cover the target and trims the edges equally. `"fit": "pad"` scales to fit and
  fills the borders with `pad_color`. For 2:3 sources and a 2:3 target, both are a
  plain resize.
- Originals in `output/images/` are never modified.

An AI upscaler (for example Real-ESRGAN, or an SDXL img2img refine pass) could be added
later as another post-processing step. Check its license first.

---

## Quality-control report

```bash
python -m cardgen qc
```

This writes `output/qc_report.csv`. For every successful image it checks:

- that the image is valid,
- its dimensions and whether it is 2:3,
- the **edge density** (a rough measure of visual detail) in the top 12% and bottom
  15% compared with the middle.

Images whose banner zones look busier than the middle are listed for review.

**This is a heuristic, not a verification.** It cannot recognise faces, hands or
weapons. A plain sky can pass and a detailed floor can be flagged. Always look at
the images yourself.

---

## Troubleshooting

**Run `python -m cardgen doctor` first.** It checks Python, the platform (including
Rosetta), RAM, packages, MPS, settings, style file, whether the model is cached, free
disk space and output-folder permissions.

### Out of memory (`MPS backend out of memory`)

A memory error stops the batch (exit code 3), because the next card would fail too.
Finished images are kept.

- Use a smaller size (`--preset test`) and close other apps, browsers especially.
- Keep `"attention_slicing": "auto"` (or set it to `true`).
- Do not run two generations at once. `cardgen` is sequential by design.
- As a last resort, set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` before running. This
  disables PyTorch's MPS memory cap, can make macOS swap heavily, and may freeze the
  system, so use it with care.

### MPS errors

- **`MPS is built but unavailable`**: Python is running under Rosetta, or macOS is
  older than 12.3. Install an arm64 Python and recreate the venv.
- **`PyTorch was not built with MPS support`**: reinstall with `pip install --force-reinstall torch`.
- **An op not implemented for MPS**: try `export PYTORCH_ENABLE_MPS_FALLBACK=1`. Ops
  that MPS lacks then run on the CPU, which is slower.
- **Black or blank images** (reported as `BlankImageError`): try another seed, or set
  `"dtype": "float32"`. float32 uses about twice the memory and is slower.
- To run on the CPU deliberately, use `--device cpu`. It is very slow.

### Model not found / download problems

- `could not load ... If the model has not been downloaded yet, run once with
  --allow-download`: the model is not in the local cache. See
  [First run](#first-run-and-model-download).
- `... has no 'fp16' weight variant`: set `"model_variant": null` for that model.
- Behind a proxy or offline: download the model elsewhere and copy the Hugging Face
  cache folder, or point `model_id` at a local folder.

### Dependency conflicts

- Use a fresh virtual environment: `rm -rf .venv && python3.11 -m venv .venv && source .venv/bin/activate && pip install -e ".[gen,dev]"`.
- `pip check` lists incompatible packages.
- If a newer Diffusers or Transformers release breaks something, pin known-good
  versions. `cardgen doctor` prints the versions in use. Example:
  `pip install "diffusers==0.41.0" "transformers==5.20.1"`.

### Not enough disk space

The SDXL fp16 weights need ~7 GB. Packages add ~1–2 GB, and each 832×1248 PNG is
roughly 1–2 MB. To free or move space:

- Run `hf cache ls` to see cached models and `hf cache rm model/<id>` to remove ones you no longer use.
- Set `cache_dir` (or `HF_HOME`) to a folder on a larger or external drive.

### Other

- **"another cardgen run appears to be using this output folder"**: another run is
  active. If none is, delete `output/.cardgen.lock`.
- **"not running interactively: pass --yes"**: add `--yes` when running from a script.

---

## Licensing and commercial use

*This is practical guidance, not legal advice.*

- **Default model:** `stabilityai/stable-diffusion-xl-base-1.0` is released under the
  **CreativeML Open RAIL++-M License**. Official text:
  <https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md>.
  - It permits commercial use of the model and of the images you generate. Stability
    AI claims no rights in your outputs.
  - It includes **use-based restrictions** (Attachment A). Examples are illegal
    content, harming minors, defamation, and harmful misinformation. You are
    responsible for how outputs are used.
  - If you redistribute the model itself, you must pass these restrictions on. Using
    it locally to make game art does not redistribute it.
  - Read the full license before commercial release. Licenses can change, so check
    the version that applies to the weights you downloaded.
- **Other models, LoRAs, adapters, upscalers** have **their own licenses**. "Open"
  or "downloadable" does not mean commercially unrestricted. For example, several
  newer Stability AI models use a community license with revenue thresholds, and many
  community LoRAs forbid commercial use. Before using any of them, record the model
  ID, license name and license link in the table below:

  | Component | Used for | License | Link |
  |---|---|---|---|
  | stabilityai/stable-diffusion-xl-base-1.0 | Base model (default) | CreativeML Open RAIL++-M | [LICENSE.md](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md) |
  | *(none yet)* | LoRA / IP-Adapter / upscaler | | |

- **Other rights:** commercial use is also subject to trademark and character rights.
  Don't prompt for existing franchises, characters, artists' names or logos. Also
  check the AI-content rules of any platform you publish on.
- **Software:** Diffusers, Transformers and Accelerate are Apache-2.0; PyTorch is
  BSD-style; Pillow uses the MIT-CMU (HPND) license.
- **Privacy:** all inference runs locally. `cardgen` never sends prompts or images to
  any service. It downloads nothing unless you pass `--allow-download` or set
  `allow_download`, and it needs no API keys. If a Hugging Face token is ever needed,
  it lives in `~/.cache/huggingface/` via `hf auth login`, never in this
  project.

---

## Limitations

What the prompt asks for are **goals, not guarantees**:

- **Style consistency** comes only from the shared prompt and negative prompt.
  Images in the same batch will vary in line weight, palette and rendering. Text in
  later prompt chunks has less influence. The most reliable way to lock in a style is
  a custom LoRA or a style-reference adapter (IP-Adapter). Neither is implemented
  yet: the prompt module (`src/cardgen/prompts.py`) and `GenerationRequest` already
  have a `reference_images` slot for it. Even then, a reference image guides the
  style; it does not enforce it.
- **Safe zones** (top 12%, bottom 15%) are requested in the prompt. Diffusion models
  follow layout instructions loosely. The QC report is a heuristic hint, and manual
  review is still needed.
- **Composition** (centred, single character, filling the frame, 2:3) is likewise
  encouraged, not enforced. The output size is exact.
- **"No text"**: SDXL sometimes draws letter-like squiggles anyway. The negative
  prompt reduces this but cannot rule it out.
- **Reproducibility** holds only for the same model revision, library versions and
  hardware backend.
- **Throughput**: generation is strictly sequential on one device.

Practical workflow: iterate on prompts at the `test` preset with `--limit 3`. Pick
the seeds you like and copy them into `cards.json`. Then render the full set with
`--preset quality --output-dir output-quality` and post-process.

---

## Development and tests

```bash
pip install -e ".[dev]"     # tests need only Pillow + pytest (no torch, no model)
python -m pytest            # mocked generator, runs in seconds, never downloads
```

The automated tests replace the image pipeline with a fake generator. They cover
card and settings validation, filename safety, prompt construction, seed stability,
manifest writes and resume, skipping valid outputs, corrupt or missing images,
retries, failure handling, interrupts, overwrite archiving, post-processing, the QC
report and the CLI commands.

**Opt-in smoke test.** This generates one real 512×768 image with an
**already downloaded** model (it never downloads):

```bash
CARDGEN_SMOKE=1 python -m pytest tests/test_smoke.py -s
# optional: CARDGEN_SMOKE_STEPS=20 CARDGEN_SMOKE_DEVICE=mps CARDGEN_SMOKE_MODEL=<id or folder>
#           CARDGEN_SMOKE_VARIANT=   (empty = no fp16 variant)
```

Project layout:

```
src/cardgen/
  cli.py          argparse commands, confirmation, exit codes
  config.py       settings loading, presets, validation
  models.py       Card model, card-file loading (JSON/CSV), ID safety, seeds
  prompts.py      style loading and prompt composition (future style hooks live here)
  generator.py    ImageGenerator interface + Diffusers implementation, device/dtype choice
  batch.py        planning (skip/generate decisions), sequential runner, post-process, QC
  manifest.py     atomic manifest writes, run lock
  image_utils.py  PNG save/verify, crop/pad resize, safe-zone heuristic
  diagnostics.py  `doctor` checks
```
