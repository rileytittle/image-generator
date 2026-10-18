"""Command-line interface: ``python -m cardgen <command> --help``."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from . import generator as generator_module
from .batch import GENERATE, PlanOptions, plan_batch, postprocess_file, qc_rows, run_batch
from .config import DEFAULT_SETTINGS_PATH, ConfigError, Settings, load_settings
from .diagnostics import FAIL, OK, WARN, run_checks
from .image_utils import check_image
from .manifest import FAILED, INTERRUPTED, PENDING, RUNNING, SUCCESS, Manifest, ManifestError, RunLock
from .models import CardSet, CardValidationError, load_cards
from .prompts import CLIP_TOKEN_LIMIT, StyleError, compose_prompt, estimate_tokens, load_style

EXIT_OK = 0
EXIT_CARD_FAILURES = 1      # the run finished but some cards failed
EXIT_INVALID = 2            # bad arguments, settings, cards or manifest
EXIT_ENVIRONMENT = 3        # missing dependency, no device, model could not load, out of memory
EXIT_INTERRUPTED = 130      # Ctrl-C

DEFAULT_INPUT = Path("cards.json")

log = logging.getLogger("cardgen")


class UserError(Exception):
    """A problem to report briefly to the user with an exit code."""

    def __init__(self, message: str, code: int = EXIT_INVALID):
        super().__init__(message)
        self.code = code


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, default=DEFAULT_SETTINGS_PATH,
                        help=f"settings file (default: {DEFAULT_SETTINGS_PATH})")
    common.add_argument("--preset", help="settings preset to use, e.g. 'test' or 'quality'")
    common.add_argument("-v", "--verbose", action="store_true", help="show debug logging")

    with_input = argparse.ArgumentParser(add_help=False)
    with_input.add_argument("--input", "-i", type=Path, default=DEFAULT_INPUT,
                            help=f"card file, .json or .csv (default: {DEFAULT_INPUT})")

    overrides = argparse.ArgumentParser(add_help=False)
    group = overrides.add_argument_group("setting overrides (take precedence over the settings file)")
    group.add_argument("--model", dest="model_id", help="Hugging Face model ID or local folder")
    group.add_argument("--device", choices=["auto", "mps", "cpu"])
    group.add_argument("--width", type=int)
    group.add_argument("--height", type=int)
    group.add_argument("--steps", type=int)
    group.add_argument("--guidance-scale", type=float)
    group.add_argument("--seed", dest="default_seed", type=int,
                       help="global seed used to derive seeds for cards without one")
    group.add_argument("--output-dir")

    parser = argparse.ArgumentParser(
        prog="cardgen", description="Generate medieval trading-card character art locally.")
    parser.add_argument("--version", action="version", version=f"cardgen {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    p = sub.add_parser("validate", parents=[common, with_input, overrides],
                       help="check cards and settings without loading the model",
                       description="Validate the card file, settings and style file. Never loads the model.")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("generate", parents=[common, with_input, overrides],
                       help="generate missing images",
                       description="Generate images for cards that do not have a valid image yet. "
                                   "Existing valid images are never replaced unless --overwrite is given.")
    p.add_argument("--limit", type=int, help="generate at most N images this run")
    p.add_argument("--only", action="append", default=[], metavar="ID",
                   help="only consider these card IDs (repeatable or comma-separated)")
    p.add_argument("--dry-run", action="store_true",
                   help="show the plan without loading the model or writing anything")
    p.add_argument("--retry-failed", action="store_true", help="also regenerate cards that failed before")
    p.add_argument("--overwrite", action="store_true",
                   help="regenerate even existing valid images (asks for confirmation; old files are "
                        "copied to output/replaced/)")
    p.add_argument("--yes", "-y", action="store_true", help="do not ask for confirmation")
    p.add_argument("--allow-download", action="store_true", default=None,
                   help="allow downloading the model from Hugging Face if it is not cached")
    p.add_argument("--show-prompts", action="store_true", help="print the full prompt for every card")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("status", parents=[common, with_input],
                       help="summarise completed, pending and failed cards",
                       description="Summarise the manifest. If the card file exists, cards not yet "
                                   "attempted are listed as pending.")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("doctor", parents=[common],
                       help="check Python, dependencies, device, disk and model cache",
                       description="Diagnose the environment. Does not download anything.")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("postprocess", parents=[common],
                       help="resize finished images to the final card size",
                       description="Resize successful images to postprocess.width x height "
                                   "(crop or pad, never stretch) into output/<output_subdir>/. "
                                   "Originals are never modified.")
    p.add_argument("--force", action="store_true", help="redo images that were already post-processed")
    p.set_defaults(func=cmd_postprocess)

    p = sub.add_parser("qc", parents=[common],
                       help="write a heuristic quality-control report",
                       description="Check dimensions/aspect ratio and estimate detail in the banner "
                                   "safe zones. The safe-zone numbers are a heuristic, not a guarantee.")
    p.set_defaults(func=cmd_qc)
    return parser


def setting_overrides(args: argparse.Namespace) -> dict[str, Any]:
    keys = ("model_id", "device", "width", "height", "steps", "guidance_scale",
            "default_seed", "output_dir", "allow_download")
    return {k: getattr(args, k) for k in keys if getattr(args, k, None) is not None}


# --------------------------------------------------------------------------- #
# Shared loading
# --------------------------------------------------------------------------- #


def load_all(args: argparse.Namespace, *, need_cards: bool = True
             ) -> tuple[Settings, str, CardSet | None]:
    errors: list[str] = []
    settings = style = cards = None
    try:
        settings = load_settings(args.config, preset=args.preset, overrides=setting_overrides(args))
    except ConfigError as exc:
        errors += exc.errors
    if settings:
        try:
            style = load_style(settings.style_file)
        except StyleError as exc:
            errors.append(str(exc))
    if need_cards:
        try:
            cards = load_cards(args.input)
        except CardValidationError as exc:
            errors += exc.errors
    if errors:
        raise UserError("\n".join(["Found problems:"] + [f"  - {e}" for e in errors]))
    for warning in [*settings.warnings, *(cards.warnings if cards else [])]:
        print(f"warning: {warning}")
    return settings, style, cards


def describe_settings(settings: Settings) -> str:
    return (f"model {settings.model_id} | preset {settings.preset or 'none'} | "
            f"{settings.width}x{settings.height} | {settings.steps} steps | "
            f"guidance {settings.guidance_scale} | device {settings.device}")


def prompt_length_note(settings: Settings, style: str, cards: CardSet) -> str | None:
    longest = max(estimate_tokens(compose_prompt(c.prompt, style, settings.prompt_template))
                  for c in cards.cards)
    if longest <= CLIP_TOKEN_LIMIT:
        return None
    if settings.long_prompts:
        return (f"note: prompts are up to ~{longest} tokens (CLIP reads {CLIP_TOKEN_LIMIT} at a time); "
                "long_prompts is on, so they will be encoded in chunks")
    return (f"warning: prompts are up to ~{longest} tokens but the text encoder reads only "
            f"{CLIP_TOKEN_LIMIT}; the end of each prompt (usually the style) will be ignored. "
            "Enable long_prompts or shorten config/style.txt")


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def cmd_validate(args: argparse.Namespace) -> int:
    settings, style, cards = load_all(args)
    note = prompt_length_note(settings, style, cards)
    if note:
        print(note)
    seeded = sum(1 for c in cards.cards if c.seed is not None)
    print(f"OK: {len(cards.cards)} cards in {args.input} ({seeded} with explicit seeds, "
          f"{len(cards.cards) - seeded} derived from global seed {settings.default_seed})")
    print(f"Settings: {describe_settings(settings)}")
    print(f"Style: {settings.style_file} ({len(style.split())} words)")
    return EXIT_OK


def _split_ids(values: list[str]) -> frozenset[str]:
    return frozenset(i.strip() for v in values for i in v.split(",") if i.strip())


def cmd_generate(args: argparse.Namespace) -> int:
    if args.limit is not None and args.limit < 1:
        raise UserError("--limit must be at least 1")
    settings, style, cards = load_all(args)
    only = _split_ids(args.only)
    unknown = sorted(only - {c.id for c in cards.cards})
    if unknown:
        raise UserError(f"--only: unknown card id(s): {', '.join(unknown)}")
    try:
        manifest = Manifest.load(settings.manifest_path)
    except ManifestError as exc:
        raise UserError(str(exc)) from exc

    options = PlanOptions(
        retry_failed=args.retry_failed or settings.retry_failed,
        overwrite=args.overwrite or settings.existing_outputs == "overwrite",
        limit=args.limit, only=only)
    jobs = plan_batch(cards.cards, settings, style, manifest, options)
    todo = [j for j in jobs if j.action == GENERATE]

    print(f"Settings: {describe_settings(settings)}")
    print(f"Output:   {settings.images_dir}")
    note = prompt_length_note(settings, style, cards)
    if note:
        print(note)
    print()
    width = max(len(j.card.id) for j in jobs) if jobs else 10
    for job in jobs:
        mark = "GENERATE" if job.action == GENERATE else "skip    "
        r = job.request
        print(f"  {mark}  {job.card.id:<{width}}  seed {r.seed:<10}  {r.width}x{r.height}  {job.reason}")
        if args.show_prompts:
            print(f"            prompt: {r.prompt}\n            negative: {r.negative_prompt}")
    print(f"\n{len(todo)} to generate, {len(jobs) - len(todo)} skipped, {len(cards.cards)} cards total.")

    if args.dry_run:
        if todo and not args.show_prompts:
            print(f"\nExample full prompt ({todo[0].card.id}):\n{todo[0].request.prompt}")
            print(f"Negative prompt:\n{todo[0].request.negative_prompt or '(none)'}")
        print("\nDry run: nothing was loaded or written.")
        return EXIT_OK
    if not todo:
        print("Nothing to do.")
        return EXIT_OK

    replacing = [j.card.id for j in todo if j.replaces_existing]
    if not args.yes:
        if not sys.stdin.isatty():
            raise UserError("not running interactively: pass --yes to confirm the run")
        if replacing:
            print(f"\n{len(replacing)} existing image(s) will be replaced (old copies go to "
                  f"{settings.output_dir / 'replaced'}): {', '.join(replacing)}")
            if input("Type 'overwrite' to confirm: ").strip().lower() != "overwrite":
                print("Cancelled.")
                return EXIT_OK
        elif input(f"\nGenerate {len(todo)} image(s)? [y/N] ").strip().lower() not in ("y", "yes"):
            print("Cancelled.")
            return EXIT_OK

    try:
        lock = RunLock(settings.output_dir).__enter__()
    except ManifestError as exc:
        raise UserError(str(exc)) from exc
    log_path = _add_file_log(settings)
    try:
        print(f"Log file: {log_path}")
        generator = generator_module.create_generator(settings)
        summary = run_batch(jobs, settings, manifest, generator, progress=_progress)
    finally:
        lock.__exit__(None, None, None)

    print()
    print(f"Done in {summary.elapsed_s / 60:.1f} min: {len(summary.succeeded)} generated, "
          f"{len(summary.failed)} failed, {summary.skipped} skipped"
          + (f", {summary.not_reached} not reached" if summary.not_reached else "") + ".")
    if summary.failed:
        print(f"Failed: {', '.join(summary.failed)}  (details: python -m cardgen status; "
              "retry: python -m cardgen generate --retry-failed)")
    if summary.interrupted:
        print("Interrupted. Run the same command again to resume; finished images are kept.")
        return EXIT_INTERRUPTED
    if summary.aborted:
        print(f"Stopped early: {summary.aborted}")
        return EXIT_ENVIRONMENT if summary.fatal else EXIT_CARD_FAILURES
    return EXIT_CARD_FAILURES if summary.failed else EXIT_OK


def cmd_status(args: argparse.Namespace) -> int:
    settings, _, _ = load_all(args, need_cards=False)
    try:
        manifest = Manifest.load(settings.manifest_path)
    except ManifestError as exc:
        raise UserError(str(exc)) from exc

    rows: dict[str, str] = {}
    problems: dict[str, str] = {}
    for card_id, entry in manifest.cards.items():
        status = entry.get("status", PENDING)
        if status == SUCCESS:
            check = check_image(Path(entry.get("output_path", "")))
            if not check.ok:
                status, problems[card_id] = "missing", f"image {check.reason}"
        elif status in (FAILED, INTERRUPTED, RUNNING):
            err = entry.get("error") or {}
            problems[card_id] = err.get("message", "run did not finish")
        rows[card_id] = status

    in_input = None
    if args.input.exists():
        try:
            in_input = load_cards(args.input)
        except CardValidationError as exc:
            print(f"warning: {args.input} has {len(exc.errors)} problem(s); run validate. "
                  "Showing manifest only.")
    if in_input:
        for card in in_input.cards:
            if card.id not in rows:
                exists = check_image(settings.images_dir / f"{card.id}.png").ok
                rows[card.id] = "untracked image" if exists else PENDING

    if not rows:
        print(f"No cards yet (manifest: {settings.manifest_path}).")
        return EXIT_OK
    counts = Counter(rows.values())
    print(f"Manifest: {settings.manifest_path}")
    print("  " + ", ".join(f"{status}: {n}" for status, n in sorted(counts.items())))
    width = max(len(i) for i in rows)
    for card_id in sorted(rows):
        line = f"  {rows[card_id]:<16} {card_id:<{width}}"
        if card_id in problems:
            line += f"  {problems[card_id]}"
        print(line)
    if in_input is not None:
        extra = sorted(set(manifest.cards) - {c.id for c in in_input.cards})
        if extra:
            print(f"Note: in the manifest but not in {args.input}: {', '.join(extra)}")
    return EXIT_OK


def cmd_doctor(args: argparse.Namespace) -> int:
    checks = run_checks(args.config, preset=args.preset)
    symbol = {OK: "[ ok ]", WARN: "[warn]", FAIL: "[FAIL]"}
    for check in checks:
        print(f"{symbol[check.status]} {check.name}: {check.detail}")
    failed = [c for c in checks if c.status == FAIL]
    print()
    print(f"{len(failed)} problem(s) found." if failed else "No blocking problems found.")
    return EXIT_ENVIRONMENT if failed else EXIT_OK


def cmd_postprocess(args: argparse.Namespace) -> int:
    settings, _, _ = load_all(args, need_cards=False)
    try:
        manifest = Manifest.load(settings.manifest_path)
    except ManifestError as exc:
        raise UserError(str(exc)) from exc
    pp = settings.postprocess
    done = failed = skipped = 0
    with RunLock(settings.output_dir):
        for card_id, entry in sorted(manifest.cards.items()):
            if entry.get("status") != SUCCESS:
                continue
            source = Path(entry["output_path"])
            target = settings.final_dir / source.name
            if not check_image(source).ok:
                print(f"  skip {card_id}: original image is missing or invalid")
                failed += 1
                continue
            if target.exists() and not args.force and check_image(target, (pp.width, pp.height)).ok:
                skipped += 1
                continue
            try:
                path = postprocess_file(source, settings)
            except Exception as exc:
                print(f"  FAILED {card_id}: {exc}")
                manifest.update(card_id, postprocess_error=str(exc))
                failed += 1
            else:
                manifest.update(card_id, final_path=str(path), postprocess_error=None)
                done += 1
        manifest.save()
    print(f"Post-processed {done} image(s) to {pp.width}x{pp.height} ({pp.fit}) in {settings.final_dir}; "
          f"{skipped} already done, {failed} failed.")
    return EXIT_CARD_FAILURES if failed else EXIT_OK


def cmd_qc(args: argparse.Namespace) -> int:
    settings, _, _ = load_all(args, need_cards=False)
    try:
        manifest = Manifest.load(settings.manifest_path)
    except ManifestError as exc:
        raise UserError(str(exc)) from exc
    rows = qc_rows(manifest, settings)
    if not rows:
        print("No successful images to check yet.")
        return EXIT_OK
    report = settings.output_dir / "qc_report.csv"
    fields = sorted({k for row in rows for k in row}, key=lambda k: (k not in ("id", "review"), k))
    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    flagged = [r for r in rows if r["review"]]
    print(f"Checked {len(rows)} image(s); report: {report}")
    print(f"Safe zones: top {settings.safe_top:.0%}, bottom {settings.safe_bottom:.0%}. "
          "Ratios compare edge detail in each zone with the middle; >1.0 is flagged for review.")
    print("This is a heuristic hint, not a verification - always look at the images.")
    for row in flagged:
        why = row.get("problem") or f"top {row['top_ratio']}, bottom {row['bottom_ratio']}"
        print(f"  review {row['id']}: {why}")
    return EXIT_OK


# --------------------------------------------------------------------------- #
# Plumbing
# --------------------------------------------------------------------------- #


def _progress(message: str) -> None:
    log.info(message)


def _add_file_log(settings: Settings) -> Path:
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    path = settings.logs_dir / f"run-{datetime.now():%Y%m%d-%H%M%S}.log"
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(handler)
    return path


def _setup_logging(verbose: bool) -> None:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    if not any(getattr(h, "_cardgen_console", False) for h in root.handlers):
        console = logging.StreamHandler(sys.stdout)
        console._cardgen_console = True  # type: ignore[attr-defined]
        console.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(console)
    for handler in root.handlers:
        if getattr(handler, "_cardgen_console", False):
            handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    # Third-party libraries are noisy at INFO/DEBUG.
    for name in ("urllib3", "filelock", "PIL", "diffusers", "transformers", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.WARNING)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    try:
        return args.func(args)
    except UserError as exc:
        print(str(exc), file=sys.stderr)
        return exc.code
    except ManifestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INVALID
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return EXIT_INTERRUPTED
