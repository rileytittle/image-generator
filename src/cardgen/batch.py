"""Batch planning and execution.

``plan_batch`` decides, without touching the model, what happens to every card
(generate or skip, and why). ``run_batch`` loads the model once and processes
the planned cards sequentially, saving each image and the manifest as it goes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

from PIL import Image

from . import __version__
from .config import Settings
from .generator import GenerationRequest, GeneratorError, ImageGenerator
from .image_utils import check_image, fit_image, safe_zone_report, save_png_atomic
from .manifest import FAILED, INTERRUPTED, RUNNING, SUCCESS, Manifest, now_iso
from .models import Card, output_filename, resolve_seed
from .prompts import build_prompt

log = logging.getLogger(__name__)

GENERATE = "generate"
SKIP = "skip"


@dataclass(frozen=True)
class PlanOptions:
    retry_failed: bool = False
    overwrite: bool = False
    limit: int | None = None
    only: frozenset[str] = frozenset()


@dataclass(frozen=True)
class CardJob:
    card: Card
    request: GenerationRequest
    output_path: Path
    fingerprint: str
    seed_source: str
    action: str
    reason: str
    replaces_existing: bool = False


@dataclass
class BatchSummary:
    planned: int = 0
    succeeded: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    skipped: int = 0
    interrupted: bool = False
    aborted: str | None = None
    fatal: bool = False  # aborted by an environment problem (dependency, device, model, memory)
    elapsed_s: float = 0.0

    @property
    def not_reached(self) -> int:
        return self.planned - len(self.succeeded) - len(self.failed)


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #


def build_request(card: Card, settings: Settings, style: str) -> GenerationRequest:
    spec = build_prompt(card, style, settings)
    o = card.overrides
    return GenerationRequest(
        prompt=spec.prompt,
        negative_prompt=spec.negative_prompt,
        width=o.get("width", settings.width),
        height=o.get("height", settings.height),
        steps=o.get("steps", settings.steps),
        guidance_scale=float(o.get("guidance_scale", settings.guidance_scale)),
        seed=resolve_seed(card, settings.default_seed),
        reference_images=spec.reference_images,
    )


def fingerprint(settings: Settings, request: GenerationRequest) -> str:
    """Hash of everything that determines the image, to spot outdated results."""
    payload = {
        "model_id": settings.model_id, "variant": settings.model_variant,
        "prompt": request.prompt, "negative": request.negative_prompt,
        "w": request.width, "h": request.height, "steps": request.steps,
        "cfg": request.guidance_scale, "seed": request.seed,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def plan_batch(cards: Iterable[Card], settings: Settings, style: str, manifest: Manifest,
               options: PlanOptions = PlanOptions()) -> list[CardJob]:
    jobs: list[CardJob] = []
    to_generate = 0
    for card in cards:
        if options.only and card.id not in options.only:
            continue
        request = build_request(card, settings, style)
        path = settings.images_dir / output_filename(card.id)
        fp = fingerprint(settings, request)
        action, reason, replaces = _decide(card, request, path, fp, manifest, options)
        if action == GENERATE and options.limit is not None and to_generate >= options.limit:
            action, reason, replaces = SKIP, f"beyond --limit {options.limit}", False
        if action == GENERATE:
            to_generate += 1
        jobs.append(CardJob(
            card=card, request=request, output_path=path, fingerprint=fp,
            seed_source="card" if card.seed is not None else "derived",
            action=action, reason=reason, replaces_existing=replaces))
    return jobs


def _decide(card: Card, request: GenerationRequest, path: Path, fp: str, manifest: Manifest,
            options: PlanOptions) -> tuple[str, str, bool]:
    entry = manifest.get(card.id) or {}
    status = entry.get("status")
    check = check_image(path, (request.width, request.height))
    readable = check.ok or (check.size is not None and check.reason.startswith("size"))

    if options.overwrite:
        return GENERATE, "overwrite requested", readable
    if check.ok:
        if status == SUCCESS:
            if entry.get("fingerprint") and entry["fingerprint"] != fp:
                return SKIP, "done, but prompt/settings changed since (use --overwrite --only to redo)", False
            return SKIP, "already generated", False
        return SKIP, "valid image already exists (not overwritten)", False
    if readable:
        # A good image at a different size, e.g. after changing presets. Never clobber it.
        return SKIP, f"existing image is {check.size[0]}x{check.size[1]}, settings say " \
                     f"{request.width}x{request.height} (use --overwrite to replace)", False
    if check.exists:
        return GENERATE, f"existing file is invalid: {check.reason}", False
    if status == FAILED:
        if options.retry_failed:
            return GENERATE, "retrying previous failure", False
        return SKIP, "failed previously (use --retry-failed)", False
    if status == SUCCESS:
        return GENERATE, "output file missing", False
    if status in (RUNNING, INTERRUPTED):
        return GENERATE, "previous run was interrupted", False
    return GENERATE, "pending", False


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #


Progress = Callable[[str], None]


def run_batch(jobs: list[CardJob], settings: Settings, manifest: Manifest,
              generator: ImageGenerator, progress: Progress = print) -> BatchSummary:
    todo = [job for job in jobs if job.action == GENERATE]
    summary = BatchSummary(planned=len(todo), skipped=len(jobs) - len(todo))
    if not todo:
        return summary

    started = time.monotonic()
    run_record: dict[str, Any] = {"started_at": now_iso(), "cardgen_version": __version__,
                                  "planned": len(todo)}
    manifest.data.setdefault("runs", []).append(run_record)

    try:
        progress(f"Loading model {settings.model_id} (this can take a minute) ...")
        try:
            generator.load()
        except GeneratorError as exc:
            summary.aborted = f"{type(exc).__name__}: {exc}"
            summary.fatal = True
            return summary
        run_record["generator"] = generator.describe()
        manifest.save()

        consecutive_failures = 0
        for index, job in enumerate(todo, start=1):
            progress(f"[{index}/{len(todo)}] {job.card.id}: generating "
                     f"{job.request.width}x{job.request.height}, {job.request.steps} steps, seed {job.request.seed}")
            try:
                elapsed = _run_job(job, settings, manifest, generator)
            except KeyboardInterrupt:
                _mark_failed(manifest, job, INTERRUPTED, "interrupted by user")
                summary.interrupted = True
                progress(f"[{index}/{len(todo)}] {job.card.id}: interrupted")
                break
            except GeneratorError as exc:
                _mark_failed(manifest, job, FAILED, str(exc), exc)
                summary.failed.append(job.card.id)
                progress(f"[{index}/{len(todo)}] {job.card.id}: FAILED - {exc}")
                if exc.fatal:
                    summary.aborted = f"{type(exc).__name__}: {exc}"
                    summary.fatal = True
                    break
                consecutive_failures += 1
            except Exception as exc:  # unexpected per-card problem: record it and keep going
                log.debug("card %s failed", job.card.id, exc_info=True)
                _mark_failed(manifest, job, FAILED, f"{type(exc).__name__}: {exc}", exc)
                summary.failed.append(job.card.id)
                progress(f"[{index}/{len(todo)}] {job.card.id}: FAILED - {type(exc).__name__}: {exc}")
                consecutive_failures += 1
            else:
                consecutive_failures = 0
                summary.succeeded.append(job.card.id)
                progress(f"[{index}/{len(todo)}] {job.card.id}: saved {job.output_path} ({elapsed:.0f}s)")
            limit = settings.max_consecutive_failures
            if limit and consecutive_failures >= limit:
                summary.aborted = (f"{consecutive_failures} cards failed in a row; stopping in case the "
                                   "problem is systemic (see max_consecutive_failures)")
                break
    except KeyboardInterrupt:
        summary.interrupted = True
    finally:
        summary.elapsed_s = time.monotonic() - started
        run_record.update(finished_at=now_iso(), succeeded=len(summary.succeeded),
                          failed=len(summary.failed), interrupted=summary.interrupted,
                          aborted=summary.aborted)
        manifest.save()
        generator.close()
    return summary


def _run_job(job: CardJob, settings: Settings, manifest: Manifest, generator: ImageGenerator) -> float:
    card, req = job.card, job.request
    previous = manifest.get(card.id) or {}
    manifest.update(
        card.id,
        name=card.name,
        prompt=card.prompt,
        full_prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        seed=req.seed,
        seed_source=job.seed_source,
        model_id=settings.model_id,
        model_variant=settings.model_variant,
        settings={"width": req.width, "height": req.height, "steps": req.steps,
                  "guidance_scale": req.guidance_scale, "preset": settings.preset,
                  "overrides": dict(card.overrides)},
        fingerprint=job.fingerprint,
        status=RUNNING,
        output_path=str(job.output_path),
        final_path=None,
        started_at=now_iso(),
        completed_at=None,
        error=None,
        attempts=previous.get("attempts", 0) + 1,
    )
    manifest.save()

    started = time.monotonic()
    result = generator.generate(req)
    image = result.image
    if image.size != (req.width, req.height):
        raise GeneratorError(f"model returned {image.size[0]}x{image.size[1]}, "
                             f"expected {req.width}x{req.height}")

    if job.replaces_existing and job.output_path.exists():
        archived = _archive(job.output_path, settings.output_dir / "replaced")
        manifest.update(card.id, replaced_previous=str(archived))
    save_png_atomic(image, job.output_path, {
        "cardgen:id": card.id, "cardgen:prompt": req.prompt,
        "cardgen:negative_prompt": req.negative_prompt, "cardgen:seed": str(req.seed),
        "cardgen:model": settings.model_id, "cardgen:steps": str(req.steps),
        "cardgen:guidance_scale": str(req.guidance_scale),
    })
    check = check_image(job.output_path, (req.width, req.height))
    if not check.ok:
        raise GeneratorError(f"saved image failed validation: {check.reason}")

    final_path, pp_error = None, None
    if settings.postprocess.enabled:
        try:
            final_path = str(postprocess_file(job.output_path, settings))
        except Exception as exc:  # keep the original; report the post-processing problem
            pp_error = f"{type(exc).__name__}: {exc}"
            log.warning("post-processing failed for %s: %s", card.id, pp_error)

    elapsed = time.monotonic() - started
    manifest.update(card.id, status=SUCCESS, completed_at=now_iso(), duration_s=round(elapsed, 1),
                    generation_info=result.info, final_path=final_path, postprocess_error=pp_error)
    manifest.save()
    return elapsed


def _mark_failed(manifest: Manifest, job: CardJob, status: str, message: str,
                 exc: BaseException | None = None) -> None:
    error = {"type": type(exc).__name__ if exc else "KeyboardInterrupt", "message": message}
    if manifest.get(job.card.id) is None:
        manifest.update(job.card.id, name=job.card.name, prompt=job.card.prompt,
                        seed=job.request.seed, output_path=str(job.output_path))
    manifest.update(job.card.id, status=status, completed_at=now_iso(), error=error)
    manifest.save()


def _archive(path: Path, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = folder / f"{path.stem}-{stamp}{path.suffix}"
    shutil.copy2(path, target)
    return target


# --------------------------------------------------------------------------- #
# Post-processing and QC
# --------------------------------------------------------------------------- #


def postprocess_file(source: Path, settings: Settings) -> Path:
    """Resize a generated image to the final card size in ``output/<subdir>/``.
    The original is never modified."""
    pp = settings.postprocess
    target = settings.final_dir / source.name
    with Image.open(source) as img:
        final = fit_image(img, pp.width, pp.height, pp.fit, pp.pad_color)
    save_png_atomic(final, target)
    return target


def qc_rows(manifest: Manifest, settings: Settings) -> list[dict[str, Any]]:
    rows = []
    for card_id, entry in sorted(manifest.cards.items()):
        if entry.get("status") != SUCCESS:
            continue
        path = Path(entry["output_path"])
        check = check_image(path)
        row: dict[str, Any] = {"id": card_id, "path": str(path), "valid": check.ok}
        if check.ok:
            with Image.open(path) as img:
                w, h = img.size
                row.update(width=w, height=h, aspect_2_3=abs(w / h - 2 / 3) <= 0.01)
                row.update(safe_zone_report(img, settings.safe_top, settings.safe_bottom))
            row["review"] = row["top_ratio"] > 1.0 or row["bottom_ratio"] > 1.0 or not row["aspect_2_3"]
        else:
            row.update(review=True, problem=check.reason)
        rows.append(row)
    return rows
