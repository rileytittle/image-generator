"""Image-generation backends.

``ImageGenerator`` is the small interface the batch runner depends on. The only
real implementation is ``DiffusersGenerator``, which loads any Hugging Face
Diffusers text-to-image pipeline via ``AutoPipelineForText2Image`` (SDXL by
default). Heavy imports (torch, diffusers) happen lazily inside ``load()`` so
validation, dry runs and tests never need them.
"""

from __future__ import annotations

import gc
import importlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from .config import Settings
from .image_utils import is_blank
from .prompts import CLIP_TOKEN_LIMIT

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class GeneratorError(RuntimeError):
    """Base class. ``fatal`` errors stop the batch; others fail only the current card."""

    fatal = False


class DependencyError(GeneratorError):
    fatal = True


class DeviceError(GeneratorError):
    fatal = True


class ModelLoadError(GeneratorError):
    fatal = True


class OutOfMemoryError(GeneratorError):
    fatal = True


class BlankImageError(GeneratorError):
    """The pipeline returned a single-colour image (usually NaNs from half precision)."""


# --------------------------------------------------------------------------- #
# Interface
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    negative_prompt: str
    width: int
    height: int
    steps: int
    guidance_scale: float
    seed: int
    reference_images: tuple[Path, ...] = field(default_factory=tuple)


@dataclass
class GenerationResult:
    image: Image.Image
    info: dict[str, Any] = field(default_factory=dict)


class ImageGenerator(Protocol):
    def load(self) -> None: ...

    def generate(self, request: GenerationRequest) -> GenerationResult: ...

    def describe(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


# --------------------------------------------------------------------------- #
# Device and precision selection (pure functions; take the torch module so they
# can be tested with a fake)
# --------------------------------------------------------------------------- #


def mps_status(torch: Any) -> tuple[bool, str]:
    backend = getattr(getattr(torch, "backends", None), "mps", None)
    if backend is None:
        return False, "this PyTorch build has no MPS backend"
    if not backend.is_built():
        return False, "PyTorch was not built with MPS support"
    if not backend.is_available():
        return False, "MPS is built but unavailable (needs Apple Silicon and macOS 12.3+)"
    return True, "MPS available"


def resolve_device(requested: str, allow_cpu_fallback: bool, torch: Any) -> tuple[str, str]:
    """Return (device, explanation). Raises DeviceError instead of silently using the CPU."""
    if requested == "cpu":
        return "cpu", "CPU explicitly requested (expect several minutes per image)"
    ok, why = mps_status(torch)
    if ok:
        return "mps", "Apple Silicon GPU via MPS"
    if requested == "mps":
        raise DeviceError(f"device 'mps' requested but {why}")
    if allow_cpu_fallback:
        return "cpu", f"falling back to CPU because {why} (allow_cpu_fallback is on)"
    raise DeviceError(
        f"no GPU device available: {why}. Set \"device\": \"cpu\" or "
        "\"allow_cpu_fallback\": true in config/settings.json (or pass --device cpu) "
        "to run on the CPU, which is very slow for SDXL.")


def resolve_dtype(requested: str, device: str, torch: Any) -> Any:
    if requested == "auto":
        return torch.float16 if device == "mps" else torch.float32
    if device == "cpu" and requested != "float32":
        raise DeviceError(f"dtype '{requested}' is not reliable on CPU; use float32 or auto")
    return getattr(torch, requested)


def is_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return type(exc).__name__ == "OutOfMemoryError" or "out of memory" in text


# --------------------------------------------------------------------------- #
# Diffusers implementation
# --------------------------------------------------------------------------- #


def _import(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise DependencyError(
            f"'{name}' is not installed. Install the generation extras: "
            "pip install -e '.[gen]'") from exc


class DiffusersGenerator:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.pipe: Any = None
        self.torch: Any = None
        self.device: str | None = None
        self.dtype: Any = None
        self.device_reason = ""
        self.versions: dict[str, str] = {}

    # ---- lifecycle ----

    def load(self) -> None:
        torch = self.torch = _import("torch")
        diffusers = _import("diffusers")
        _import("transformers")
        self.versions = {"torch": torch.__version__, "diffusers": diffusers.__version__,
                         "transformers": importlib.import_module("transformers").__version__}

        self.device, self.device_reason = resolve_device(
            self.settings.device, self.settings.allow_cpu_fallback, torch)
        self.dtype = resolve_dtype(self.settings.dtype, self.device, torch)
        log.info("Device: %s (%s), dtype: %s", self.device, self.device_reason,
                 str(self.dtype).replace("torch.", ""))

        # diffusers renamed torch_dtype -> dtype (0.41 deprecates the old name).
        dtype_key = "dtype" if _version_tuple(diffusers.__version__) >= (0, 41) else "torch_dtype"
        kwargs: dict[str, Any] = {
            dtype_key: self.dtype,
            "use_safetensors": True,
            "local_files_only": not self.settings.allow_download,
        }
        if self.settings.model_variant:
            kwargs["variant"] = self.settings.model_variant
        if self.settings.cache_dir:
            kwargs["cache_dir"] = str(self.settings.cache_dir)

        log.debug("Loading model %s%s ...", self.settings.model_id,
                 "" if self.settings.allow_download else " from the local cache")
        started = time.monotonic()
        try:
            pipe = diffusers.AutoPipelineForText2Image.from_pretrained(self.settings.model_id, **kwargs)
        except Exception as exc:  # diffusers raises OSError/ValueError/EnvironmentError variants
            hint = ""
            if not self.settings.allow_download:
                hint = (" If the model has not been downloaded yet, run once with --allow-download "
                        "(SDXL base fp16 is about 7 GB).")
            if self.settings.model_variant:
                hint += (f" If this model has no '{self.settings.model_variant}' weight variant, "
                         "set \"model_variant\": null.")
            reason = str(exc).strip().rstrip(".")
            raise ModelLoadError(f"could not load '{self.settings.model_id}': {reason}.{hint}") from exc

        try:
            pipe = pipe.to(self.device)
        except Exception as exc:
            if is_oom(exc):
                raise OutOfMemoryError(f"not enough memory to move the model to {self.device}: {exc}") from exc
            raise ModelLoadError(f"could not move the model to {self.device}: {exc}") from exc

        slicing = self.settings.attention_slicing
        if slicing is True or (slicing == "auto" and self.device == "mps"):
            pipe.enable_attention_slicing()
            log.info("Attention slicing enabled (lower peak memory, slightly slower)")
        pipe.set_progress_bar_config(leave=False)
        self.pipe = pipe
        log.info("Model ready in %.0fs (pipeline: %s)", time.monotonic() - started, type(pipe).__name__)

    def close(self) -> None:
        self.pipe = None
        gc.collect()
        self._empty_cache()

    def describe(self) -> dict[str, Any]:
        return {
            "model_id": self.settings.model_id,
            "model_variant": self.settings.model_variant,
            "pipeline": type(self.pipe).__name__ if self.pipe is not None else None,
            "device": self.device,
            "dtype": str(self.dtype).replace("torch.", "") if self.dtype is not None else None,
            "versions": self.versions,
        }

    # ---- generation ----

    def generate(self, request: GenerationRequest) -> GenerationResult:
        if self.pipe is None:
            raise GeneratorError("generator used before load()")
        if request.reference_images:
            raise GeneratorError("reference-image conditioning is not implemented in this version")
        torch = self.torch
        info: dict[str, Any] = {}
        # A CPU generator gives the most repeatable results across devices (recommended for MPS).
        rng = torch.Generator(device="cpu").manual_seed(request.seed)
        kwargs: dict[str, Any] = {
            "width": request.width,
            "height": request.height,
            "num_inference_steps": request.steps,
            "guidance_scale": request.guidance_scale,
            "generator": rng,
            "output_type": "pil",
        }
        started = time.monotonic()
        try:
            with torch.inference_mode():
                kwargs.update(self._prompt_kwargs(request, info))
                result = self.pipe(**kwargs)
        except Exception as exc:
            if is_oom(exc):
                self._empty_cache()
                raise OutOfMemoryError(
                    f"ran out of memory at {request.width}x{request.height}: {exc}") from exc
            raise
        finally:
            self._empty_cache()
        image = result.images[0]
        info["elapsed_s"] = round(time.monotonic() - started, 1)
        if is_blank(image):
            raise BlankImageError(
                "the model returned a blank single-colour image (often NaNs in float16); "
                "try a different seed, or set \"dtype\": \"float32\"")
        return GenerationResult(image=image, info=info)

    def _empty_cache(self) -> None:
        torch = self.torch
        if torch is not None and self.device == "mps" and hasattr(torch, "mps"):
            try:
                torch.mps.empty_cache()
            except Exception:  # pragma: no cover - best effort
                pass

    # ---- prompt encoding ----

    def _encoders(self) -> list[tuple[Any, Any]]:
        pairs = []
        for tok_name, enc_name in (("tokenizer", "text_encoder"), ("tokenizer_2", "text_encoder_2")):
            tok, enc = getattr(self.pipe, tok_name, None), getattr(self.pipe, enc_name, None)
            if tok is not None and enc is not None:
                pairs.append((tok, enc))
        return pairs

    def _prompt_kwargs(self, request: GenerationRequest, info: dict[str, Any]) -> dict[str, Any]:
        encoders = self._encoders()
        if encoders:
            tok, enc = encoders[0]
            count = len(tok(request.prompt, add_special_tokens=True).input_ids)
            info["prompt_tokens"] = count
            info["token_limit"] = token_window(tok, enc)
        over_limit = info.get("prompt_tokens", 0) > info.get("token_limit", CLIP_TOKEN_LIMIT)
        is_sdxl = len(encoders) == 2 and hasattr(self.pipe, "text_encoder_2")
        is_sd = len(encoders) == 1 and not hasattr(self.pipe, "text_encoder_2")

        if over_limit and self.settings.long_prompts and (is_sdxl or is_sd):
            info["prompt_mode"] = "long (chunked embeddings)"
            return encode_long_prompt(self.torch, self.pipe, encoders, request.prompt,
                                      request.negative_prompt, sdxl=is_sdxl)
        if over_limit:
            info["prompt_mode"] = "truncated"
            log.warning("Prompt is %d tokens; the text encoder only reads %d, the rest is ignored. "
                        "Shorten the prompt/style or enable long_prompts.",
                        info["prompt_tokens"], info["token_limit"])
        else:
            info.setdefault("prompt_mode", "standard")
        return {"prompt": request.prompt, "negative_prompt": request.negative_prompt or None}


def token_window(tokenizer: Any, encoder: Any) -> int:
    """Tokens one encoder pass can read (77 for CLIP). Tokenizers without a configured
    limit report a huge sentinel, so also respect the encoder's position embeddings."""
    limits = [getattr(tokenizer, "model_max_length", None),
              getattr(getattr(encoder, "config", None), "max_position_embeddings", None)]
    valid = [n for n in limits if isinstance(n, int) and 2 < n < 100_000]
    return min(valid) if valid else CLIP_TOKEN_LIMIT


def _version_tuple(version: str) -> tuple[int, ...]:
    parts = []
    for piece in version.split(".")[:3]:
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def encode_long_prompt(torch: Any, pipe: Any, encoders: list[tuple[Any, Any]], prompt: str,
                       negative: str, *, sdxl: bool) -> dict[str, Any]:
    """Encode prompts longer than CLIP's 77-token window by splitting them into
    75-token chunks, encoding each chunk separately and concatenating the results
    (the approach used by popular SD front ends). Works for SD 1.x/2.x (one CLIP
    encoder) and SDXL (two encoders + pooled embedding)."""
    device = pipe.device

    windows = {id(tok): token_window(tok, enc) for tok, enc in encoders}

    def token_chunks(tok: Any, text: str) -> list[list[int]]:
        ids = tok(text, add_special_tokens=False, truncation=False).input_ids
        size = windows[id(tok)] - 2
        return [ids[i:i + size] for i in range(0, len(ids), size)] or [[]]

    def pad_chunk(tok: Any, chunk: list[int]) -> list[int]:
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        ids = [tok.bos_token_id, *chunk, tok.eos_token_id]
        return ids + [pad_id] * (windows[id(tok)] - len(ids))

    def encode(text: str, n_chunks: int | None) -> tuple[Any, Any, int]:
        per_encoder, pooled, chunk_count = [], None, 0
        for index, (tok, enc) in enumerate(encoders):
            chunks = token_chunks(tok, text)
            if n_chunks is not None:
                chunks = chunks[:n_chunks] + [[]] * (n_chunks - len(chunks))
            chunk_count = len(chunks)
            hidden = []
            for c_index, chunk in enumerate(chunks):
                ids = torch.tensor([pad_chunk(tok, chunk)], device=device)
                if sdxl:
                    out = enc(ids, output_hidden_states=True)
                    hidden.append(out.hidden_states[-2])
                    if index == len(encoders) - 1 and c_index == 0:
                        pooled = out[0]
                else:
                    hidden.append(enc(ids)[0])
            per_encoder.append(torch.cat(hidden, dim=1))
        return torch.cat(per_encoder, dim=-1), pooled, chunk_count

    def chunks_needed(text: str) -> int:
        return max(len(token_chunks(tok, text)) for tok, _ in encoders)

    n = max(chunks_needed(prompt), chunks_needed(negative) if negative else 1)
    prompt_embeds, pooled, _ = encode(prompt, n)
    zero_negative = not negative and getattr(pipe.config, "force_zeros_for_empty_prompt", False)
    if zero_negative:
        negative_embeds = torch.zeros_like(prompt_embeds)
        negative_pooled = torch.zeros_like(pooled) if pooled is not None else None
    else:
        negative_embeds, negative_pooled, _ = encode(negative, n)

    dtype = next(iter(encoders[-1][1].parameters())).dtype
    result = {
        "prompt_embeds": prompt_embeds.to(dtype),
        "negative_prompt_embeds": negative_embeds.to(dtype),
    }
    if sdxl:
        result["pooled_prompt_embeds"] = pooled.to(dtype)
        result["negative_pooled_prompt_embeds"] = negative_pooled.to(dtype)
    return result


def create_generator(settings: Settings) -> ImageGenerator:
    """Factory used by the batch runner (and patched in tests)."""
    return DiffusersGenerator(settings)
