"""voxcpm-mcp: MCP server exposing the VoxCPM TTS pipeline.

Wraps the OpenBMB/VoxCPM Python API (loaded lazily on first use) and
exposes a small set of FastMCP tools. Heavy ML dependencies (torch,
transformers, ...) are imported on demand so that ``voxcpm-mcp --help``
does not require the model to be downloaded.

The server holds a single VoxCPM model instance in memory — lazy-loaded
on the first call to ``voxcpm_synthesize`` / ``voxcpm_describe_voice`` /
``voxcpm_clone_voice``. Use ``voxcpm_load_model`` / ``voxcpm_unload_model``
to control the lifecycle explicitly. The model lives at HF Hub id
``openbmb/VoxCPM2`` by default (override with the ``VOXCPM_MODEL_ID`` env
var or the ``model_id`` argument).

Output files are written to a cache directory (default
``C:\\Data\\Hermes\\audio_cache\\voxcpm\\`` — overridable with
``VOXCPM_OUTPUT_DIR``) and returned by absolute path. Files are kept across
sessions unless explicitly cleared; remove them with ``voxcpm_clear_cache``.

NOTE: keep ``from __future__ import annotations`` OUT of this file —
FastMCP's tool decorator inspects parameter annotations via
``inspect.signature`` and crashes on string forward references with
``TypeError: issubclass() arg 1 must be a class``. Use bare ``X = None``
(not ``Optional[X]``) for the same reason.
"""

import asyncio
import contextlib
import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from voxcpm_mcp.__about__ import __version__


# --- Configuration ---------------------------------------------------------

DEFAULT_MODEL_ID = "openbmb/VoxCPM2"
DEFAULT_DENOISER_ID = "iic/speech_zipenhancer_ans_multiloss_16k_base"

DEFAULT_OUTPUT_DIR = Path(r"C:\Data\Hermes\audio_cache\voxcpm")
SAMPLE_RATE_INFO_DEFAULT = 48000  # VoxCPM2 outputs 48 kHz


def _model_id() -> str:
    return os.environ.get("VOXCPM_MODEL_ID", DEFAULT_MODEL_ID)


def _denoiser_id() -> str | None:
    val = os.environ.get("VOXCPM_DENOISER_ID")
    if val and val.strip().lower() in ("none", "false", "0", "off"):
        return None
    return val or DEFAULT_DENOISER_ID


def _device() -> str | None:
    val = os.environ.get("VOXCPM_DEVICE")
    if val and val.strip().lower() in ("auto", "none"):
        return None
    return val or None


def _optimize() -> bool:
    val = os.environ.get("VOXCPM_OPTIMIZE", "1").strip().lower()
    return val not in ("0", "false", "off", "no")


def _default_voice_description() -> str | None:
    val = os.environ.get("VOXCPM_DEFAULT_VOICE_DESCRIPTION")
    if val and val.strip().lower() in ("none", "false", "0", "off", ""):
        return None
    return val or None


def _output_dir() -> Path:
    raw = os.environ.get("VOXCPM_OUTPUT_DIR")
    return Path(raw) if raw else DEFAULT_OUTPUT_DIR


def _cache_max_bytes() -> int | None:
    """Optional cap on cache dir size in bytes (0 = uncapped)."""
    val = os.environ.get("VOXCPM_CACHE_MAX_BYTES")
    if not val:
        return None
    try:
        n = int(val)
        return n if n > 0 else None
    except ValueError:
        return None


def _max_len() -> int:
    """Hard cap on generation length, default 4096 (matches VoxCPM default)."""
    val = os.environ.get("VOXCPM_MAX_LEN")
    if not val:
        return 4096
    try:
        return max(64, int(val))
    except ValueError:
        return 4096


# --- Model lifecycle -------------------------------------------------------

_MODEL_LOCK = threading.Lock()
_MODEL: Any = None  # VoxCPMModel | None
_MODEL_LOADED_ID: str | None = None
_MODEL_LOADED_AT: float | None = None
_MODEL_DEVICE: str | None = None


def _is_loaded() -> bool:
    return _MODEL is not None


def _info_dict() -> dict[str, Any]:
    sample_rate = None
    arch = None
    if _MODEL is not None:
        try:
            sample_rate = _MODEL.tts_model.sample_rate
        except Exception:
            sample_rate = None
        arch = type(_MODEL.tts_model).__name__
    return {
        "loaded": _MODEL is not None,
        "model_id": _MODEL_LOADED_ID,
        "loaded_at": _MODEL_LOADED_AT,
        "device": _MODEL_DEVICE,
        "architecture": arch,
        "sample_rate": sample_rate,
        "output_dir": str(_output_dir()),
        "denoiser_id": _denoiser_id(),
        "optimize": _optimize(),
    }


def _ensure_loaded(
    model_id: str | None = None,
    force_reload: bool = False,
    load_denoiser: bool | None = None,
) -> Any:
    """Load the VoxCPM model on first call; reuse the cached instance otherwise.

    Thread-safe via ``_MODEL_LOCK``. Heavy imports happen here so that tools
    that don't need synthesis (e.g. ``voxcpm_health``) stay cheap.
    """
    global _MODEL, _MODEL_LOADED_ID, _MODEL_LOADED_AT, _MODEL_DEVICE

    target_id = model_id or _model_id()
    if load_denoiser is None:
        load_denoiser = _denoiser_id() is not None

    with _MODEL_LOCK:
        if _MODEL is not None and not force_reload and _MODEL_LOADED_ID == target_id:
            return _MODEL

        # Lazy import — torch is heavy and we don't want it at server start.
        from voxcpm import VoxCPM  # type: ignore

        started = time.time()
        model = VoxCPM.from_pretrained(
            hf_model_id=target_id,
            load_denoiser=load_denoiser,
            zipenhancer_model_id=_denoiser_id() if load_denoiser else None,
            optimize=_optimize(),
            device=_device(),
            cache_dir=os.environ.get("VOXCPM_CACHE_DIR") or None,
            local_files_only=os.environ.get("VOXCPM_LOCAL_FILES_ONLY", "").lower()
            in ("1", "true", "yes"),
        )

        if _MODEL is not None:
            # Releasing the previous model explicitly is the polite thing to do,
            # but VoxCPM doesn't expose __del__ cleanup, so just drop the ref.
            del _MODEL

        _MODEL = model
        _MODEL_LOADED_ID = target_id
        _MODEL_LOADED_AT = time.time()
        try:
            _MODEL_DEVICE = str(model.tts_model.device) if hasattr(model.tts_model, "device") else None
        except Exception:
            _MODEL_DEVICE = None

        elapsed = _MODEL_LOADED_AT - started
        print(
            f"[voxcpm-mcp] loaded {target_id} in {elapsed:.1f}s "
            f"(denoiser={load_denoiser}, device={_MODEL_DEVICE})",
            flush=True,
        )
        return _MODEL


def _unload() -> None:
    global _MODEL, _MODEL_LOADED_ID, _MODEL_LOADED_AT, _MODEL_DEVICE
    with _MODEL_LOCK:
        _MODEL = None
        _MODEL_LOADED_ID = None
        _MODEL_LOADED_AT = None
        _MODEL_DEVICE = None


# --- Helpers ---------------------------------------------------------------

def _ensure_output_dir() -> Path:
    out = _output_dir()
    out.mkdir(parents=True, exist_ok=True)
    return out


def _trim_cache() -> None:
    """Best-effort cleanup: drop oldest files when the cache exceeds the cap."""
    cap = _cache_max_bytes()
    if not cap:
        return
    out = _ensure_output_dir()
    files = [p for p in out.glob("*.wav") if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    if total <= cap:
        return
    files.sort(key=lambda p: p.stat().st_mtime)  # oldest first
    for p in files:
        if total <= cap:
            break
        try:
            size = p.stat().st_size
            p.unlink()
            total -= size
        except OSError:
            pass


def _resolve_output_path(output_path: str | None, *, prefix: str) -> Path:
    out_dir = _ensure_output_dir()
    if output_path:
        p = Path(output_path)
        if not p.is_absolute():
            p = out_dir / p
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    name = f"{prefix}-{uuid.uuid4().hex[:10]}.wav"
    return out_dir / name


def _to_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)


def _coerce_audio_file(uri: str | None) -> str | None:
    """Accept a file path, ``file:`` URI, or ``http(s):`` URL and return a
    local filesystem path. http(s) URLs are downloaded to the cache dir."""
    if not uri:
        return None
    s = uri.strip()
    if s.startswith("file:"):
        # file:///C:/path → C:/path on Windows; file:///tmp/foo → /tmp/foo on POSIX
        from urllib.parse import unquote, urlparse

        parsed = urlparse(s)
        local = unquote(parsed.path)
        if os.name == "nt" and local.startswith("/") and len(local) > 2 and local[2] == ":":
            local = local[1:]
        return local
    if s.startswith(("http://", "https://")):
        import urllib.request

        out_dir = _ensure_output_dir()
        name = f"ref-{uuid.uuid4().hex[:10]}.wav"
        dest = out_dir / name
        with urllib.request.urlopen(s) as resp, open(dest, "wb") as f:
            f.write(resp.read())
        return str(dest)
    return s  # treat as plain path


def _strip_control(text: str, voice_description: str | None) -> str:
    """VoxCPM2 voice design convention: describe in parens at the start.

    Examples:
        "(warm female voice) Hello!"  — explicit control token
        text="Hello!", description="warm female voice" → "(warm female voice) Hello!"
    """
    if not text:
        raise ValueError("text must be a non-empty string")
    text = text.replace("\n", " ").strip()
    if voice_description:
        voice_description = voice_description.strip()
        if voice_description and not text.startswith("("):
            return f"({voice_description}) {text}"
    return text


# --- MCP server ------------------------------------------------------------

mcp = FastMCP("voxcpm")


# ---- Health / introspection ------------------------------------------------


@mcp.tool()
async def voxcpm_health() -> str:
    """Check whether VoxCPM is reachable and ready.

    Returns server version, model load state, default model id, output
    directory, and the underlying PyTorch version. Call this first to
    confirm the server is up. The model is NOT loaded by this call — that
    happens on the first synthesize call (or via ``voxcpm_load_model``).
    """
    return await asyncio.to_thread(_health_sync)


def _health_sync() -> str:
    info: dict[str, Any] = {
        "server_version": __version__,
        "default_model_id": _model_id(),
        "denoiser_id": _denoiser_id(),
        "optimize": _optimize(),
        "output_dir": str(_output_dir()),
        "cache_max_bytes": _cache_max_bytes(),
        "default_voice_description": _default_voice_description(),
    }
    info["model"] = _info_dict()

    try:
        import torch  # type: ignore

        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
    except Exception as exc:
        info["torch_error"] = f"{type(exc).__name__}: {exc}"

    return _to_json(info)


@mcp.tool()
async def voxcpm_model_info() -> str:
    """Return details about the currently loaded model (or null if not loaded).

    Includes sample rate, architecture, device, and a timestamp of when the
    model was loaded.
    """
    return _to_json(_info_dict())


# ---- Lifecycle -------------------------------------------------------------


@mcp.tool()
async def voxcpm_load_model(
    model_id: str = None,
    load_denoiser: bool = True,
    force_reload: bool = False,
) -> str:
    """Preload a VoxCPM model into memory.

    Use this to warm the model before a long batch, or to switch models
    (e.g. swap to ``openbmb/VoxCPM-0.5B`` for speed). The default model id
    comes from ``VOXCPM_MODEL_ID`` (or ``openbmb/VoxCPM2``).

    Args:
        model_id: HuggingFace repo id or local path. None = use default.
        load_denoiser: Whether to also load the speech-enhancement model
            (ZipEnhancer). Set False on devices with limited memory.
        force_reload: Re-download / re-initialise even if the same model id
            is already loaded.
    """
    return await asyncio.to_thread(_load_model_sync, model_id, load_denoiser, force_reload)


def _load_model_sync(model_id: str | None, load_denoiser: bool, force_reload: bool) -> str:
    target = model_id or _model_id()
    started = time.time()
    _ensure_loaded(model_id=target, force_reload=force_reload, load_denoiser=load_denoiser)
    elapsed = time.time() - started
    info = _info_dict()
    info["load_seconds"] = round(elapsed, 2)
    return _to_json(info)


@mcp.tool()
async def voxcpm_unload_model() -> str:
    """Drop the loaded model and free its memory.

    Safe to call even if no model is loaded — returns a no-op confirmation.
    """
    return await asyncio.to_thread(_unload_sync)


def _unload_sync() -> str:
    global _MODEL, _MODEL_LOADED_ID, _MODEL_LOADED_AT, _MODEL_DEVICE
    with _MODEL_LOCK:
        was_loaded = _MODEL is not None
        _unload()
    return _to_json({"unloaded": was_loaded})


# ---- Synthesis -------------------------------------------------------------


@mcp.tool()
async def voxcpm_synthesize(
    text: str,
    voice_description: str = None,
    reference_audio: str = None,
    prompt_text: str = None,
    output_path: str = None,
    cfg_value: float = 2.0,
    inference_timesteps: int = 10,
    normalize: bool = False,
    denoise: bool = False,
) -> str:
    """Synthesize speech from text. Returns the WAV file path + metadata.

    Three modes, in priority order:
      1. **Voice clone** — pass ``reference_audio`` (and optionally
         ``prompt_text``). The model clones the speaker's timbre from the
         reference clip. ``voice_description`` is appended as a control
         instruction for style (rate/emotion/etc.).
      2. **Voice design** — pass ``voice_description`` (e.g. ``"warm
         female voice"``). The model creates a new voice from the natural
         language description. No reference audio needed.
      3. **Plain TTS** — just ``text``. VoxCPM2 infers prosody from context.

    The first call lazily loads the model (a few GB download on first run,
    then cached). Subsequent calls are fast.

    Args:
        text: Text to speak. Non-empty.
        voice_description: Natural-language voice description (e.g.
            ``"young British male, deep and measured"``). For VoxCPM2.
        reference_audio: Path / ``file:`` URI / ``http(s)`` URL to a 16 kHz
            reference clip for voice cloning.
        prompt_text: Transcript of the reference clip — use together with
            ``reference_audio`` for ultimate-clone mode.
        output_path: Where to save the WAV. None = auto-named file in the
            cache directory.
        cfg_value: Guidance scale (1.0–3.0 recommended; default 2.0).
        inference_timesteps: Diffusion steps (4–30 recommended; default 10).
        normalize: Run text normalization (numbers, dates, etc.) first.
        denoise: Denoise the reference audio (requires denoiser; ignored
            when no reference audio is provided).

    Returns:
        JSON string with:
          - ``output_path``: absolute path of the WAV file.
          - ``duration_s``: audio length in seconds.
          - ``sample_rate``: Hz (48000 for VoxCPM2).
          - ``samples``: total sample count.
          - ``model_id``: which model produced this.
          - ``mode``: ``"clone"`` / ``"design"`` / ``"plain"``.
    """
    return await asyncio.to_thread(
        _synthesize_sync,
        text,
        voice_description,
        reference_audio,
        prompt_text,
        output_path,
        cfg_value,
        inference_timesteps,
        normalize,
        denoise,
    )


def _synthesize_sync(
    text: str,
    voice_description: str | None,
    reference_audio: str | None,
    prompt_text: str | None,
    output_path: str | None,
    cfg_value: float,
    inference_timesteps: int,
    normalize: bool,
    denoise: bool,
) -> str:
    if not isinstance(text, str) or not text.strip():
        return _to_json({"error": "text must be a non-empty string"})

    # Apply env-var default voice description when caller didn't specify one
    # and no reference audio is given (voice-design / plain mode).
    if not voice_description and not reference_audio:
        voice_description = _default_voice_description()

    # Mode detection
    ref = _coerce_audio_file(reference_audio) if reference_audio else None
    if ref and not Path(ref).exists():
        return _to_json({"error": f"reference_audio not found: {ref}"})

    if ref and prompt_text:
        mode = "ultimate_clone"
        # VoxCPM convention: in ultimate-clone mode, the same reference
        # clip doubles as both the prompt (continuation) and the voice
        # timbre source for max fidelity. See README.
        final_text = _strip_control(text, voice_description)
        prompt_wav = ref
        prompt_text_arg = prompt_text
        reference_wav = ref
    elif ref:
        mode = "clone"
        final_text = _strip_control(text, voice_description)
        prompt_wav = None
        prompt_text_arg = None
        reference_wav = ref
    elif voice_description:
        mode = "design"
        final_text = _strip_control(text, voice_description)
        prompt_wav = None
        prompt_text_arg = None
        reference_wav = None
    else:
        mode = "plain"
        final_text = _strip_control(text, None)
        prompt_wav = None
        prompt_text_arg = None
        reference_wav = None

    try:
        model = _ensure_loaded()
    except Exception as exc:
        return _to_json({"error": f"failed to load model: {type(exc).__name__}: {exc}"})

    out_path = _resolve_output_path(output_path, prefix=f"{mode}-{int(time.time())}")

    started = time.time()
    try:
        wav = model.generate(
            text=final_text,
            prompt_wav_path=prompt_wav,
            prompt_text=prompt_text_arg,
            reference_wav_path=reference_wav,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            normalize=normalize,
            denoise=denoise and (ref is not None),
            max_len=_max_len(),
        )
    except Exception as exc:
        return _to_json({
            "error": f"{type(exc).__name__}: {exc}",
            "mode": mode,
            "final_text": final_text,
        })

    # wav is a numpy.ndarray (1D float32 on CPU). Write it out.
    import numpy as np
    import soundfile as sf

    if isinstance(wav, np.ndarray):
        wav_np = wav
    else:
        # Some code paths return a generator or list of chunks; concatenate.
        try:
            wav_np = np.concatenate(list(wav))
        except Exception:
            wav_np = np.asarray(wav)

    wav_np = np.asarray(wav_np, dtype=np.float32).squeeze()
    sample_rate = int(getattr(model.tts_model, "sample_rate", SAMPLE_RATE_INFO_DEFAULT))

    try:
        sf.write(str(out_path), wav_np, sample_rate, subtype="PCM_16")
    except Exception as exc:
        return _to_json({"error": f"failed to write WAV: {type(exc).__name__}: {exc}"})

    duration_s = round(float(len(wav_np)) / float(sample_rate), 3)
    elapsed = round(time.time() - started, 2)
    file_size = out_path.stat().st_size

    _trim_cache()

    return _to_json({
        "output_path": str(out_path),
        "duration_s": duration_s,
        "sample_rate": sample_rate,
        "samples": int(wav_np.shape[0]),
        "file_bytes": file_size,
        "model_id": _MODEL_LOADED_ID,
        "mode": mode,
        "cfg_value": cfg_value,
        "inference_timesteps": inference_timesteps,
        "elapsed_s": elapsed,
        "rtf": round(elapsed / duration_s, 3) if duration_s > 0 else None,
    })


@mcp.tool()
async def voxcpm_describe_voice(
    text: str,
    voice_description: str,
    output_path: str = None,
    cfg_value: float = 2.0,
    inference_timesteps: int = 10,
    normalize: bool = False,
) -> str:
    """Synthesize speech with a brand-new voice designed from a description.

    Thin alias around ``voxcpm_synthesize`` for clarity — pass a
    natural-language voice description (e.g. ``"deep male, slow pacing,
    theatrical tone"``) and the model creates that voice from scratch
    (no reference audio needed).

    Args:
        text: Text to speak. Non-empty.
        voice_description: Natural-language voice description. VoxCPM2-only.
        output_path: Where to save the WAV. None = auto-named.
        cfg_value: Guidance scale (1.0–3.0; default 2.0).
        inference_timesteps: Diffusion steps (4–30; default 10).
        normalize: Run text normalization first.
    """
    return await voxcpm_synthesize(
        text=text,
        voice_description=voice_description,
        output_path=output_path,
        cfg_value=cfg_value,
        inference_timesteps=inference_timesteps,
        normalize=normalize,
    )


@mcp.tool()
async def voxcpm_clone_voice(
    text: str,
    reference_audio: str,
    prompt_text: str = None,
    voice_description: str = None,
    output_path: str = None,
    cfg_value: float = 2.0,
    inference_timesteps: int = 10,
    normalize: bool = False,
    denoise: bool = False,
) -> str:
    """Clone a voice from a reference audio clip.

    Args:
        text: Text to speak in the cloned voice.
        reference_audio: Path / ``file:`` URI / ``http(s)`` URL to the
            reference clip (16 kHz WAV recommended).
        prompt_text: Transcript of the reference clip. When provided, the
            model continues seamlessly from the reference for ultimate-fidelity
            cloning (VoxCPM1.5+ behavior).
        voice_description: Optional style guidance (e.g. ``"faster,
            cheerful"``) layered on top of the cloned timbre.
        output_path: Where to save the WAV. None = auto-named.
        cfg_value: Guidance scale (1.0–3.0; default 2.0).
        inference_timesteps: Diffusion steps (4–30; default 10).
        normalize: Run text normalization first.
        denoise: Denoise the reference audio (requires denoiser).
    """
    return await voxcpm_synthesize(
        text=text,
        voice_description=voice_description,
        reference_audio=reference_audio,
        prompt_text=prompt_text,
        output_path=output_path,
        cfg_value=cfg_value,
        inference_timesteps=inference_timesteps,
        normalize=normalize,
        denoise=denoise,
    )


# ---- Cache management ------------------------------------------------------


@mcp.tool()
async def voxcpm_list_outputs() -> str:
    """List WAV files in the VoxCPM output cache.

    Useful for finding previously-generated audio without re-synthesizing.
    """
    return await asyncio.to_thread(_list_outputs_sync)


def _list_outputs_sync() -> str:
    out = _ensure_output_dir()
    files = sorted(out.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    items = []
    for p in files:
        st = p.stat()
        items.append({
            "path": str(p),
            "name": p.name,
            "size_bytes": st.st_size,
            "modified": st.st_mtime,
        })
    return _to_json({"count": len(items), "dir": str(out), "items": items})


@mcp.tool()
async def voxcpm_clear_cache(prefix: str = None) -> str:
    """Delete cached output WAVs.

    Args:
        prefix: Only delete files starting with this string (e.g.
            ``"clone-"``). None = delete all ``*.wav`` in the cache dir.
    """
    return await asyncio.to_thread(_clear_cache_sync, prefix)


def _clear_cache_sync(prefix: str | None) -> str:
    out = _ensure_output_dir()
    deleted = 0
    bytes_freed = 0
    for p in out.glob("*.wav"):
        if prefix and not p.name.startswith(prefix):
            continue
        try:
            sz = p.stat().st_size
            p.unlink()
            deleted += 1
            bytes_freed += sz
        except OSError:
            pass
    return _to_json({
        "deleted": deleted,
        "bytes_freed": bytes_freed,
        "dir": str(out),
        "prefix": prefix,
    })


# --- Entry point -----------------------------------------------------------


def main() -> None:
    """Run the MCP server over stdio (default transport for Hermes)."""
    mcp.run()


if __name__ == "__main__":
    main()