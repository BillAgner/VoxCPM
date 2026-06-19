---
name: voxcpm
description: "Synthesize speech locally using VoxCPM — a 2B-parameter tokenizer-free TTS model supporting 30 languages, voice design from a natural-language description, and voice cloning from a reference audio clip. Backed by the voxcpm MCP server with 9 tools (voxcpm_*). Use when the user wants to generate speech, read something aloud, create a voice, clone a voice, or produce an audio file."
version: 0.1.0
author: Bill Agner
license: MIT
platforms: [windows]
metadata:
  hermes:
    tags: [tts, text-to-speech, voice, audio, voice-cloning, voice-design, local-ai, openbmb]
    related_skills: [mcp-server-setup]
---

# VoxCPM

Local, high-quality text-to-speech via [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM).  
**VoxCPM2** — 2B parameters, 30 languages, 48 kHz output, voice design + voice cloning.

The MCP server (`voxcpm`) wraps the Python API and exposes 9 `voxcpm_*` tools.

## When to use

Reach for these tools whenever the user says any of:

- "say this", "read this aloud", "generate speech for…"
- "create a voice that sounds like…" / "design a voice"
- "clone my voice", "clone this voice from <audio>"
- "make an audio file", "synthesize…"
- "what audio files have you generated?"

## Modes

| Mode | Trigger | Tool |
|------|---------|------|
| **Plain TTS** | Just text | `voxcpm_synthesize(text=...)` |
| **Voice design** | Text + description | `voxcpm_synthesize(text=..., voice_description=...)` or `voxcpm_describe_voice` |
| **Controllable clone** | Text + reference audio | `voxcpm_synthesize(text=..., reference_audio=...)` or `voxcpm_clone_voice` |
| **Ultimate clone** | Text + reference + transcript | `voxcpm_synthesize(text=..., reference_audio=..., prompt_text=...)` |

Voice design uses VoxCPM2's natural-language control token — the description goes in parentheses before the text automatically. Example description: `"warm female voice, slow and deliberate"`.

## Tool reference

| Tool | Purpose |
|------|---------|
| `voxcpm_health` | Check server is up; reports loaded model, device, torch version |
| `voxcpm_model_info` | Details on currently loaded model (sample rate, architecture) |
| `voxcpm_load_model` | Pre-load a specific model id; blocks until ready |
| `voxcpm_unload_model` | Drop the model and free GPU/CPU memory |
| `voxcpm_synthesize` | **Main tool** — one-shot text→WAV, auto-detects mode |
| `voxcpm_describe_voice` | Explicit voice-design shortcut |
| `voxcpm_clone_voice` | Explicit voice-clone shortcut |
| `voxcpm_list_outputs` | List previously-generated WAV files in cache |
| `voxcpm_clear_cache` | Delete cached WAVs (all, or by prefix) |

## Typical workflow

```python
# 1. Check server is alive (optional, skip on repeat calls)
voxcpm_health()
# → {"loaded": false, "default_model_id": "openbmb/VoxCPM2", ...}

# 2. Synthesize — model loads lazily on first call (~90s first run, cached after)
voxcpm_synthesize(
    text="Hello, this is Hermes speaking.",
    cfg_value=2.0,
    inference_timesteps=10,
)
# → {"output_path": "C:\\Data\\Hermes\\audio_cache\\voxcpm\\plain-....wav",
#    "duration_s": 2.1, "sample_rate": 48000, "mode": "plain", ...}

# 3. Voice design
voxcpm_synthesize(
    text="The meeting starts in five minutes.",
    voice_description="calm British male, measured pace",
)

# 4. Voice clone
voxcpm_clone_voice(
    text="This is a cloned voice.",
    reference_audio="C:\\path\\to\\reference.wav",
)
```

## Operational notes

- **First call is slow** — VoxCPM2 is a ~4–8 GB download on first use; cached under `%USERPROFILE%\.cache\huggingface\hub`. Plan for 2–3 minutes first time.
- **Model stays loaded** — once loaded the model stays in memory for the lifetime of the MCP server process. Subsequent calls are fast (~8s on CPU for a short sentence with VoxCPM-0.5B; faster on GPU).
- **CPU vs GPU** — `VOXCPM_DEVICE=auto` picks CUDA if available, else CPU. On CPU, VoxCPM2 is slow; VoxCPM-0.5B is faster but lower quality. Switch model with `voxcpm_load_model(model_id="openbmb/VoxCPM-0.5B", load_denoiser=False)`.
- **Optimization** — `VOXCPM_OPTIMIZE=0` in config (torch.compile off) for safety on Windows/CPU. Enable with `VOXCPM_OPTIMIZE=1` on CUDA for ~30% speedup.
- **Output files** — WAVs land in `C:\Data\Hermes\audio_cache\voxcpm\`. Use `voxcpm_list_outputs` to find them, `voxcpm_clear_cache` to purge.
- **HF symlinks** — `HF_HUB_DISABLE_SYMLINKS=1` is set in the MCP env; required on Windows without Developer Mode to avoid WinError 1314.

## Configuration (env vars passed via Hermes config)

| Variable | Default in config | Purpose |
|---|---|---|
| `VOXCPM_MODEL_ID` | `openbmb/VoxCPM2` | Default model (change to `openbmb/VoxCPM-0.5B` for speed) |
| `VOXCPM_DENOISER_ID` | `iic/speech_zipenhancer_ans_multiloss_16k_base` | ZipEnhancer denoiser for voice cloning |
| `VOXCPM_DEVICE` | `auto` | `auto` / `cpu` / `cuda` / `cuda:0` |
| `VOXCPM_OPTIMIZE` | `0` | `1` to enable torch.compile (GPU only) |
| `VOXCPM_OUTPUT_DIR` | `C:\Data\Hermes\audio_cache\voxcpm` | Where WAVs are saved |
| `HF_HUB_DISABLE_SYMLINKS` | `1` | Required on Windows without Developer Mode |

## Files

- Source: `C:\Data\Hermes\~\VoxCPM\` (main library)
- MCP package: `C:\Data\Hermes\~\VoxCPM\packages\voxcpm-mcp\`
- MCP binary: `C:\Data\Hermes\~\VoxCPM\.venv\Scripts\voxcpm-mcp.exe`
- Skill junction: `C:\Data\Hermes\skills\media\voxcpm` → `C:\Data\Hermes\~\VoxCPM\packages\voxcpm-mcp\skills\voxcpm`
- Hermes config: `mcp_servers.voxcpm` in `C:\Data\Hermes\config.yaml`
- Audio output: `C:\Data\Hermes\audio_cache\voxcpm\`
- Venv: `C:\Data\Hermes\~\VoxCPM\.venv\` (separate from Hermes venv — torch/ML deps isolated here)
