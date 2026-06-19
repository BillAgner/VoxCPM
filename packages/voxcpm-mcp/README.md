# voxcpm-mcp

MCP server for [VoxCPM](https://github.com/OpenBMB/VoxCPM) — a local,
high-quality text-to-speech pipeline with **30-language** multilingual
support, **voice design** from natural-language descriptions, and
**voice cloning** from a short reference clip.

Wraps the OpenBMB/VoxCPM Python API and exposes it as a small set of
FastMCP tools over stdio. Designed to drop into the Hermes agent.

## Tools

| Tool | When to use |
|------|-------------|
| `voxcpm_health` | First call — confirms server is up, reports model/device state |
| `voxcpm_model_info` | What's currently loaded |
| `voxcpm_load_model` | Preload a specific model id (default `openbmb/VoxCPM2`) |
| `voxcpm_unload_model` | Drop the loaded model and free memory |
| `voxcpm_synthesize` | One-shot text → WAV (auto-detects clone / design / plain mode) |
| `voxcpm_describe_voice` | Explicit voice-design mode |
| `voxcpm_clone_voice` | Explicit voice-clone mode |
| `voxcpm_list_outputs` | List previously-generated WAV files |
| `voxcpm_clear_cache` | Delete cached outputs |

## Configuration (environment variables)

| Variable | Default | Purpose |
|----------|---------|---------|
| `VOXCPM_MODEL_ID` | `openbmb/VoxCPM2` | Default HuggingFace repo id |
| `VOXCPM_DENOISER_ID` | `iic/speech_zipenhancer_ans_multiloss_16k_base` | Denoiser model id; set to `none` to disable |
| `VOXCPM_DEVICE` | `auto` | `auto` / `cpu` / `cuda` / `cuda:N` |
| `VOXCPM_OPTIMIZE` | `1` | torch.compile optimization (set `0` to disable for debugging) |
| `VOXCPM_OUTPUT_DIR` | `C:\Data\Hermes\audio_cache\voxcpm` | Where WAVs are written |
| `VOXCPM_CACHE_DIR` | (HF default) | HuggingFace snapshot cache |
| `VOXCPM_LOCAL_FILES_ONLY` | `0` | Set `1` to forbid network downloads |
| `VOXCPM_MAX_LEN` | `4096` | Generation length cap |
| `VOXCPM_CACHE_MAX_BYTES` | (uncapped) | Soft cap on output dir size in bytes |

## Install

```sh
# From a checkout of the VoxCPM repo:
cd ~/VoxCPM
uv venv .venv --python 3.11
source .venv/Scripts/activate          # POSIX: source .venv/bin/activate
uv pip install -e .
cd packages/voxcpm-mcp
uv pip install -e .
```

The first call to `voxcpm_synthesize` triggers a model download
(~4–8 GB depending on the variant). Subsequent calls reuse the cached
snapshot.

## License

MIT.