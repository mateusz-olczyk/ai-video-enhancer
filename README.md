# ai-video-enhancer

Upscale 24/30 fps mp4 video to 60 fps on macOS and Linux (including WSL2 Ubuntu) using [RIFE](https://github.com/hzwer/Practical-RIFE) with PyTorch acceleration (MPS/CUDA/CPU fallback).

The pipeline preserves resolution and audio, applies an initial denoise pass, and uses scene-cut detection so it never morphs across cuts.

```text
input.mp4 ─► ffprobe ─► hqdn3d denoise ─► PySceneDetect ─► RIFE (staged) ─► H.264 + original audio ─► output.mp4
```

## Setup (one time)

System prerequisites:

- macOS: `python3`, `git`, `unzip`
- Ubuntu/WSL2: `python3`, `python3-venv`, `git`, `unzip`
- Optional but recommended on Linux/WSL2: system `ffprobe`/`ffmpeg` for robust probing

If you want GPU acceleration on WSL2 with NVIDIA, install a CUDA-enabled PyTorch build that matches your environment using the official selector at [pytorch.org](https://pytorch.org/get-started/locally/). The rest of this project setup stays the same.

```bash
bash scripts/setup_venv.sh
source .venv/bin/activate
bash scripts/download_model.sh
```

`setup_venv.sh` injects cache env vars into the venv's activate script so all caches/models live under `<repo>/.cache/`:

| Var | Value |
| --- | --- |
| `XDG_CACHE_HOME` | `.cache/` |
| `PIP_CACHE_DIR` | `.cache/pip/` |
| `TORCH_HOME` | `.cache/torch/` |
| `HF_HOME` | `.cache/huggingface/` |
| `RIFE_MODEL_DIR` | `.cache/rife/` |
| `PYTORCH_ENABLE_MPS_FALLBACK` | `1` (macOS/MPS-oriented; harmless on Linux/WSL2) |

To wipe all caches and downloaded weights:

```bash
rm -rf .cache
```

## Run

```bash
python -m src.main path/to/input.mp4 -o output_60fps.mp4
```

Quick validation run (first 5 seconds only):

```bash
python -m src.main path/to/input.mp4 -o output_5s_60fps.mp4 --trim-end 5
```

CLI flags:

- `--target-fps` (default `60`)
- `--interp-strategy {auto,direct,staged}` — `auto` picks `staged` for non-integer ratios (e.g. 24→60) and `direct` for integer ratios (e.g. 30→60).
- `--no-denoise` — skip the `hqdn3d` pre-filter.
- `--scene-threshold` — PySceneDetect `ContentDetector` threshold (default `27`).

## How it works

1. **Probe** with `ffprobe` (or ffmpeg fallback) for resolution, fps, frame count.
2. **Scene detection** on the original video (denoise would weaken cut signal).
3. **Schedule** every 60 fps output frame:
   - Integer 2x ratios (e.g. 30→60): every other output is a midpoint RIFE call (`t=0.5`, optimal accuracy).
   - Non-integer ratios (24→60): two-stage. Stage A inserts midpoints between every source pair (the doubled-fps stream); stage B fractional-interpolates between adjacent stage-A frames where needed. Smaller motion gaps per RIFE call than a single 0.4 / 0.8 jump from raw source.
   - Where a scene cut lies between bracketing frames, duplicate the previous frame (no morph).
4. **Denoise + decode** via `ffmpeg -vf hqdn3d=1.5:1.5:6:6` piped as raw RGB.
5. **Interpolate** on Apple MPS (macOS), CUDA (Linux/WSL2 with NVIDIA), or CPU fallback.
6. **Encode** H.264 (CRF 17) and **mux** the original audio with `-c copy`.

## Project layout

```
src/
  main.py          # argparse CLI
  pipeline.py      # orchestration + sliding cache
  schedule.py      # multi-stage interpolation planner
  interpolate.py   # RIFE wrapper (MPS/CUDA/CPU selection)
  scene_detect.py  # PySceneDetect ContentDetector
  video_io.py      # ffmpeg probe/read/write/mux
scripts/
  setup_venv.sh
  download_model.sh
```
