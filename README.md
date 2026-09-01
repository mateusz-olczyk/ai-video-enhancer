# ai-video-enhancer

Enhance mp4 video frame rate and, optionally, resolution on macOS and Linux
(including WSL2 Ubuntu). The local pipeline uses
[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) for resolution and
[RIFE](https://github.com/hzwer/Practical-RIFE) for frame interpolation, with
PyTorch acceleration (MPS/CUDA/CPU fallback).

Without `--resolution`, the pipeline preserves the source resolution. It
always preserves audio, applies an initial denoise pass, and uses scene-cut
detection so it never morphs across cuts.

```text
input.mp4 ─► ffprobe / PySceneDetect ─► hqdn3d ─► Real-ESRGAN (optional) ─► RIFE ─► H.264 + original audio ─► output.mp4
```

## Setup (one time)

System prerequisites:

- macOS: `python3`, `git`, `unzip`, `curl`
- Ubuntu/WSL2: `python3`, `python3-venv`, `git`, `unzip`, `curl`
- Optional but recommended on Linux/WSL2: system `ffprobe`/`ffmpeg` for robust probing

If you want GPU acceleration on WSL2 with NVIDIA, install a CUDA-enabled PyTorch build that matches your environment using the official selector at [pytorch.org](https://pytorch.org/get-started/locally/). The rest of this project setup stays the same.

```bash
bash scripts/setup_venv.sh
source .venv/bin/activate
bash scripts/download_model.sh
```

`setup_venv.sh` installs the package and development tools, installs the repository's pre-commit hooks,
and injects cache env vars into the venv's activate script so all caches/models live under `<repo>/.cache/`:

| Var | Value |
| --- | --- |
| `XDG_CACHE_HOME` | `.cache/` |
| `PIP_CACHE_DIR` | `.cache/pip/` |
| `TORCH_HOME` | `.cache/torch/` |
| `HF_HOME` | `.cache/huggingface/` |
| `RIFE_MODEL_DIR` | `.cache/rife/` |
| `REALESRGAN_MODEL_DIR` | `.cache/realesrgan/` |
| `PYTORCH_ENABLE_MPS_FALLBACK` | `1` (macOS/MPS-oriented; harmless on Linux/WSL2) |

To wipe all caches and downloaded weights:

```bash
rm -rf .cache
```

## Run

```bash
enhancer path/to/input.mp4 -o output_60fps.mp4
```

Enhance a 720p source to 4K and 60 fps:

```bash
enhancer path/to/input.mp4 -o output_4k_60fps.mp4 --resolution 4k
```

Equivalent:

```bash
python -m enhancer.main path/to/input.mp4 -o output_60fps.mp4
```

Quick validation run (first 5 seconds only):

```bash
enhancer path/to/input.mp4 -o output_5s_60fps.mp4 --trim-end 5
```

CLI flags:

- `--version` — print the package name and version, then exit.
- `--target-fps` (default `60`)
- `--resolution {720p,1080p,4k}` — locally enhance to the named progressive
  resolution (`4k` is 2160p), preserving aspect ratio. Omit the flag to
  preserve the source dimensions. A target lower than the source is rejected.
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
5. **Enhance resolution** when requested, using tiled Real-ESRGAN x4 inference.
   The x4 result is resized to the exact target while preserving aspect ratio.
   A separate Rich progress row reports completed source frames and throughput.
6. **Interpolate** on Apple MPS (macOS), CUDA (Linux/WSL2 with NVIDIA), or CPU fallback.
7. **Encode** H.264 (CRF 17) and **mux** the original audio with `-c copy`.

Resolution enhancement runs before interpolation. This means Real-ESRGAN only
processes original source frames instead of every generated 60-fps frame, and
RIFE estimates motion from the enhanced detail. The tradeoff is that 4K RIFE
needs more device memory. Both models run entirely on the local machine. The
setup script downloads model weights, but processing never calls an external
model API.

## Project layout

```
pyproject.toml    # package metadata + enhancer console script
src/enhancer/
  main.py          # argparse CLI
  pipeline.py      # orchestration + sliding cache
  schedule.py      # multi-stage interpolation planner
  interpolate.py   # RIFE wrapper (MPS/CUDA/CPU selection)
  upscale.py       # tiled Real-ESRGAN resolution enhancement
  scene_detect.py  # PySceneDetect ContentDetector
  video_io.py      # ffmpeg probe/read/write/mux
scripts/
  setup_venv.sh
  download_model.sh
```
