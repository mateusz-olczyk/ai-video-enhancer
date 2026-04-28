"""End-to-end orchestration: probe -> denoise -> scene detect -> schedule
-> RIFE interpolation -> encode -> mux original audio.

Crucial pipeline steps are commented inline as requested.
"""
from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterator, Optional

import numpy as np
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from .interpolate import Interpolator
from .scene_detect import detect_scene_cuts
from .schedule import FrameKey, Recipe, build_schedule
from .video_io import FrameEncoder, VideoInfo, ffmpeg_bin, iter_denoised_frames, mux_audio, probe

console = Console(highlight=False)


def enhance(
    input_path: Path,
    output_path: Path,
    target_fps: float = 60.0,
    strategy: str = "auto",
    denoise: bool = True,
    scene_threshold: float = 27.0,
    trim_end_sec: Optional[float] = None,
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Step 0 (optional): trim the input down to the first N seconds. Re-encoding
        # (rather than -c copy) guarantees a frame-accurate cut so the schedule's
        # source-frame indices line up with what ffmpeg actually decodes downstream.
        work_input = input_path
        if trim_end_sec is not None:
            work_input = tmp / "trimmed.mp4"
            with console.status(f"[trim] writing first {trim_end_sec:.3f}s to temp clip"):
                _trim_input(input_path, work_input, trim_end_sec)
            console.log(f"[trim] wrote first {trim_end_sec:.3f}s to temp clip")

        _run_pipeline(
            input_path=work_input,
            output_path=output_path,
            tmpdir=tmp,
            target_fps=target_fps,
            strategy=strategy,
            denoise=denoise,
            scene_threshold=scene_threshold,
        )


def _trim_input(src: Path, dst: Path, seconds: float) -> None:
    cmd = [
        ffmpeg_bin(),
        "-loglevel", "error",
        "-y",
        "-i", str(src),
        "-t", f"{seconds}",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "16",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        str(dst),
    ]
    subprocess.check_call(cmd, start_new_session=True)


def _run_pipeline(
    input_path: Path,
    output_path: Path,
    tmpdir: Path,
    target_fps: float,
    strategy: str,
    denoise: bool,
    scene_threshold: float,
) -> None:
    with console.status("[probe] reading metadata"):
        info: VideoInfo = probe(input_path)
    console.log(f"[probe] {info.width}x{info.height} @ {info.fps:.3f}fps  frames={info.num_frames}  audio={info.has_audio}")

    # Step 1: scene-cut detection (runs on the ORIGINAL file -- denoise would
    # smooth the signal and weaken cut detection).
    with console.status("[scenes] detecting cuts"):
        cuts = detect_scene_cuts(input_path, threshold=scene_threshold)
    console.log(f"[scenes] {len(cuts)} cut(s) found")

    # Step 2: build the per-output-frame recipe list. Selecting `staged` for
    # non-integer ratios (24->60) bridges the largest motion gaps via t=0.5
    # midpoint calls before the final fractional pass.
    with console.status("[schedule] building recipe"):
        recipes = build_schedule(
            src_fps=info.fps,
            target_fps=target_fps,
            src_num_frames=info.num_frames,
            scene_cuts=cuts,
            strategy=strategy,
        )
    console.log(f"[schedule] strategy={strategy} -> {len(recipes)} output frames")

    # Step 3: load RIFE on Apple GPU (MPS) with CPU fallback.
    with console.status("[rife] loading model"):
        interp = Interpolator()
    console.log(f"[rife] device={interp.device}")

    # Step 4: stream frames through the pipeline. We keep a small sliding
    # buffer of source frames + a cache of stage-A midpoints. The recipes
    # arrive in temporal order, so we can evict source frames once no
    # remaining recipe references them.
    src_iter = iter_denoised_frames(input_path, info.width, info.height, denoise=denoise)

    video_only = tmpdir / "video_only.mp4"
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[progress.percentage]{task.percentage:>5.1f}%"),
        TimeElapsedColumn(),
        TextColumn("<"),
        TimeRemainingColumn(),
        TextColumn("{task.fields[speed]:.2f} frame/s"),
        console=console,
        transient=False,
    )
    interrupted = False
    with progress, FrameEncoder(video_only, info.width, info.height, target_fps) as enc:
        task = progress.add_task("interp", total=len(recipes), speed=0.0)
        t0 = time.monotonic()
        try:
            for i, frame in enumerate(_execute(recipes, src_iter, interp), start=1):
                enc.write(frame)
                elapsed = max(time.monotonic() - t0, 1e-6)
                progress.update(task, advance=1, speed=i / elapsed)
        except KeyboardInterrupt:
            interrupted = True
            enc.cancel()
            console.log("[cancel] stopping; finalizing partial output...")
    # Encoder context-exit above flushes the trailer so video_only.mp4 is
    # playable even on an interrupted run.

    # Step 5: mux the ORIGINAL audio in unchanged (-c copy, no re-encode).
    # `mux_audio` uses -shortest, so audio is auto-trimmed to the partial
    # video's duration when interrupted.
    if info.has_audio:
        with console.status("[mux] copying original audio into output"):
            mux_audio(video_only, input_path, output_path)
        console.log("[mux] copied original audio into output")
    else:
        with console.status("[mux] no audio stream; copying video only"):
            output_path.write_bytes(video_only.read_bytes())
        console.log("[mux] no audio stream; copied video only")

    if interrupted:
        console.log(f"[done] partial {output_path}")
        raise KeyboardInterrupt
    console.log(f"[done] {output_path}")


def _execute(
    recipes,
    src_iter: Iterator[np.ndarray],
    interp: Interpolator,
) -> Iterator[np.ndarray]:
    """Walk recipes in order, lazily pulling source frames and caching
    midpoints. Yields the synthesized output frame for each recipe.
    """
    # Highest source index already needed by some past or current recipe.
    src_buf: Dict[int, np.ndarray] = {}
    mid_cache: Dict[FrameKey, np.ndarray] = {}
    next_src_idx = 0

    def ensure_src(i: int) -> np.ndarray:
        nonlocal next_src_idx
        while next_src_idx <= i:
            try:
                frame = next(src_iter)
            except StopIteration:
                # Out of frames: pad by repeating the last we saw.
                if not src_buf:
                    raise
                last = max(src_buf)
                src_buf[next_src_idx] = src_buf[last]
                next_src_idx += 1
                continue
            src_buf[next_src_idx] = frame
            next_src_idx += 1
        return src_buf[i]

    def resolve(key: FrameKey) -> np.ndarray:
        if key.is_source:
            return ensure_src(key.src_a)
        if key in mid_cache:
            return mid_cache[key]
        # Stage-A midpoint: t=0.5 between two adjacent source frames -> highest
        # quality RIFE call.
        a = ensure_src(key.src_a)
        b = ensure_src(key.src_b)
        m = interp.interpolate(a, b, 0.5)
        mid_cache[key] = m
        return m

    # Pre-compute, per recipe, the maximum source index it touches so we can
    # evict frames/midpoints we won't need again.
    max_src_after = _compute_eviction_horizon(recipes)

    for idx, recipe in enumerate(recipes):
        if recipe.is_copy:
            yield resolve(recipe.left).copy()
        else:
            left = resolve(recipe.left)
            right = resolve(recipe.right)
            yield interp.interpolate(left, right, recipe.t)

        # Evict caches: drop source frames whose index < min_needed_from_now.
        min_needed = max_src_after[idx]
        for k in list(src_buf):
            if k < min_needed:
                del src_buf[k]
        for k in list(mid_cache):
            if k.src_b < min_needed:
                del mid_cache[k]


def _compute_eviction_horizon(recipes) -> list[int]:
    """For each recipe i, return the smallest source index that ANY recipe at
    j > i still needs. Anything below this is safe to evict after recipe i.
    """
    n = len(recipes)
    horizon = [0] * n
    # Walk backwards.
    running_min = 1 << 30
    for i in range(n - 1, -1, -1):
        r: Recipe = recipes[i]
        used = min(r.left.src_a, r.right.src_a)
        running_min = min(running_min, used)
        horizon[i] = running_min
    # Shift by one: after emitting recipe i we only care about recipes > i.
    shifted = horizon[1:] + [1 << 30]
    return shifted
