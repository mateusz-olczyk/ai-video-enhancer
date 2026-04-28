"""ffmpeg-driven video I/O: probe, denoised raw-frame reader, encoder, audio mux.

Uses the ffmpeg binary bundled with `imageio-ffmpeg` so users don't need a
system-wide ffmpeg install on macOS.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import imageio_ffmpeg
import numpy as np


def ffmpeg_bin() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def ffprobe_bin() -> str:
    # imageio-ffmpeg ships ffmpeg only; rely on system ffprobe if present,
    # else parse metadata via ffmpeg's stderr (fallback below).
    sys_ffprobe = shutil.which("ffprobe")
    return sys_ffprobe or ""


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    num_frames: int
    has_audio: bool


def probe(path: Path) -> VideoInfo:
    """Return basic stream info. Prefers ffprobe; falls back to ffmpeg parse."""
    probe_cmd = ffprobe_bin()
    if probe_cmd:
        out = subprocess.check_output(
            [
                probe_cmd, "-v", "error", "-print_format", "json",
                "-show_streams", "-show_format", str(path),
            ],
            text=True,
        )
        data = json.loads(out)
        video = next(s for s in data["streams"] if s["codec_type"] == "video")
        audio = any(s["codec_type"] == "audio" for s in data["streams"])
        num, den = (int(x) for x in video["avg_frame_rate"].split("/"))
        fps = num / den if den else 0.0
        nb_frames = int(video.get("nb_frames") or 0)
        if not nb_frames:
            duration = float(data["format"].get("duration", 0.0))
            nb_frames = int(round(duration * fps))
        return VideoInfo(int(video["width"]), int(video["height"]), fps, nb_frames, audio)

    # Fallback: parse ffmpeg -i stderr
    proc = subprocess.run(
        [ffmpeg_bin(), "-i", str(path)], capture_output=True, text=True
    )
    err = proc.stderr
    import re
    m_dim = re.search(r"(\d{2,5})x(\d{2,5})", err)
    m_fps = re.search(r"(\d+(?:\.\d+)?)\s+fps", err)
    m_dur = re.search(r"Duration:\s+(\d+):(\d+):(\d+\.\d+)", err)
    has_audio = "Audio:" in err
    if not (m_dim and m_fps and m_dur):
        raise RuntimeError(f"failed to probe {path}; install ffprobe for reliable metadata")
    w, h = int(m_dim.group(1)), int(m_dim.group(2))
    fps = float(m_fps.group(1))
    dur = int(m_dur.group(1)) * 3600 + int(m_dur.group(2)) * 60 + float(m_dur.group(3))
    return VideoInfo(w, h, fps, int(round(dur * fps)), has_audio)


def iter_denoised_frames(
    path: Path, width: int, height: int, denoise: bool = True
) -> Iterator[np.ndarray]:
    """Yield decoded frames as HxWx3 uint8 RGB numpy arrays.

    A `hqdn3d` filter is applied here as the initial noise-cancellation step:
    denoising BEFORE interpolation prevents RIFE from amplifying sensor noise
    into temporally-incoherent artefacts.
    """
    vf_chain = []
    if denoise:
        vf_chain.append("hqdn3d=1.5:1.5:6:6")  # mild spatial+temporal denoise
    vf = ",".join(vf_chain) if vf_chain else "null"

    cmd = [
        ffmpeg_bin(),
        "-loglevel", "error",
        "-i", str(path),
        "-vf", vf,
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-",
    ]
    frame_size = width * height * 3
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=10**8,
        start_new_session=True,
    )
    assert proc.stdout is not None
    try:
        while True:
            buf = proc.stdout.read(frame_size)
            if len(buf) < frame_size:
                break
            yield np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


class FrameEncoder:
    """Streams RGB frames into ffmpeg stdin and produces an H.264 mp4."""

    def __init__(self, out_path: Path, width: int, height: int, fps: float, crf: int = 17):
        self.out_path = out_path
        self.width = width
        self.height = height
        self.fps = fps
        # CRF 17 is visually near-lossless; pix_fmt yuv420p for broad compat.
        self.cmd = [
            ffmpeg_bin(),
            "-loglevel", "error",
            "-y",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}",
            "-r", f"{fps}",
            "-i", "-",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(out_path),
        ]
        self.proc: Optional[subprocess.Popen] = None
        self._cancelled = False
        self._broken = False

    def cancel(self) -> None:
        """Mark the encoder as user-cancelled so non-zero ffmpeg exits in
        __exit__ are treated as expected (no RuntimeError)."""
        self._cancelled = True

    def __enter__(self) -> "FrameEncoder":
        # start_new_session=True puts ffmpeg in its own process group, so a
        # terminal Ctrl+C (SIGINT to the foreground group) is delivered only
        # to Python -- ffmpeg keeps running until we close its stdin, and can
        # then finalize the mp4 trailer cleanly.
        self.proc = subprocess.Popen(
            self.cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10**8,
            start_new_session=True,
        )
        return self

    def write(self, frame: np.ndarray) -> None:
        assert self.proc is not None and self.proc.stdin is not None
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        try:
            self.proc.stdin.write(frame.tobytes())
        except BrokenPipeError:
            # Encoder died (e.g. disk full). Stop quietly; __exit__ surfaces
            # the real error if there is one.
            self._broken = True

    def __exit__(self, exc_type, exc, tb) -> None:
        assert self.proc is not None and self.proc.stdin is not None
        try:
            self.proc.stdin.close()
        except BrokenPipeError:
            self._broken = True
        # On Ctrl+C we still want ffmpeg to finalize the trailer for whatever
        # frames it already received, producing a playable (shorter) mp4.
        try:
            rc = self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
            try:
                rc = self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                rc = self.proc.wait()
        stderr_bytes = b""
        if self.proc.stderr is not None:
            try:
                stderr_bytes = self.proc.stderr.read() or b""
            except Exception:
                pass
            finally:
                self.proc.stderr.close()
        # Don't surface ffmpeg failures when:
        # - another exception is already propagating (exc_type set)
        # - the pipeline marked us cancelled (Ctrl+C path)
        # - our stdin pipe broke mid-stream (encoder died early; that's the
        #   real error to surface, but only if nothing else explains it)
        if rc != 0 and exc_type is None and not self._cancelled:
            msg = stderr_bytes.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"ffmpeg encoder exited with code {rc}" + (f": {msg}" if msg else "")
            )


def mux_audio(video_only: Path, source_with_audio: Path, out: Path) -> None:
    """Copy audio from source into the new video, no re-encode."""
    cmd = [
        ffmpeg_bin(),
        "-loglevel", "error",
        "-y",
        "-i", str(video_only),
        "-i", str(source_with_audio),
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c", "copy",
        "-shortest",
        str(out),
    ]
    subprocess.check_call(cmd, start_new_session=True)
