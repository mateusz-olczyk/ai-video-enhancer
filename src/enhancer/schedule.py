"""Multi-stage interpolation planner.

Given a source fps and a target fps, decide for each output frame which input
frames feed RIFE and at what timestep, optionally going through an
intermediate doubled-fps stage so that the largest motion gaps are bridged
with high-quality midpoint (t=0.5) calls.

Strategy choice:
  * `direct`  - one fractional RIFE call per non-aligned output frame.
  * `staged`  - first synthesize midpoints between every source pair (the
                doubled-fps stage), then derive the final output from that
                denser timeline. The remaining fractional jumps are smaller
                (~half the motion per call), reducing morph artifacts.
  * `auto`    - integer 2x ratio -> midpoint-only; non-integer -> staged;
                any other ratio with `target % src == 0` -> direct copies +
                midpoints; everything else -> direct fractional.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set


@dataclass(frozen=True)
class FrameKey:
    """Identifies a frame in the input timeline.

    src_a == src_b -> raw source frame at that index.
    src_a + 1 == src_b -> midpoint synthesized between two source frames
    (the "doubled-fps" stage frame).
    """

    src_a: int
    src_b: int

    @property
    def is_source(self) -> bool:
        return self.src_a == self.src_b


@dataclass
class Recipe:
    """How to produce a single output frame."""

    left: FrameKey
    right: FrameKey
    t: float  # 0.0 -> emit left as-is; 1.0 -> emit right; else interpolate

    @property
    def is_copy(self) -> bool:
        return self.t <= 0.0 or self.left == self.right


def _src(i: int) -> FrameKey:
    return FrameKey(i, i)


def _mid(a: int) -> FrameKey:
    return FrameKey(a, a + 1)


def build_schedule(
    src_fps: float,
    target_fps: float,
    src_num_frames: int,
    scene_cuts: Set[int],
    strategy: str = "auto",
) -> List[Recipe]:
    """Return one Recipe per output frame.

    `scene_cuts` is the set of source frame indices that BEGIN a new scene;
    when an output's bracketing pair straddles such a cut we duplicate the
    left (previous) source frame instead of blending across the cut.
    """
    target_num_frames = int(round(src_num_frames * target_fps / src_fps))
    ratio = target_fps / src_fps

    if strategy == "auto":
        if abs(ratio - round(ratio)) < 1e-6 and abs(ratio - 2.0) < 1e-6:
            strategy = "direct"  # 2x: midpoint-only; "direct" handles it perfectly
        elif abs(ratio - round(ratio)) < 1e-6:
            strategy = "direct"  # any integer ratio: trivial copies + midpoints
        else:
            strategy = "staged"  # non-integer (the 24->60 case): two-stage

    if strategy == "direct":
        return _build_direct(src_fps, target_fps, src_num_frames, target_num_frames, scene_cuts)
    if strategy == "staged":
        return _build_staged(src_fps, target_fps, src_num_frames, target_num_frames, scene_cuts)
    raise ValueError(f"unknown strategy: {strategy}")


def _straddles_cut(left_src: int, right_src: int, scene_cuts: Set[int]) -> bool:
    # A cut at index k means frame k is the FIRST frame of the new scene,
    # so the boundary lies between k-1 and k. We must avoid blending any
    # pair (a, b) with a < k <= b.
    for k in range(left_src + 1, right_src + 1):
        if k in scene_cuts:
            return True
    return False


def _build_direct(
    src_fps: float,
    target_fps: float,
    src_num: int,
    target_num: int,
    cuts: Set[int],
) -> List[Recipe]:
    out: List[Recipe] = []
    for i in range(target_num):
        t_src = i * src_fps / target_fps
        a = int(t_src)
        frac = t_src - a
        b = min(a + 1, src_num - 1)
        if a >= src_num - 1 or frac < 1e-6:
            out.append(Recipe(_src(min(a, src_num - 1)), _src(min(a, src_num - 1)), 0.0))
            continue
        # Scene cut: emit `a` again (duplicate) instead of blending across.
        if _straddles_cut(a, b, cuts):
            out.append(Recipe(_src(a), _src(a), 0.0))
            continue
        out.append(Recipe(_src(a), _src(b), frac))
    return out


def _build_staged(
    src_fps: float,
    target_fps: float,
    src_num: int,
    target_num: int,
    cuts: Set[int],
) -> List[Recipe]:
    """Stage-A: every source pair (a, a+1) gets a midpoint. Stage-B: final
    output is produced by walking the doubled-fps virtual timeline.
    """
    # Doubled timeline length in "half-source" units: each source frame is at
    # index 2*i, each midpoint is at 2*i + 1. There are 2*src_num - 1 frames.
    dbl_num = 2 * src_num - 1
    dbl_fps = src_fps * 2.0

    out: List[Recipe] = []
    for i in range(target_num):
        # Position in doubled-fps timeline (float index).
        t_dbl = i * dbl_fps / target_fps
        l = int(t_dbl)
        frac = t_dbl - l
        r = min(l + 1, dbl_num - 1)

        left = _key_from_dbl(l)
        right = _key_from_dbl(r)

        # Source-frame extents this recipe touches (for cut checking).
        left_src = left.src_a
        right_src = right.src_b
        if l >= dbl_num - 1 or frac < 1e-6:
            out.append(Recipe(left, left, 0.0))
            continue
        if _straddles_cut(left_src, right_src, cuts):
            # Replace with a copy of the previous source frame to avoid morphing
            # across the cut at any stage (A or B).
            prev_src = left_src
            out.append(Recipe(_src(prev_src), _src(prev_src), 0.0))
            continue
        out.append(Recipe(left, right, frac))
    return out


def _key_from_dbl(idx: int) -> FrameKey:
    """Map doubled-fps index -> FrameKey in the source-frame namespace."""
    if idx % 2 == 0:
        return _src(idx // 2)
    return _mid(idx // 2)
