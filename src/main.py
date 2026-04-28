"""CLI entry point.

Run from the repo root with the venv activated:

    python -m src.main path/to/input.mp4 -o output_60fps.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .pipeline import enhance


def main() -> None:
    p = argparse.ArgumentParser(description="Upscale FPS to 60 using RIFE on Apple MPS.")
    p.add_argument("input", type=Path, help="input video (mp4)")
    p.add_argument("-o", "--output", type=Path, required=True, help="output video path")
    p.add_argument("--target-fps", type=float, default=60.0, help="target fps (default: 60)")
    p.add_argument(
        "--interp-strategy",
        choices=["auto", "direct", "staged"],
        default="auto",
        help="RIFE interpolation strategy (default: auto)",
    )
    p.add_argument("--no-denoise", action="store_true", help="disable hqdn3d denoise pre-filter")
    p.add_argument("--scene-threshold", type=float, default=27.0, help="PySceneDetect ContentDetector threshold")
    p.add_argument(
        "--trim-end",
        type=float,
        default=None,
        metavar="SECONDS",
        help="trim input to the first N seconds before processing (handy for quick testing)",
    )
    args = p.parse_args()

    if not args.input.exists():
        p.error(f"input not found: {args.input}")
    if args.trim_end is not None and args.trim_end <= 0:
        p.error("--trim-end must be > 0")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    try:
        enhance(
            input_path=args.input,
            output_path=args.output,
            target_fps=args.target_fps,
            strategy=args.interp_strategy,
            denoise=not args.no_denoise,
            scene_threshold=args.scene_threshold,
            trim_end_sec=args.trim_end,
        )
    except KeyboardInterrupt:
        print("cancelled by user", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
