import unittest

import numpy as np

from enhancer.pipeline import _execute
from enhancer.schedule import FrameKey, Recipe


class _FakeUpscaler:
    def __init__(self) -> None:
        self.calls = 0

    def upscale(self, frame: np.ndarray, width: int, height: int) -> np.ndarray:
        self.calls += 1
        return np.full((height, width, 3), int(frame[0, 0, 0]) + 10, dtype=np.uint8)


class _FakeInterpolator:
    def interpolate(
        self,
        frame_a: np.ndarray,
        frame_b: np.ndarray,
        timestep: float,
    ) -> np.ndarray:
        self.input_shape = frame_a.shape
        return (frame_a.astype(np.float32) * (1.0 - timestep) + frame_b.astype(np.float32) * timestep).astype(np.uint8)


class PipelineResolutionTests(unittest.TestCase):
    def test_upscales_each_source_once_before_interpolation(self) -> None:
        source_zero = FrameKey(0, 0)
        source_one = FrameKey(1, 1)
        recipes = [
            Recipe(source_zero, source_zero, 0.0),
            Recipe(source_zero, source_one, 0.5),
            Recipe(source_one, source_one, 0.0),
        ]
        source_frames = iter(
            [
                np.zeros((2, 2, 3), dtype=np.uint8),
                np.full((2, 2, 3), 2, dtype=np.uint8),
            ]
        )
        upscaler = _FakeUpscaler()
        interpolator = _FakeInterpolator()
        progress_updates = 0

        def on_upscale() -> None:
            nonlocal progress_updates
            progress_updates += 1

        output = list(
            _execute(
                recipes,
                source_frames,
                interpolator,  # type: ignore[arg-type]
                upscaler=upscaler,  # type: ignore[arg-type]
                output_dimensions=(4, 4),
                on_upscale=on_upscale,
            )
        )

        self.assertEqual(upscaler.calls, 2)
        self.assertEqual(progress_updates, 2)
        self.assertEqual(interpolator.input_shape, (4, 4, 3))
        self.assertEqual([int(frame[0, 0, 0]) for frame in output], [10, 11, 12])


if __name__ == "__main__":
    unittest.main()
