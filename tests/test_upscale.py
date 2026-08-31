from __future__ import annotations

import unittest
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from enhancer.upscale import Upscaler, target_dimensions


class TargetDimensionsTests(unittest.TestCase):
    def test_preserves_dimensions_without_resolution(self) -> None:
        self.assertEqual(target_dimensions(853, 480, None), (853, 480))

    def test_maps_named_resolution_and_preserves_aspect(self) -> None:
        self.assertEqual(target_dimensions(640, 360, "720p"), (1280, 720))
        self.assertEqual(target_dimensions(1440, 1080, "4k"), (2880, 2160))

    def test_rounds_derived_width_to_even_pixels(self) -> None:
        self.assertEqual(target_dimensions(1080, 1920, "4k"), (1216, 2160))

    def test_matching_height_is_a_no_op(self) -> None:
        self.assertEqual(target_dimensions(1920, 1080, "1080p"), (1920, 1080))

    def test_rejects_resolution_downgrade(self) -> None:
        with self.assertRaisesRegex(ValueError, "would downgrade"):
            target_dimensions(1920, 1080, "720p")

    def test_rejects_unknown_resolution(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown resolution"):
            target_dimensions(640, 360, "8k")


class _NearestX2:
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return F.interpolate(tensor, scale_factor=2, mode="nearest")


class TiledUpscalerTests(unittest.TestCase):
    def test_tiled_inference_stitches_without_seams(self) -> None:
        frame = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
        frame.setflags(write=False)
        upscaler = Upscaler.__new__(Upscaler)
        upscaler.device = torch.device("cpu")
        upscaler.dtype = torch.float32
        upscaler.model = _NearestX2()
        upscaler.scale = 2
        upscaler.tile_size = 3
        upscaler.tile_pad = 1

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            actual = upscaler.upscale(frame, width=14, height=10)
        expected = np.repeat(np.repeat(frame, 2, axis=0), 2, axis=1)

        np.testing.assert_array_equal(actual, expected)
        self.assertFalse(any("not writable" in str(item.message) for item in caught))


if __name__ == "__main__":
    unittest.main()
