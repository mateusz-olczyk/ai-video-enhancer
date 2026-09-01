"""Local Real-ESRGAN resolution enhancement.

The model runs on the same PyTorch device selection used by RIFE. Inference is
tiled so 4K targets do not require the complete source frame to be resident in
GPU memory at once.
"""

import os
from pathlib import Path
from typing import Final

import cv2
import numpy as np
import torch
from spandrel import ImageModelDescriptor, ModelLoader

TARGET_HEIGHTS: Final[dict[str, int]] = {
    "720p": 720,
    "1080p": 1080,
    "4k": 2160,
}


def target_dimensions(width: int, height: int, resolution: str | None) -> tuple[int, int]:
    """Resolve a named target while preserving display aspect ratio.

    Resolution names follow the conventional progressive-scan height. The
    derived width is rounded to an even number for yuv420p compatibility.
    """
    if width <= 0 or height <= 0:
        raise ValueError("source dimensions must be positive")
    if resolution is None:
        return width, height
    try:
        target_height = TARGET_HEIGHTS[resolution]
    except KeyError as exc:
        choices = ", ".join(TARGET_HEIGHTS)
        raise ValueError(f"unknown resolution {resolution!r}; choose from {choices}") from exc

    if height > target_height:
        raise ValueError(
            f"--resolution {resolution} would downgrade {width}x{height}; "
            "omit the flag to preserve the original resolution"
        )
    if height == target_height:
        return width, height

    scaled_width = width * target_height / height
    target_width = max(2, int(round(scaled_width / 2.0)) * 2)
    return target_width, target_height


def _model_path() -> Path:
    default_root = Path(__file__).resolve().parents[2] / ".cache" / "realesrgan"
    root = Path(os.environ.get("REALESRGAN_MODEL_DIR", default_root))
    return root / "RealESRGAN_x4plus.pth"


def _select_device() -> torch.device:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class Upscaler:
    """Tiled Real-ESRGAN x4 inference with arbitrary final output dimensions."""

    def __init__(
        self,
        device: torch.device | None = None,
        *,
        model_path: Path | None = None,
        tile_size: int = 256,
        tile_pad: int = 16,
    ) -> None:
        path = model_path or _model_path()
        if not path.exists():
            raise FileNotFoundError(f"Real-ESRGAN weights not found at {path}. " "Run scripts/download_model.sh first.")

        descriptor = ModelLoader().load_from_file(path)
        if not isinstance(descriptor, ImageModelDescriptor):
            raise TypeError(f"expected an image super-resolution model at {path}")
        if descriptor.scale <= 1:
            raise ValueError(f"model at {path} does not increase resolution")

        self.device = device or _select_device()
        self.dtype = torch.float16 if self.device.type == "cuda" and descriptor.supports_half else torch.float32
        self.model = descriptor.to(device=self.device, dtype=self.dtype).eval()
        self.scale = int(descriptor.scale)
        self.tile_size = max(32, int(tile_size))
        self.tile_pad = max(0, int(tile_pad))

    @torch.inference_mode()
    def upscale(self, frame: np.ndarray, width: int, height: int) -> np.ndarray:
        """Enhance one RGB uint8 frame and resize it to the exact target."""
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("expected an HxWx3 RGB frame")
        if width <= 0 or height <= 0:
            raise ValueError("target dimensions must be positive")

        src_height, src_width = frame.shape[:2]
        native = np.empty(
            (src_height * self.scale, src_width * self.scale, 3),
            dtype=np.uint8,
        )

        for top in range(0, src_height, self.tile_size):
            bottom = min(top + self.tile_size, src_height)
            for left in range(0, src_width, self.tile_size):
                right = min(left + self.tile_size, src_width)
                pad_top = max(0, top - self.tile_pad)
                pad_bottom = min(src_height, bottom + self.tile_pad)
                pad_left = max(0, left - self.tile_pad)
                pad_right = min(src_width, right + self.tile_pad)

                patch = np.array(
                    frame[pad_top:pad_bottom, pad_left:pad_right],
                    copy=True,
                    order="C",
                )
                tensor = torch.from_numpy(patch).to(
                    device=self.device,
                    dtype=self.dtype,
                    non_blocking=True,
                )
                tensor = tensor.permute(2, 0, 1).unsqueeze(0) / 255.0
                enhanced = self.model(tensor).clamp_(0, 1)
                enhanced = (enhanced * 255.0).byte().squeeze(0).permute(1, 2, 0).contiguous().cpu().numpy()

                crop_top = (top - pad_top) * self.scale
                crop_bottom = crop_top + (bottom - top) * self.scale
                crop_left = (left - pad_left) * self.scale
                crop_right = crop_left + (right - left) * self.scale
                native[
                    top * self.scale : bottom * self.scale,
                    left * self.scale : right * self.scale,
                ] = enhanced[crop_top:crop_bottom, crop_left:crop_right]

        if native.shape[1] == width and native.shape[0] == height:
            return native
        return cv2.resize(native, (width, height), interpolation=cv2.INTER_LANCZOS4)
