"""RIFE interpolation wrapper.

Loads Practical-RIFE (cloned by scripts/download_model.sh) and runs inference
on Apple's MPS GPU when available, with graceful CPU fallback.
"""

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


def _rife_source_dir() -> Path:
    base = Path(os.environ.get("RIFE_MODEL_DIR", Path(__file__).resolve().parents[2] / ".cache/rife"))
    return base / "Practical-RIFE"


def _select_device() -> torch.device:
    # MPS = Apple GPU; falls back to CPU if not built/available.
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class Interpolator:
    """Thin facade over Practical-RIFE's `Model` class."""

    def __init__(self, device: Optional[torch.device] = None) -> None:
        self.device = device or _select_device()
        src = _rife_source_dir()
        if not (src / "train_log" / "flownet.pkl").exists():
            raise FileNotFoundError(f"RIFE weights not found under {src}. Run scripts/download_model.sh first.")
        # Inject Practical-RIFE into sys.path so its `train_log.RIFE_HDv3` import works.
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))

        # Patch the hard-coded `device = cuda or cpu` globals in upstream
        # modules BEFORE constructing the model, so warp grids etc. are
        # allocated on MPS instead of CPU when we're on Apple Silicon.
        import model.warplayer as _warplayer  # type: ignore
        from train_log import RIFE_HDv3 as _rife_hdv3  # type: ignore

        _warplayer.device = self.device
        _rife_hdv3.device = self.device
        # Reset any grid cache that may have been keyed under the old device.
        _warplayer.backwarp_tenGrid = {}

        from train_log.RIFE_HDv3 import Model  # type: ignore

        self.model = Model()
        self.model.load_model(str(src / "train_log"), -1)
        self.model.eval()
        # Move flownet (the only submodule used at inference) onto our device.
        self.model.flownet.to(self.device)

    @torch.no_grad()
    def interpolate(self, frame_a: np.ndarray, frame_b: np.ndarray, timestep: float) -> np.ndarray:
        """Synthesize a frame at fractional `timestep` in [0,1] between A and B."""
        ta = self._to_tensor(frame_a)
        tb = self._to_tensor(frame_b)
        # RIFE requires H,W to be multiples of 32 -- pad on bottom/right then crop back.
        _, _, h, w = ta.shape
        ph = (32 - h % 32) % 32
        pw = (32 - w % 32) % 32
        if ph or pw:
            ta = F.pad(ta, (0, pw, 0, ph), mode="replicate")
            tb = F.pad(tb, (0, pw, 0, ph), mode="replicate")

        # Practical-RIFE >=4.x: `inference(img0, img1, timestep)`; older variants
        # accept only a midpoint and require recursion. We rely on v4.x.
        try:
            mid = self.model.inference(ta, tb, timestep)
        except TypeError:
            # Older API: only midpoint. Recurse to approximate fractional t.
            mid = self._recursive_midpoint(ta, tb, timestep)

        if ph or pw:
            mid = mid[:, :, :h, :w]
        return self._to_numpy(mid)

    def _recursive_midpoint(self, a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
        # Used only for legacy RIFE: split interval in half toward t.
        if abs(t - 0.5) < 1e-3:
            return self.model.inference(a, b)
        if t < 0.5:
            mid = self.model.inference(a, b)
            return self._recursive_midpoint(a, mid, t * 2)
        mid = self.model.inference(a, b)
        return self._recursive_midpoint(mid, b, (t - 0.5) * 2)

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        # HxWx3 uint8 RGB -> 1x3xHxW float in [0,1] on device.
        if not arr.flags.writeable:
            arr = np.array(arr, copy=True)
        t = torch.from_numpy(arr).to(self.device, non_blocking=True)
        t = t.permute(2, 0, 1).unsqueeze(0).float() / 255.0
        return t

    def _to_numpy(self, t: torch.Tensor) -> np.ndarray:
        out = (t.clamp(0, 1) * 255.0).byte().squeeze(0).permute(1, 2, 0).contiguous()
        return out.cpu().numpy()
