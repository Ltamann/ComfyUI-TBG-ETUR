"""Coordinate and mask primitives shared by ETUR tile paths."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def normalize_mask(mask, device=None, dtype=torch.float32):
    if mask is None:
        return None
    value = mask
    if value.ndim == 4 and value.shape[1] == 1 and value.shape[-1] != 1:
        value = value.permute(0, 2, 3, 1)
    if value.ndim == 3:
        value = value.unsqueeze(-1)
    if value.ndim != 4:
        raise ValueError(f"ETUR mask must be BHWC/BHW, got {tuple(value.shape)}")
    return value.to(device=device, dtype=dtype).clamp(0.0, 1.0)


def resize_mask(mask, height, width, device=None, dtype=torch.float32):
    value = normalize_mask(mask, device=device, dtype=dtype)
    if value is None or value.shape[1:3] == (height, width):
        return value
    return F.interpolate(
        value.permute(0, 3, 1, 2),
        size=(int(height), int(width)),
        mode="bilinear",
        align_corners=False,
    ).permute(0, 2, 3, 1).contiguous().clamp(0.0, 1.0)


@dataclass(frozen=True)
class TileGeometry:
    native_size: tuple[int, int]
    sampling_size: tuple[int, int]
    native_segment_size: tuple[int, int] | None = None
    sampling_segment_size: tuple[int, int] | None = None
    pid_work_size: tuple[int, int] | None = None

    @property
    def spaces(self):
        return {
            "native_tile": self.native_size,
            "sampling_tile": self.sampling_size,
            "native_segment_crop": self.native_segment_size,
            "sampling_segment_crop": self.sampling_segment_size,
            "pid_work_canvas": self.pid_work_size,
        }

    def to_sampling_mask(self, mask, device=None, dtype=torch.float32):
        return resize_mask(mask, self.sampling_size[1], self.sampling_size[0], device, dtype)

    def to_native_mask(self, mask, device=None, dtype=torch.float32):
        return resize_mask(mask, self.native_size[1], self.native_size[0], device, dtype)

    def to_pid_mask(self, mask, device=None, dtype=torch.float32):
        if self.pid_work_size is None:
            return normalize_mask(mask, device=device, dtype=dtype)
        return resize_mask(mask, self.pid_work_size[1], self.pid_work_size[0], device, dtype)
