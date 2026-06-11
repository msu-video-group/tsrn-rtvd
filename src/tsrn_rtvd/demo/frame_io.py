from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch


@dataclass(frozen=True)
class FrameSize:
    width: int
    height: int


def resize_for_model(
    frame_bgr: np.ndarray,
    max_width: int,
    max_height: int,
    multiple: int = 8,
) -> np.ndarray:
    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError("Empty camera frame.")

    h, w = frame_bgr.shape[:2]
    scale = 1.0
    if max_width > 0:
        scale = min(scale, max_width / float(w))
    if max_height > 0:
        scale = min(scale, max_height / float(h))

    target_w = max(multiple, int(w * scale) // multiple * multiple)
    target_h = max(multiple, int(h * scale) // multiple * multiple)

    if target_w == w and target_h == h:
        return frame_bgr

    return cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)


def bgr_to_tensor(
    frame_bgr: np.ndarray,
    device: torch.device,
    channels_last: bool,
) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    chw = np.ascontiguousarray(rgb.transpose(2, 0, 1))
    tensor = torch.from_numpy(chw).to(device=device, dtype=torch.float32).div_(255.0)
    tensor = tensor.unsqueeze(0)
    if channels_last:
        tensor = tensor.contiguous(memory_format=torch.channels_last)
    return tensor


def tensor_to_bgr(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.detach().float().clamp(0.0, 1.0).squeeze(0)
    image = image.permute(1, 2, 0).cpu().numpy()
    rgb = np.clip(image * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
