from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class DisplayStats:
    device: str
    frame_index: int
    fps: float
    model_ms: float
    total_ms: float
    size: str


class FpsMeter:
    def __init__(self, alpha: float = 0.12) -> None:
        self.alpha = alpha
        self.value = 0.0

    def update(self, sample_fps: float) -> float:
        if sample_fps <= 0:
            return self.value
        if self.value == 0:
            self.value = sample_fps
        else:
            self.value = self.alpha * sample_fps + (1.0 - self.alpha) * self.value
        return self.value


def compose_side_by_side(
    source_bgr: np.ndarray,
    output_bgr: np.ndarray,
    stats: DisplayStats,
) -> np.ndarray:
    if source_bgr.shape[:2] != output_bgr.shape[:2]:
        h, w = source_bgr.shape[:2]
        output_bgr = cv2.resize(output_bgr, (w, h), interpolation=cv2.INTER_AREA)

    h = source_bgr.shape[0]
    separator = np.full((h, 2, 3), 34, dtype=np.uint8)
    canvas = np.concatenate([source_bgr, separator, output_bgr], axis=1)

    right_x = source_bgr.shape[1] + separator.shape[1]
    _label(canvas, "Input", (12, 28))
    _label(canvas, "TSRNN_BT1", (right_x + 12, 28))

    footer = (
        f"{stats.device} | {stats.size} | frame {stats.frame_index} | "
        f"{stats.fps:.1f} FPS | model {stats.model_ms:.1f} ms | total {stats.total_ms:.1f} ms"
    )
    _label(canvas, footer, (12, h - 14), scale=0.52)
    return canvas


def _label(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float = 0.7,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    pad = 6
    cv2.rectangle(
        image,
        (x - pad, y - th - pad),
        (x + tw + pad, y + baseline + pad),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        image,
        text,
        (x, y),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
