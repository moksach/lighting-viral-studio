from __future__ import annotations

from typing import Optional, Tuple

from .contracts import VideoInfo
from .utils import clamp, even_int


def fit_crop_to_aspect(
    box: Optional[Tuple[int, int, int, int]],
    info: VideoInfo,
    aspect_w: int,
    aspect_h: int,
    padding_pct: float,
    min_frame_coverage: float,
    crop_mode: str,
    composition_bias: str = "Center lightning",
) -> Tuple[int, int, int, int, float]:
    """Return x, y, w, h, crop_confidence.

    The crop is stable for the entire event. It never animates per frame.
    """
    frame_w, frame_h = int(info.width), int(info.height)
    target = aspect_w / max(1, aspect_h)
    frame_aspect = frame_w / max(1, frame_h)

    if crop_mode == "Full frame / no smart crop" or abs(target - frame_aspect) < 1e-3:
        if abs(target - frame_aspect) < 1e-3:
            return 0, 0, even_int(frame_w), even_int(frame_h), 1.0
        box = None

    if crop_mode == "Center crop only":
        box = None

    if box is None:
        cx, cy = frame_w / 2.0, frame_h / 2.0
        box_w = frame_w * min_frame_coverage
        box_h = frame_h * min_frame_coverage
        confidence = 0.45 if crop_mode == "Smart per-strike crop" else 0.70
    else:
        x, y, w, h = box
        pad_x = max(24.0, w * padding_pct)
        pad_y = max(24.0, h * padding_pct)
        x1 = clamp(x - pad_x, 0, frame_w)
        y1 = clamp(y - pad_y, 0, frame_h)
        x2 = clamp(x + w + pad_x, 0, frame_w)
        y2 = clamp(y + h + pad_y, 0, frame_h)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        if composition_bias == "Lightning slightly high":
            cy += 0.08 * frame_h
        elif composition_bias == "Show more sky above":
            cy += 0.13 * frame_h
        elif composition_bias == "Show more ground/context":
            cy -= 0.06 * frame_h

        box_w = max(48.0, x2 - x1)
        box_h = max(48.0, y2 - y1)
        area_ratio = min(1.0, (w * h) / max(1.0, frame_w * frame_h))
        confidence = 0.65 + 0.30 * min(1.0, area_ratio * 35.0)

    if box_w / max(1.0, box_h) > target:
        crop_w = box_w
        crop_h = crop_w / target
    else:
        crop_h = box_h
        crop_w = crop_h * target

    # Preserve storm context; viral crops should not feel like random lightning pixels.
    crop_w = max(crop_w, frame_w * min_frame_coverage)
    crop_h = max(crop_h, frame_h * min_frame_coverage)

    if crop_w > frame_w:
        crop_w = frame_w
        crop_h = crop_w / target
    if crop_h > frame_h:
        crop_h = frame_h
        crop_w = crop_h * target

    crop_w = min(even_int(crop_w), even_int(frame_w))
    crop_h = min(even_int(crop_h), even_int(frame_h))
    x = int(round(cx - crop_w / 2.0))
    y = int(round(cy - crop_h / 2.0))
    x = max(0, min(frame_w - crop_w, x))
    y = max(0, min(frame_h - crop_h, y))
    x -= x % 2
    y -= y % 2
    return int(x), int(y), int(crop_w), int(crop_h), round(float(confidence), 3)
