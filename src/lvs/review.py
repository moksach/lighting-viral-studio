from __future__ import annotations

from dataclasses import asdict
from typing import Iterable, List, Optional

import cv2
import numpy as np
import pandas as pd

from .contracts import LightningEvent, VideoInfo
from .utils import clamp, even_int, parse_time, seconds_to_timestamp
from .video_io import read_frame_at


def event_to_row(event: LightningEvent) -> dict:
    row = asdict(event)
    row.update({
        "first_light": seconds_to_timestamp(event.first_light_time),
        "last_light": seconds_to_timestamp(event.last_light_time),
        "cut_start": seconds_to_timestamp(event.start_time),
        "cut_end": seconds_to_timestamp(event.end_time),
        "crop": f"{event.crop_w}x{event.crop_h}+{event.crop_x}+{event.crop_y}",
    })
    return row


def events_to_dataframe(events: Iterable[LightningEvent]) -> pd.DataFrame:
    rows = [event_to_row(e) for e in events]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    columns = [
        "include", "index", "event_type", "score", "hook_score", "confidence", "crop_confidence",
        "first_light", "last_light", "cut_start", "cut_end", "duration", "crop",
        "peak_time", "peak_bright_pct", "peak_delta_mean", "peak_delta_p99", "structure_score", "light_frames",
        "start_time", "end_time", "crop_x", "crop_y", "crop_w", "crop_h", "notes",
    ]
    return df[[c for c in columns if c in df.columns]]


def dataframe_to_events(df: pd.DataFrame, originals: List[LightningEvent], info: VideoInfo) -> List[LightningEvent]:
    if df is None or df.empty:
        return []
    original_by_idx = {int(e.index): e for e in originals}
    out: List[LightningEvent] = []
    for _, row in df.iterrows():
        try:
            original_idx = int(row.get("index", len(out) + 1))
        except Exception:
            original_idx = len(out) + 1
        base = original_by_idx.get(original_idx)
        if base is None:
            base = originals[min(len(out), len(originals) - 1)] if originals else None
        if base is None:
            continue
        payload = asdict(base)
        ev = LightningEvent(**payload)
        ev.include = bool(row.get("include", True))
        ev.start_time = parse_time(row.get("start_time", row.get("cut_start", ev.start_time)), ev.start_time)
        ev.end_time = parse_time(row.get("end_time", row.get("cut_end", ev.end_time)), ev.end_time)
        ev.start_time = clamp(ev.start_time, 0.0, info.duration)
        ev.end_time = clamp(ev.end_time, ev.start_time + 0.05, info.duration)
        ev.duration = round(max(0.05, ev.end_time - ev.start_time), 4)
        for name in ["crop_x", "crop_y", "crop_w", "crop_h"]:
            if name in row and not pd.isna(row[name]):
                try:
                    setattr(ev, name, int(row[name]))
                except Exception:
                    pass
        ev.crop_x = int(clamp(ev.crop_x, 0, max(0, info.width - 2)))
        ev.crop_y = int(clamp(ev.crop_y, 0, max(0, info.height - 2)))
        ev.crop_w = even_int(clamp(ev.crop_w, 2, max(2, info.width - ev.crop_x)))
        ev.crop_h = even_int(clamp(ev.crop_h, 2, max(2, info.height - ev.crop_y)))
        ev.notes = str(row.get("notes", ev.notes) or "")
        out.append(ev)
    for i, ev in enumerate(out, start=1):
        ev.index = i
    return out


def draw_crop_box(image_rgb: np.ndarray, event: LightningEvent, label: Optional[str] = None) -> np.ndarray:
    img = image_rgb.copy()
    x, y, w, h = int(event.crop_x), int(event.crop_y), int(event.crop_w), int(event.crop_h)
    thickness = max(2, img.shape[1] // 320)
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), thickness)
    cv2.rectangle(img, (x + thickness, y + thickness), (x + w - thickness, y + h - thickness), (0, 0, 0), 1)
    text = label or f"Strike {event.index} | score {event.score:.2f} | hook {event.hook_score:.2f}"
    font_scale = max(0.55, img.shape[1] / 1800)
    y_text = max(32, y + 34)
    cv2.putText(img, text, (max(12, x + 8), y_text), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), max(4, thickness + 2), cv2.LINE_AA)
    cv2.putText(img, text, (max(12, x + 8), y_text), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), max(2, thickness), cv2.LINE_AA)
    return img


def crop_frame(image_rgb: np.ndarray, event: LightningEvent) -> np.ndarray:
    x, y, w, h = int(event.crop_x), int(event.crop_y), int(event.crop_w), int(event.crop_h)
    return image_rgb[y:y+h, x:x+w]


def make_contact_sheet(video_path: str, event: LightningEvent, cols: int = 3, thumb_w: int = 420) -> Optional[np.ndarray]:
    times = np.linspace(event.start_time, event.end_time, num=6)
    thumbs = []
    for t in times:
        frame = read_frame_at(video_path, float(t))
        if frame is None:
            continue
        frame = draw_crop_box(frame, event, seconds_to_timestamp(float(t)))
        h, w = frame.shape[:2]
        thumb_h = int(round(h * thumb_w / max(1, w)))
        thumbs.append(cv2.resize(frame, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA))
    if not thumbs:
        return None
    rows = []
    for r in range(0, len(thumbs), cols):
        row = thumbs[r:r+cols]
        max_h = max(im.shape[0] for im in row)
        padded = []
        for im in row:
            if im.shape[0] < max_h:
                pad = np.full((max_h - im.shape[0], im.shape[1], 3), 24, dtype=np.uint8)
                im = np.vstack([im, pad])
            padded.append(im)
        while len(padded) < cols:
            padded.append(np.full((max_h, thumb_w, 3), 24, dtype=np.uint8))
        rows.append(np.hstack(padded))
    return np.vstack(rows)


def make_event_triptych(video_path: str, event: LightningEvent, thumb_h: int = 360) -> Optional[np.ndarray]:
    times = [event.start_time, event.peak_time, event.end_time]
    labels = ["start", "peak", "end"]
    panels = []
    for label, t in zip(labels, times):
        frame = read_frame_at(video_path, t)
        if frame is None:
            continue
        framed = draw_crop_box(frame, event, f"{label} {seconds_to_timestamp(t)}")
        h, w = framed.shape[:2]
        tw = int(round(w * thumb_h / max(1, h)))
        panels.append(cv2.resize(framed, (tw, thumb_h), interpolation=cv2.INTER_AREA))
    if not panels:
        return None
    return np.hstack(panels)
