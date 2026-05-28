from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .contracts import VideoInfo
from .utils import run_command


def probe_rotation(path: str) -> int:
    ok, out = run_command([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream_tags=rotate:stream_side_data=rotation",
        "-of", "json", path,
    ])
    if not ok:
        return 0
    try:
        payload = json.loads(out or "{}")
        streams = payload.get("streams") or []
        if not streams:
            return 0
        stream = streams[0]
        tags = stream.get("tags") or {}
        if "rotate" in tags:
            return int(float(tags["rotate"])) % 360
        for sd in stream.get("side_data_list") or []:
            if "rotation" in sd:
                return int(float(sd["rotation"])) % 360
    except Exception:
        return 0
    return 0


def get_video_info(path: str) -> VideoInfo:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if fps <= 0:
        fps = 30.0
    duration = frames / fps if frames > 0 else 0.0
    return VideoInfo(
        path=str(path), width=width, height=height, fps=fps,
        frame_count=frames, duration=duration, rotation=probe_rotation(path),
    )


def read_frame_at(video_path: str, time_sec: float) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(time_sec)) * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def write_synthetic_lightning_video(path: Path, width: int = 720, height: int = 1280, fps: int = 30, seconds: int = 8) -> Path:
    """Create a deterministic synthetic video used for the self-test."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Could not create synthetic video")

    strikes = {
        45: [(360, 80), (340, 260), (380, 430), (345, 640)],
        112: [(500, 40), (480, 210), (540, 410), (505, 720)],
        185: [(220, 160), (260, 360), (235, 520), (300, 780)],
    }
    total_frames = fps * seconds
    rng = np.random.default_rng(7)
    for i in range(total_frames):
        base = np.full((height, width, 3), 12, dtype=np.uint8)
        noise = rng.normal(0, 2.2, base.shape).astype(np.int16)
        frame = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        # dim horizon / city lights
        cv2.rectangle(frame, (0, int(height * 0.86)), (width, height), (18, 18, 20), -1)
        for x in range(40, width, 130):
            cv2.circle(frame, (x, int(height * 0.90)), 3, (120, 110, 70), -1)
        for start, pts in strikes.items():
            if start <= i <= start + 3:
                flash = 1.0 - 0.16 * (i - start)
                frame = np.clip(frame.astype(np.float32) + 45 * flash, 0, 255).astype(np.uint8)
                for a, b in zip(pts[:-1], pts[1:]):
                    cv2.line(frame, a, b, (255, 255, 245), 5, cv2.LINE_AA)
                    cv2.line(frame, a, b, (190, 210, 255), 12, cv2.LINE_AA)
                branch_start = pts[1]
                cv2.line(frame, branch_start, (branch_start[0] - 70, branch_start[1] + 130), (230, 235, 255), 3, cv2.LINE_AA)
                cv2.line(frame, pts[2], (pts[2][0] + 90, pts[2][1] + 100), (225, 230, 255), 3, cv2.LINE_AA)
        writer.write(frame)
    writer.release()
    return path
