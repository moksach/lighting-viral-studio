from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np


def seconds_to_timestamp(seconds: float) -> str:
    seconds = max(0.0, float(seconds or 0.0))
    hh = int(seconds // 3600)
    mm = int((seconds % 3600) // 60)
    ss = seconds - hh * 3600 - mm * 60
    if hh:
        return f"{hh:02d}:{mm:02d}:{ss:06.3f}"
    return f"{mm:02d}:{ss:06.3f}"


def parse_time(value: object, fallback: float = 0.0) -> float:
    if value is None:
        return fallback
    if isinstance(value, (int, float, np.number)):
        return max(0.0, float(value))
    s = str(value).strip()
    if not s:
        return fallback
    try:
        return max(0.0, float(s))
    except ValueError:
        pass
    parts = s.split(":")
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return fallback
    if len(nums) == 2:
        return max(0.0, nums[0] * 60 + nums[1])
    if len(nums) == 3:
        return max(0.0, nums[0] * 3600 + nums[1] * 60 + nums[2])
    return fallback


def safe_name(name: str, default: str = "video.mp4") -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("._-")
    return stem or default


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def even_int(value: float, lo: int = 2) -> int:
    n = int(round(float(value)))
    n = max(lo, n)
    return n - (n % 2)


def run_command(cmd: Sequence[str]) -> Tuple[bool, str]:
    try:
        resolved = list(cmd)
        if resolved:
            resolved[0] = _resolve_command(resolved[0])
        p = subprocess.run(
            resolved, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=False, text=True
        )
        output = (p.stdout or "") + (("\n" if p.stdout and p.stderr else "") + (p.stderr or ""))
        return p.returncode == 0, output.strip()
    except FileNotFoundError as exc:
        return False, str(exc)


def _resolve_command(command: str) -> str:
    if command == "ffmpeg":
        system = shutil.which("ffmpeg")
        if system:
            return system
        try:
            import imageio_ffmpeg

            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return command
    if command == "ffprobe":
        return shutil.which("ffprobe") or command
    return command


def has_ffmpeg() -> bool:
    ok, _ = run_command(["ffmpeg", "-version"])
    return ok


def has_ffprobe() -> bool:
    ok, _ = run_command(["ffprobe", "-version"])
    return ok


def clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def zip_folder(folder: Path, zip_path: Path) -> Path:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in folder.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(folder))
    return zip_path


def union_boxes(boxes: Iterable[Optional[Tuple[int, int, int, int]]]) -> Optional[Tuple[int, int, int, int]]:
    good = []
    for b in boxes:
        if b is None:
            continue
        if isinstance(b, float) and math.isnan(b):
            continue
        if isinstance(b, str):
            continue
        try:
            x, y, w, h = [int(v) for v in b]
            if w > 0 and h > 0:
                good.append((x, y, w, h))
        except Exception:
            continue
    if not good:
        return None
    x1 = min(x for x, y, w, h in good)
    y1 = min(y for x, y, w, h in good)
    x2 = max(x + w for x, y, w, h in good)
    y2 = max(y + h for x, y, w, h in good)
    return int(x1), int(y1), int(x2 - x1), int(y2 - y1)


def list_to_tuple(value: Any) -> Optional[Tuple[int, int, int, int]]:
    if value is None:
        return None
    try:
        if isinstance(value, str):
            return None
        if len(value) != 4:
            return None
        x, y, w, h = [int(v) for v in value]
        return x, y, w, h
    except Exception:
        return None
