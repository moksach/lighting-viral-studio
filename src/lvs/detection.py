from __future__ import annotations

import math
from dataclasses import asdict
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from .config import aspect_pair
from .contracts import CropConfig, DetectionConfig, FrameSignal, LightningEvent, TrimConfig, VideoInfo
from .crop import fit_crop_to_aspect
from .utils import union_boxes
from .viral import score_event

Progress = Optional[Callable[[float, str], None]]


def _mask_analysis_area(gray: np.ndarray, ignore_bottom_pct: float) -> np.ndarray:
    if ignore_bottom_pct <= 0:
        return gray
    h = gray.shape[0]
    cut = int(round(h * (1.0 - ignore_bottom_pct / 100.0)))
    cut = max(1, min(h, cut))
    return gray[:cut, :]


def _component_bbox(mask: np.ndarray, min_area: int = 8) -> Optional[Tuple[int, int, int, int]]:
    pts = cv2.findNonZero(mask.astype(np.uint8))
    if pts is None:
        return None
    x, y, w, h = cv2.boundingRect(pts)
    if w * h < min_area:
        return None
    return int(x), int(y), int(w), int(h)


def _quantile_from_hist(hist: np.ndarray, q: float) -> float:
    total = float(hist.sum())
    if total <= 0:
        return 0.0
    cutoff = total * q
    cdf = np.cumsum(hist.reshape(-1))
    return float(np.searchsorted(cdf, cutoff, side="left"))


def _fast_luma_stats(gray: np.ndarray, bright_threshold: int) -> tuple[float, float, float, float, float, float, float, float]:
    # Histogram-based percentiles are far faster than np.percentile for video scans.
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).reshape(-1)
    total = max(1.0, float(gray.size))
    bins = np.arange(256, dtype=np.float64)
    mean = float((hist * bins).sum() / total)
    median = _quantile_from_hist(hist, 0.50)
    p95 = _quantile_from_hist(hist, 0.95)
    p99 = _quantile_from_hist(hist, 0.99)
    p986 = _quantile_from_hist(hist, 0.986)
    p9935 = _quantile_from_hist(hist, 0.9935)
    max_luma = float(np.max(np.flatnonzero(hist))) if np.any(hist) else 0.0
    threshold = int(max(0, min(255, bright_threshold)))
    bright_pct = float(hist[threshold:].sum() / total * 100.0)
    return mean, median, p95, p99, p986, p9935, max_luma, bright_pct


def analyze_frame(
    frame_bgr: np.ndarray,
    frame_idx: int,
    fps: float,
    prev: Optional[FrameSignal],
    cfg: DetectionConfig,
    scale_back: float = 1.0,
) -> FrameSignal:
    gray_full = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = _mask_analysis_area(gray_full, cfg.ignore_bottom_pct)

    mean, median, p95, p99, p986, p9935, max_luma, bright_pct = _fast_luma_stats(gray, int(cfg.bright_threshold))
    delta_mean = 0.0 if prev is None else mean - prev.mean_luma
    delta_p99 = 0.0 if prev is None else p99 - prev.p99_luma
    delta_bright = 0.0 if prev is None else bright_pct - prev.bright_pct

    edges = cv2.Canny(gray, 80, 180)
    edge_energy = float(edges.mean() / 255.0)

    reason_parts: List[str] = []
    mode = cfg.detection_mode

    if mode == "Cloud glow / behind-cloud flash":
        is_light = (
            delta_mean >= cfg.delta_mean_threshold
            or delta_p99 >= cfg.delta_p99_threshold
            or (bright_pct >= cfg.min_bright_pct and delta_bright >= 0.02)
            or (p99 >= cfg.bright_threshold and delta_mean >= cfg.delta_mean_threshold * 0.45)
        )
    elif mode == "Forked visible bolts":
        is_light = (
            bright_pct >= cfg.min_bright_pct
            or (p99 >= cfg.bright_threshold and delta_p99 >= cfg.delta_p99_threshold * 0.35)
            or (max_luma >= 250 and delta_bright >= max(0.01, cfg.min_bright_pct * 0.12))
        )
    else:
        is_light = (
            bright_pct >= cfg.min_bright_pct
            or (delta_mean >= cfg.delta_mean_threshold and bright_pct >= max(0.015, cfg.min_bright_pct * 0.18))
            or (delta_p99 >= cfg.delta_p99_threshold and p99 >= cfg.bright_threshold * 0.85)
            or (p99 >= cfg.bright_threshold and delta_mean >= cfg.delta_mean_threshold * 0.45)
        )

    if bright_pct >= cfg.min_bright_pct:
        reason_parts.append("bright_pct")
    if delta_mean >= cfg.delta_mean_threshold:
        reason_parts.append("global_flash")
    if delta_p99 >= cfg.delta_p99_threshold:
        reason_parts.append("highlight_jump")
    if max_luma >= 250:
        reason_parts.append("max_luma")

    # Optional false-positive suppression: static lights tend to have high bright pct but weak frame-to-frame movement.
    if cfg.suppress_static_lights and is_light:
        static_like = (
            bright_pct >= cfg.min_bright_pct
            and delta_mean < cfg.delta_mean_threshold * 0.30
            and delta_p99 < cfg.delta_p99_threshold * 0.30
            and delta_bright < max(0.01, cfg.min_bright_pct * 0.08)
        )
        if static_like:
            is_light = False
            reason_parts.append("static_suppressed")

    if cfg.min_edge_energy > 0 and edge_energy < cfg.min_edge_energy and mode != "Cloud glow / behind-cloud flash":
        is_light = False
        reason_parts.append("low_structure")

    bbox = None
    if is_light:
        # Use an adaptive high percentile mask so the crop focuses on the strike, not all lifted sky.
        if mode == "Cloud glow / behind-cloud flash":
            adaptive = int(max(min(255, cfg.bright_threshold - 12), p986))
        else:
            adaptive = int(max(cfg.bright_threshold, p9935))
        focus = (gray >= adaptive).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        if mode != "Cloud glow / behind-cloud flash":
            focus = cv2.morphologyEx(focus, cv2.MORPH_OPEN, kernel)
        focus = cv2.dilate(focus, kernel, iterations=1)
        raw_bbox = _component_bbox(focus, min_area=12)
        if raw_bbox is not None:
            x, y, w, h = raw_bbox
            bbox = (
                int(round(x * scale_back)), int(round(y * scale_back)),
                max(2, int(round(w * scale_back))), max(2, int(round(h * scale_back))),
            )

    return FrameSignal(
        frame=int(frame_idx), time=float(frame_idx / fps), mean_luma=mean, median_luma=median,
        p95_luma=p95, p99_luma=p99, max_luma=max_luma, bright_pct=bright_pct,
        delta_mean=delta_mean, delta_p99=delta_p99, delta_bright_pct=delta_bright,
        edge_energy=edge_energy, is_light=bool(is_light), bbox=bbox,
        reason="+".join(reason_parts),
    )


def sample_video_signals(video_path: str, info: VideoInfo, cfg: DetectionConfig, progress: Progress = None) -> pd.DataFrame:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video for analysis")

    step = max(1, int(round(info.fps / max(0.5, cfg.sample_fps))))
    resize_w = min(480, int(info.width))
    scale = resize_w / max(1, info.width) if info.width > resize_w else 1.0
    resize_h = max(2, int(round(info.height * scale)))
    scale_back = 1.0 / scale

    rows: List[Dict[str, object]] = []
    prev: Optional[FrameSignal] = None
    total = max(1, math.ceil(max(1, info.frame_count) / step))
    sampled = 0
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step == 0:
            small = cv2.resize(frame, (resize_w, resize_h), interpolation=cv2.INTER_AREA) if scale != 1.0 else frame
            signal = analyze_frame(small, frame_idx, info.fps, prev, cfg, scale_back=scale_back)
            prev = signal
            rows.append(asdict(signal))
            sampled += 1
            if progress and sampled % 8 == 0:
                progress(min(0.58, sampled / total * 0.58), "First-pass scan")
        frame_idx += 1
    cap.release()
    return pd.DataFrame(rows)


def group_candidate_windows(frame_df: pd.DataFrame, merge_gap_sec: float, max_events: int = 80) -> List[Tuple[float, float]]:
    if frame_df.empty or "is_light" not in frame_df.columns:
        return []
    light = frame_df[frame_df["is_light"] == True].copy()  # noqa: E712
    if light.empty:
        return []
    windows: List[Tuple[float, float]] = []
    start = float(light.iloc[0]["time"])
    prev = start
    for _, row in light.iloc[1:].iterrows():
        t = float(row["time"])
        if t - prev <= merge_gap_sec:
            prev = t
        else:
            windows.append((start, prev))
            if len(windows) >= max_events:
                return windows
            start = prev = t
    windows.append((start, prev))
    return windows[:max_events]


def refine_window_full_fps(video_path: str, info: VideoInfo, rough_start: float, rough_end: float, cfg: DetectionConfig) -> pd.DataFrame:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video for full-FPS refinement")

    margin = max(0.20, float(cfg.search_margin_sec))
    start_t = max(0.0, rough_start - margin)
    end_t = min(info.duration, rough_end + margin)
    start_frame = max(0, int(math.floor(start_t * info.fps)))
    end_frame = min(max(0, info.frame_count - 1), int(math.ceil(end_t * info.fps)))

    resize_w = min(640, info.width)
    scale = resize_w / max(1, info.width) if info.width > resize_w else 1.0
    resize_h = max(2, int(round(info.height * scale)))
    scale_back = 1.0 / scale

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    rows: List[Dict[str, object]] = []
    prev: Optional[FrameSignal] = None
    idx = start_frame
    while idx <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (resize_w, resize_h), interpolation=cv2.INTER_AREA) if scale != 1.0 else frame
        signal = analyze_frame(small, idx, info.fps, prev, cfg, scale_back=scale_back)
        prev = signal
        rows.append(asdict(signal))
        idx += 1
    cap.release()
    return pd.DataFrame(rows)


def _event_type(peak_bright: float, crop_box: Optional[Tuple[int, int, int, int]], info: VideoInfo, min_bright_pct: float) -> str:
    event_type = "forked bolt" if peak_bright >= max(0.30, min_bright_pct * 2.5) else "cloud glow"
    if crop_box is not None:
        _, _, bw, bh = crop_box
        if max(bw, bh) > 0.55 * max(info.width, info.height):
            event_type = "wide sky flash"
    return event_type


def _structure_score(light_df: pd.DataFrame) -> float:
    if light_df.empty:
        return 0.0
    edge = float(light_df["edge_energy"].max()) if "edge_energy" in light_df else 0.0
    # Edge energy is usually small; map practical range to 0..1.
    return min(1.0, edge / 0.09)


def build_events(
    video_path: str,
    info: VideoInfo,
    sampled_df: pd.DataFrame,
    det: DetectionConfig,
    trim: TrimConfig,
    crop: CropConfig,
    progress: Progress = None,
) -> Tuple[List[LightningEvent], pd.DataFrame]:
    rough_windows = group_candidate_windows(sampled_df, det.merge_gap_sec, det.max_events)
    if not rough_windows:
        return [], pd.DataFrame()

    aspect_w, aspect_h = aspect_pair(crop.aspect_label, info.width, info.height)
    refined_parts: List[pd.DataFrame] = []
    events: List[LightningEvent] = []

    for i, (rough_start, rough_end) in enumerate(rough_windows, start=1):
        if progress:
            progress(0.58 + 0.32 * (i - 1) / max(1, len(rough_windows)), "Refining lightning windows")
        if det.refine_full_fps:
            refined = refine_window_full_fps(video_path, info, rough_start, rough_end, det)
        else:
            refined = sampled_df[(sampled_df["time"] >= rough_start) & (sampled_df["time"] <= rough_end)].copy()
        if refined.empty:
            continue
        refined_parts.append(refined.assign(rough_event=i))
        light = refined[refined["is_light"] == True].copy()  # noqa: E712
        if light.empty:
            continue

        first_light = float(light["time"].min())
        last_light = float(light["time"].max())
        start = max(0.0, first_light - trim.pre_buffer_sec)
        end = min(info.duration, last_light + trim.post_buffer_sec)

        if end - start < trim.min_event_sec:
            extra = trim.min_event_sec - (end - start)
            start = max(0.0, start - extra / 2.0)
            end = min(info.duration, end + extra / 2.0)
        if trim.max_event_sec > 0 and end - start > trim.max_event_sec:
            center = (first_light + last_light) / 2.0
            start = max(0.0, center - trim.max_event_sec / 2.0)
            end = min(info.duration, start + trim.max_event_sec)
            start = max(0.0, end - trim.max_event_sec)

        crop_box = union_boxes(light["bbox"].tolist())
        crop_x, crop_y, crop_w, crop_h, crop_conf = fit_crop_to_aspect(
            crop_box, info, aspect_w, aspect_h, crop.crop_padding_pct,
            crop.min_crop_coverage, crop.crop_mode, crop.composition_bias,
        )

        # Peak row prioritizes creator impact rather than just raw bright percentage.
        if "delta_p99" in light.columns:
            impact = light["bright_pct"].astype(float) + light["delta_mean"].clip(lower=0).astype(float) * 0.05 + light["delta_p99"].clip(lower=0).astype(float) * 0.03
        else:
            impact = light["bright_pct"].astype(float)
        peak_idx = impact.idxmax()
        peak = light.loc[peak_idx]
        peak_bright = float(light["bright_pct"].max())
        peak_delta = float(light["delta_mean"].max())
        peak_delta_p99 = float(light["delta_p99"].max()) if "delta_p99" in light else 0.0
        peak_p99 = float(light["p99_luma"].max())
        structure = _structure_score(light)
        score, hook_score, conf_seed = score_event(
            peak_bright, peak_delta, peak_delta_p99, peak_p99, structure,
            int(light.shape[0]), det.min_bright_pct, det.delta_mean_threshold,
        )
        confidence = min(1.0, max(0.05, 0.65 * conf_seed + 0.35 * crop_conf))

        events.append(LightningEvent(
            include=True,
            index=len(events) + 1,
            first_light_time=round(first_light, 4),
            last_light_time=round(last_light, 4),
            start_time=round(start, 4),
            end_time=round(end, 4),
            duration=round(max(0.05, end - start), 4),
            crop_x=int(crop_x), crop_y=int(crop_y), crop_w=int(crop_w), crop_h=int(crop_h),
            score=score, confidence=round(float(confidence), 3),
            event_type=_event_type(peak_bright, crop_box, info, det.min_bright_pct),
            peak_time=round(float(peak["time"]), 4),
            peak_bright_pct=round(peak_bright, 4),
            peak_delta_mean=round(peak_delta, 4),
            peak_delta_p99=round(peak_delta_p99, 4),
            peak_p99_luma=round(peak_p99, 2),
            structure_score=round(structure, 3),
            hook_score=hook_score,
            light_frames=int(light.shape[0]),
            crop_confidence=float(crop_conf),
        ))

    refined_df = pd.concat(refined_parts, ignore_index=True) if refined_parts else pd.DataFrame()
    return events, refined_df


def analyze_video(
    video_path: str,
    info: VideoInfo,
    det: DetectionConfig,
    trim: TrimConfig,
    crop: CropConfig,
    progress: Progress = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[LightningEvent]]:
    sampled = sample_video_signals(video_path, info, det, progress=progress)
    events, refined = build_events(video_path, info, sampled, det, trim, crop, progress=progress)
    if progress:
        progress(1.0, "Analysis complete")
    return sampled, refined, events
