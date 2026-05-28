from __future__ import annotations

from typing import Dict, Iterable, List

import pandas as pd

from .contracts import DetectionConfig, LightningEvent


def signal_summary(sampled_df: pd.DataFrame) -> Dict[str, float]:
    if sampled_df is None or sampled_df.empty:
        return {
            "frames_sampled": 0.0,
            "light_frames": 0.0,
            "max_bright_pct": 0.0,
            "max_delta_mean": 0.0,
            "max_delta_p99": 0.0,
            "max_p99_luma": 0.0,
        }

    def max_col(name: str) -> float:
        if name not in sampled_df:
            return 0.0
        return float(pd.to_numeric(sampled_df[name], errors="coerce").fillna(0.0).max())

    light_frames = 0.0
    if "is_light" in sampled_df:
        light_frames = float(sampled_df["is_light"].fillna(False).astype(bool).sum())

    return {
        "frames_sampled": float(len(sampled_df)),
        "light_frames": light_frames,
        "max_bright_pct": max_col("bright_pct"),
        "max_delta_mean": max_col("delta_mean"),
        "max_delta_p99": max_col("delta_p99"),
        "max_p99_luma": max_col("p99_luma"),
    }


def tuning_recommendations(
    sampled_df: pd.DataFrame,
    events: Iterable[LightningEvent],
    cfg: DetectionConfig,
) -> List[str]:
    summary = signal_summary(sampled_df)
    event_list = list(events or [])
    recs: List[str] = []

    if summary["frames_sampled"] <= 0:
        return ["No analysis samples were produced. Confirm the video opens correctly and try the synthetic test video."]

    if not event_list:
        recs.append("No events were built from the signal. Start with the Very dark sky preset, then re-analyze.")
        if summary["max_p99_luma"] >= cfg.bright_threshold * 0.80:
            recs.append("The highlights came close to the bright threshold. Lower Bright pixel threshold by 10-20 points.")
        if summary["max_bright_pct"] >= cfg.min_bright_pct * 0.35:
            recs.append("Bright pixels were present but not enough to qualify. Lower Minimum bright pixels (%) by 30-50%.")
        if summary["max_delta_mean"] >= cfg.delta_mean_threshold * 0.50:
            recs.append("The whole frame brightened but missed the cutoff. Lower Sudden brightness jump by 25-40%.")
        if summary["max_delta_p99"] >= cfg.delta_p99_threshold * 0.50:
            recs.append("Highlight jumps were close. Lower Highlight jump sensitivity by 25-40%.")
        if cfg.detection_mode != "Cloud glow / behind-cloud flash":
            recs.append("If the sky glows without a visible bolt, switch Lightning type to Cloud glow / behind-cloud flash.")
        return recs[:5]

    selected = [e for e in event_list if e.include]
    weak = [e for e in selected if e.confidence < 0.45 or e.score < 0.20]
    low_crop = [e for e in selected if e.crop_confidence < 0.45]
    long_events = [e for e in selected if e.duration > 7.0]

    if len(event_list) > 24:
        recs.append("Many events were detected. Use City lights / false positives, raise Bright pixel threshold, or raise Sudden brightness jump.")
    if summary["light_frames"] > summary["frames_sampled"] * 0.25:
        recs.append("A large share of sampled frames were flagged. Raise Minimum bright pixels (%) or enable Suppress static lights.")
    if weak:
        recs.append("Some detections are weak. Sort the table by score, uncheck low-confidence rows, then make one preview clip before exporting.")
    if low_crop:
        recs.append("Some crops are uncertain. Increase Minimum crop coverage or switch those strikes to Center crop only before export.")
    if long_events:
        recs.append("Some clips are long for short-form pacing. Lower Maximum clip length or trim end_time in the correction table.")
    if not recs:
        recs.append("Detection looks usable. Review the top strike, create one preview clip, then export the strongest-first reel.")
    return recs[:5]
