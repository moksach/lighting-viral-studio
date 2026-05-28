from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Tuple

from .contracts import SCHEMA_VERSION, DetectionConfig, TrimConfig, CropConfig, ExportConfig
from .utils import even_int

ASPECT_LABELS = [
    "9:16 vertical - Reels / Shorts / TikTok",
    "4:5 vertical feed",
    "1:1 square",
    "16:9 landscape",
    "Original aspect",
]

ASPECT_ALIASES = {
    "9:16 vertical — Reels / Shorts / TikTok": ASPECT_LABELS[0],
    "9:16 vertical - Reels / Shorts / TikTok": ASPECT_LABELS[0],
}

DETECTION_MODES = [
    "Balanced",
    "Forked visible bolts",
    "Cloud glow / behind-cloud flash",
]

DETECTION_PROFILES: Dict[str, Dict[str, Any]] = {
    "Find every possible strike (review mode)": dict(
        bright_threshold=165, min_bright_pct=0.01, delta_mean_threshold=3.0,
        delta_p99_threshold=5.0, merge_gap_sec=0.35, sample_fps=30.0,
        scan_every_frame=True, ignore_bottom_pct=0.0, suppress_static_lights=False,
        auto_reject_low_confidence=False, min_export_score=0.05, max_events=0,
    ),
    "Storm with road/street lights (recommended)": dict(
        bright_threshold=210, min_bright_pct=0.20, delta_mean_threshold=10.0,
        delta_p99_threshold=16.0, merge_gap_sec=0.45, sample_fps=12.0,
        scan_every_frame=True,
        ignore_bottom_pct=30.0, suppress_static_lights=True,
        auto_reject_low_confidence=True, min_export_score=0.30, max_events=0,
    ),
    "Normal storm video": dict(
        bright_threshold=205, min_bright_pct=0.25, delta_mean_threshold=12.0,
        delta_p99_threshold=18.0, merge_gap_sec=0.45, sample_fps=12.0,
        scan_every_frame=True,
        ignore_bottom_pct=18.0, suppress_static_lights=True,
        auto_reject_low_confidence=True, min_export_score=0.30, max_events=0,
    ),
    "Very dark sky": dict(
        bright_threshold=185, min_bright_pct=0.12, delta_mean_threshold=8.0,
        delta_p99_threshold=12.0, merge_gap_sec=0.55, sample_fps=12.0,
        scan_every_frame=True,
        ignore_bottom_pct=20.0, suppress_static_lights=True,
        auto_reject_low_confidence=True, min_export_score=0.25, max_events=0,
    ),
    "City lights / false positives": dict(
        bright_threshold=225, min_bright_pct=0.45, delta_mean_threshold=18.0,
        delta_p99_threshold=28.0, merge_gap_sec=0.35, sample_fps=15.0,
        scan_every_frame=True,
        ignore_bottom_pct=35.0, suppress_static_lights=True,
        auto_reject_low_confidence=True, min_export_score=0.35, max_events=0,
    ),
    "Fast many strikes": dict(
        bright_threshold=200, min_bright_pct=0.20, delta_mean_threshold=10.0,
        delta_p99_threshold=15.0, merge_gap_sec=0.25, sample_fps=20.0,
        scan_every_frame=True,
        ignore_bottom_pct=25.0, suppress_static_lights=True,
        auto_reject_low_confidence=True, min_export_score=0.30, max_events=0,
    ),
    "Cloud glow heavy rain": dict(
        bright_threshold=180, min_bright_pct=0.08, delta_mean_threshold=6.0,
        delta_p99_threshold=10.0, merge_gap_sec=0.65, sample_fps=12.0,
        scan_every_frame=True,
        ignore_bottom_pct=20.0, suppress_static_lights=False,
        auto_reject_low_confidence=True, min_export_score=0.22, max_events=0,
    ),
}


def aspect_pair(label: str, video_width: int, video_height: int) -> Tuple[int, int]:
    label = ASPECT_ALIASES.get(label, label)
    mapping = {
        ASPECT_LABELS[0]: (9, 16),
        ASPECT_LABELS[1]: (4, 5),
        ASPECT_LABELS[2]: (1, 1),
        ASPECT_LABELS[3]: (16, 9),
        ASPECT_LABELS[4]: (max(2, video_width), max(2, video_height)),
    }
    return mapping.get(label, (9, 16))


def output_dimensions(label: str, video_width: int, video_height: int) -> Tuple[int, int]:
    label = ASPECT_ALIASES.get(label, label)
    mapping = {
        ASPECT_LABELS[0]: (1080, 1920),
        ASPECT_LABELS[1]: (1080, 1350),
        ASPECT_LABELS[2]: (1080, 1080),
        ASPECT_LABELS[3]: (1920, 1080),
        ASPECT_LABELS[4]: (even_int(video_width), even_int(video_height)),
    }
    return mapping.get(label, (1080, 1920))


def apply_profile(base: DetectionConfig, profile: str) -> DetectionConfig:
    cfg = DetectionConfig(**asdict(base))
    cfg.profile = profile
    for k, v in DETECTION_PROFILES.get(profile, {}).items():
        setattr(cfg, k, v)
    return cfg


def migrate_config(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate older settings into the current schema.

    Future rule: never delete a key silently. Map old keys to new keys here.
    """
    if not isinstance(payload, dict):
        return {"schema_version": SCHEMA_VERSION}
    migrated = dict(payload)
    migrated.setdefault("schema_version", SCHEMA_VERSION)
    # Compatibility with earlier v1/vPro app names.
    if "before_sec" in migrated and "pre_buffer_sec" not in migrated:
        migrated["pre_buffer_sec"] = migrated["before_sec"]
    if "after_sec" in migrated and "post_buffer_sec" not in migrated:
        migrated["post_buffer_sec"] = migrated["after_sec"]
    if "aspect" in migrated and "aspect_label" not in migrated:
        migrated["aspect_label"] = migrated["aspect"]
    if "aspect_label" in migrated:
        migrated["aspect_label"] = ASPECT_ALIASES.get(migrated["aspect_label"], migrated["aspect_label"])
    return migrated


def default_configs(profile: str = "Find every possible strike (review mode)") -> tuple[DetectionConfig, TrimConfig, CropConfig, ExportConfig]:
    det = apply_profile(DetectionConfig(), profile)
    return det, TrimConfig(), CropConfig(), ExportConfig()
