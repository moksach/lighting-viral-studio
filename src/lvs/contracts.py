"""Stable contracts for Lightning Viral Studio.

Future modules should depend on these dataclasses instead of reaching into UI or
implementation internals. Add fields only with defaults. Do not rename existing
fields without adding a migration in config.py.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

SCHEMA_VERSION = "1.0.0"
APP_VERSION = "Lightning Viral Studio 1.0.0"


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float
    rotation: int = 0


@dataclass
class FrameSignal:
    frame: int
    time: float
    mean_luma: float
    median_luma: float
    p95_luma: float
    p99_luma: float
    max_luma: float
    bright_pct: float
    delta_mean: float
    delta_p99: float
    delta_bright_pct: float
    edge_energy: float
    is_light: bool
    bbox: Optional[Tuple[int, int, int, int]] = None
    reason: str = ""


@dataclass
class DetectionConfig:
    schema_version: str = SCHEMA_VERSION
    detection_mode: str = "Balanced"
    profile: str = "Normal storm video"
    sample_fps: float = 12.0
    scan_every_frame: bool = True
    refine_full_fps: bool = True
    bright_threshold: int = 205
    min_bright_pct: float = 0.25
    delta_mean_threshold: float = 12.0
    delta_p99_threshold: float = 18.0
    min_edge_energy: float = 0.0
    merge_gap_sec: float = 0.45
    search_margin_sec: float = 0.75
    ignore_bottom_pct: float = 0.0
    suppress_static_lights: bool = True
    auto_reject_low_confidence: bool = True
    min_export_score: float = 0.30
    max_events: int = 0


@dataclass
class TrimConfig:
    schema_version: str = SCHEMA_VERSION
    pre_buffer_sec: float = 0.50
    post_buffer_sec: float = 0.90
    min_event_sec: float = 1.20
    max_event_sec: float = 7.00


@dataclass
class CropConfig:
    schema_version: str = SCHEMA_VERSION
    aspect_label: str = "9:16 vertical - Reels / Shorts / TikTok"
    crop_mode: str = "Smart per-strike crop"
    crop_padding_pct: float = 0.70
    min_crop_coverage: float = 0.55
    composition_bias: str = "Center lightning"
    crop_lock: str = "Per event stable crop"


@dataclass
class ExportConfig:
    schema_version: str = SCHEMA_VERSION
    ordering: str = "Strongest strike first"
    keep_audio: bool = True
    slow_motion: bool = False
    slow_motion_factor: float = 1.50
    enhancement: str = "Light contrast boost"
    add_hook_text: bool = False
    hook_text: str = "Caught this on my phone"
    add_countdown_flash: bool = False
    crf: int = 18
    preset: str = "medium"
    max_compilation_events: int = 8
    target_compilation_sec: float = 18.0
    export_individual_clips: bool = True
    export_compilation: bool = True


@dataclass
class LightningEvent:
    include: bool
    index: int
    first_light_time: float
    last_light_time: float
    start_time: float
    end_time: float
    duration: float
    crop_x: int
    crop_y: int
    crop_w: int
    crop_h: int
    score: float
    confidence: float
    event_type: str
    peak_time: float
    peak_bright_pct: float
    peak_delta_mean: float
    peak_delta_p99: float
    peak_p99_luma: float
    structure_score: float
    hook_score: float
    light_frames: int
    crop_confidence: float
    notes: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModuleManifest:
    """Contract for future modules.

    The current app ships with only the lightning module active. Future modules
    can register here without changing the core project state or export package.
    """
    module_id: str
    name: str
    version: str
    description: str
    schema_version: str = SCHEMA_VERSION
    enabled: bool = True
    dependencies: List[str] = field(default_factory=list)


@dataclass
class ProjectManifest:
    schema_version: str
    app_version: str
    created_by: str
    source_video: Dict[str, Any]
    detection: Dict[str, Any]
    trim: Dict[str, Any]
    crop: Dict[str, Any]
    export: Dict[str, Any]
    modules: List[Dict[str, Any]]
    selected_events: List[Dict[str, Any]]
    compilation_plan: List[Dict[str, Any]]
