import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lvs.contracts import DetectionConfig, TrimConfig, CropConfig, ExportConfig, LightningEvent, SCHEMA_VERSION
from lvs.config import migrate_config
from lvs.crop import fit_crop_to_aspect
from lvs.contracts import VideoInfo
from lvs.diagnostics import signal_summary, tuning_recommendations


def test_schema_version_present():
    assert DetectionConfig().schema_version == SCHEMA_VERSION
    assert TrimConfig().schema_version == SCHEMA_VERSION
    assert CropConfig().schema_version == SCHEMA_VERSION
    assert ExportConfig().schema_version == SCHEMA_VERSION


def test_migration_aliases():
    migrated = migrate_config({"before_sec": 0.4, "after_sec": 1.0, "aspect": "9:16 vertical — Reels / Shorts / TikTok"})
    assert migrated["pre_buffer_sec"] == 0.4
    assert migrated["post_buffer_sec"] == 1.0
    assert migrated["aspect_label"].startswith("9:16")


def test_crop_even_and_in_bounds():
    info = VideoInfo(path="x", width=1920, height=1080, fps=30, frame_count=300, duration=10)
    x, y, w, h, conf = fit_crop_to_aspect((100, 100, 300, 400), info, 9, 16, 0.7, 0.55, "Smart per-strike crop")
    assert x >= 0 and y >= 0
    assert w % 2 == 0 and h % 2 == 0
    assert x + w <= info.width
    assert y + h <= info.height
    assert 0 <= conf <= 1


def test_diagnostics_no_events_recommends_tuning():
    import pandas as pd

    df = pd.DataFrame([
        {"is_light": False, "bright_pct": 0.04, "delta_mean": 3.0, "delta_p99": 5.0, "p99_luma": 170.0},
        {"is_light": False, "bright_pct": 0.10, "delta_mean": 8.0, "delta_p99": 9.0, "p99_luma": 180.0},
    ])
    summary = signal_summary(df)
    recs = tuning_recommendations(df, [], DetectionConfig())
    assert summary["frames_sampled"] == 2
    assert recs
    assert "No events" in recs[0]
