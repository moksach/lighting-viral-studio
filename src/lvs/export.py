from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd

from .config import output_dimensions
from .contracts import APP_VERSION, SCHEMA_VERSION, CropConfig, DetectionConfig, ExportConfig, LightningEvent, ProjectManifest, TrimConfig, VideoInfo
from .registry import default_registry
from .utils import clean_dir, run_command, seconds_to_timestamp, write_json, zip_folder
from .viral import caption_ideas, manifest_rows, plan_compilation
from .social import social_pack_text


def _escape_drawtext(text: str) -> str:
    # FFmpeg drawtext escapes: backslash, colon, apostrophe, percent.
    s = str(text or "")
    s = s.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'").replace("%", "\\%")
    return s


def ffmpeg_filter_for_event(event: LightningEvent, info: VideoInfo, crop: CropConfig, export: ExportConfig) -> str:
    out_w, out_h = output_dimensions(crop.aspect_label, info.width, info.height)
    parts = [
        f"crop={event.crop_w}:{event.crop_h}:{event.crop_x}:{event.crop_y}",
        f"scale={out_w}:{out_h}:force_original_aspect_ratio=decrease",
        f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2",
        "setsar=1",
    ]
    if export.enhancement == "Light contrast boost":
        parts.append("eq=contrast=1.13:brightness=0.01:saturation=1.05")
    elif export.enhancement == "Moody storm boost":
        parts.append("eq=contrast=1.22:brightness=-0.02:saturation=0.95")
    elif export.enhancement == "Clean phone footage sharpen":
        parts.append("eq=contrast=1.10:saturation=1.04,unsharp=5:5:0.55:3:3:0.2")
    elif export.enhancement == "Flash clarity boost":
        parts.append("eq=contrast=1.18:brightness=0.005:saturation=1.02,unsharp=7:7:0.42:3:3:0.12")

    if export.add_hook_text and export.hook_text.strip():
        escaped = _escape_drawtext(export.hook_text.strip())
        # No fontfile: lets FFmpeg use its default fontconfig if available.
        parts.append(
            "drawtext="
            f"text='{escaped}':"
            "x=(w-text_w)/2:y=h*0.075:"
            "fontsize=max(34\\,h/28):fontcolor=white:"
            "borderw=4:bordercolor=black@0.75"
        )
    return ",".join(parts)


def export_event_clip(
    video_path: str,
    event: LightningEvent,
    info: VideoInfo,
    crop: CropConfig,
    export: ExportConfig,
    out_dir: Path,
    suffix: str = "",
    preview: bool = False,
) -> Tuple[bool, str, Optional[Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    start_ts = seconds_to_timestamp(event.start_time).replace(":", "-")
    end_ts = seconds_to_timestamp(event.end_time).replace(":", "-")
    name_suffix = f"_{suffix}" if suffix else ""
    out_file = out_dir / f"strike_{event.index:02d}_score_{event.score:.2f}_{start_ts}_to_{end_ts}{name_suffix}.mp4"
    vf = ffmpeg_filter_for_event(event, info, crop, export)
    duration = max(0.05, event.end_time - event.start_time)

    cmd = [
        "ffmpeg", "-hide_banner", "-y",
        "-ss", f"{event.start_time:.3f}",
        "-t", f"{duration:.3f}",
        "-i", video_path,
    ]
    if export.slow_motion and not preview:
        vf = vf + f",setpts={float(export.slow_motion_factor):.3f}*PTS"
        audio_args = ["-an"]
    elif export.keep_audio and not preview:
        audio_args = ["-c:a", "aac", "-b:a", "160k", "-af", "aresample=async=1:first_pts=0"]
    else:
        audio_args = ["-an"]

    cmd += [
        "-vf", vf,
        "-c:v", "libx264",
        "-profile:v", "high",
        "-pix_fmt", "yuv420p",
        "-crf", str(26 if preview else export.crf),
        "-preset", "veryfast" if preview else export.preset,
        "-movflags", "+faststart",
    ] + audio_args + [str(out_file)]
    ok, output = run_command(cmd)
    return ok, output, out_file if ok else None


def concatenate_clips(clip_paths: Sequence[Path], out_file: Path, reencode: bool = False) -> Tuple[bool, str]:
    if not clip_paths:
        return False, "No clips to concatenate"
    concat_file = out_file.parent / "concat_list.txt"
    with concat_file.open("w", encoding="utf-8") as f:
        for p in clip_paths:
            escaped = str(p).replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")
    cmd = ["ffmpeg", "-hide_banner", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file)]
    if reencode:
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium", "-c:a", "aac", "-b:a", "160k"]
    else:
        cmd += ["-c", "copy"]
    cmd += ["-movflags", "+faststart", str(out_file)]
    ok, out = run_command(cmd)
    if ok:
        return ok, out
    # Fallback: re-encode if stream-copy concat fails due audio/video metadata differences.
    if not reencode:
        return concatenate_clips(clip_paths, out_file, reencode=True)
    return ok, out


def build_project_manifest(
    info: VideoInfo,
    det: DetectionConfig,
    trim: TrimConfig,
    crop: CropConfig,
    export: ExportConfig,
    selected_events: List[LightningEvent],
    compilation_events: List[LightningEvent],
) -> ProjectManifest:
    return ProjectManifest(
        schema_version=SCHEMA_VERSION,
        app_version=APP_VERSION,
        created_by="Lightning Viral Studio",
        source_video=asdict(info),
        detection=asdict(det),
        trim=asdict(trim),
        crop=asdict(crop),
        export=asdict(export),
        modules=default_registry().as_dicts(),
        selected_events=manifest_rows(selected_events),
        compilation_plan=manifest_rows(compilation_events),
    )


def export_package(
    video_path: str,
    info: VideoInfo,
    events: List[LightningEvent],
    sampled_df: pd.DataFrame,
    refined_df: pd.DataFrame,
    det: DetectionConfig,
    trim: TrimConfig,
    crop: CropConfig,
    export: ExportConfig,
    workdir: Path,
) -> Tuple[Path, List[Path], Optional[Path], List[str]]:
    export_dir = workdir / "exports"
    clean_dir(export_dir)
    clips_dir = export_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    logs: List[str] = []
    exported: List[Path] = []

    selected = [e for e in events if e.include]
    selected.sort(key=lambda e: e.index)

    if export.export_individual_clips:
        for ev in selected:
            ok, output, path = export_event_clip(video_path, ev, info, crop, export, clips_dir)
            if ok and path:
                exported.append(path)
            else:
                logs.append(f"Strike {ev.index} failed:\n{output}")
    else:
        # Compilation still needs temporary clip files.
        temp_dir = export_dir / "_temp_compilation_clips"
        temp_dir.mkdir(exist_ok=True)
        for ev in selected:
            ok, output, path = export_event_clip(video_path, ev, info, crop, export, temp_dir, suffix="tmp")
            if ok and path:
                exported.append(path)
            else:
                logs.append(f"Strike {ev.index} temp export failed:\n{output}")

    compilation_path: Optional[Path] = None
    compilation_events = plan_compilation(selected, export.ordering, export.max_compilation_events, export.target_compilation_sec)
    if export.export_compilation and exported and compilation_events:
        ordered_paths: List[Path] = []
        for ev in compilation_events:
            matches = [p for p in exported if p.name.startswith(f"strike_{ev.index:02d}_")]
            ordered_paths.extend(matches[:1])
        compilation_path = export_dir / "lightning_viral_reel.mp4"
        ok, output = concatenate_clips(ordered_paths, compilation_path)
        if not ok:
            logs.append("Compilation failed:\n" + output)
            compilation_path = None

    # Reports and manifest.
    if selected:
        pd.DataFrame([asdict(e) for e in selected]).to_csv(export_dir / "events_selected.csv", index=False)
    sampled_df.to_csv(export_dir / "analysis_signal_sampled.csv", index=False)
    if refined_df is not None and not refined_df.empty:
        refined_df.to_csv(export_dir / "analysis_signal_refined.csv", index=False)
    manifest = build_project_manifest(info, det, trim, crop, export, selected, compilation_events)
    write_json(export_dir / "project_manifest.json", asdict(manifest))
    (export_dir / "caption_ideas.txt").write_text(caption_ideas(selected), encoding="utf-8")
    (export_dir / "social_media_pack.txt").write_text(social_pack_text(selected), encoding="utf-8")
    (export_dir / "README_EXPORT.txt").write_text(
        "Lightning Viral Studio export package\n\n"
        "Files:\n"
        "- clips/: individual selected strikes\n"
        "- lightning_viral_reel.mp4: strongest-first or timeline compilation, if enabled\n"
        "- events_selected.csv: exact time/crop metadata\n"
        "- analysis_signal_*.csv: detection diagnostics\n"
        "- project_manifest.json: versioned reproducibility data\n"
        "- caption_ideas.txt: quick social caption starters\n"
        "- social_media_pack.txt: hooks, captions, hashtags, and posting notes\n",
        encoding="utf-8",
    )
    zip_path = zip_folder(export_dir, workdir / "lightning_viral_studio_exports.zip")
    return zip_path, exported, compilation_path, logs
