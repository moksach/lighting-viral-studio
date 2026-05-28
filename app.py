from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
import streamlit as st

from lvs.config import ASPECT_LABELS, DETECTION_MODES, DETECTION_PROFILES, default_configs
from lvs.contracts import APP_VERSION, CropConfig, DetectionConfig, ExportConfig, LightningEvent, TrimConfig, VideoInfo
from lvs.detection import analyze_video
from lvs.export import export_event_clip, export_package
from lvs.registry import default_registry
from lvs.review import dataframe_to_events, events_to_dataframe, make_contact_sheet, make_event_triptych, draw_crop_box
from lvs.utils import clean_dir, has_ffmpeg, safe_name, seconds_to_timestamp
from lvs.video_io import get_video_info, read_frame_at, write_synthetic_lightning_video
from lvs.viral import plan_compilation


st.set_page_config(page_title="Lightning Viral Studio", page_icon="⚡", layout="wide")

# -----------------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------------

if "working_dir" not in st.session_state:
    st.session_state.working_dir = tempfile.mkdtemp(prefix="lightning_viral_studio_")
if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "edited_events_df" not in st.session_state:
    st.session_state.edited_events_df = None
if "last_video_path" not in st.session_state:
    st.session_state.last_video_path = None
if "profile_applied" not in st.session_state:
    st.session_state.profile_applied = "Normal storm video"

workdir = Path(st.session_state.working_dir)
workdir.mkdir(parents=True, exist_ok=True)


def hydrate_analysis(payload: dict) -> tuple[VideoInfo, pd.DataFrame, pd.DataFrame, List[LightningEvent], DetectionConfig, TrimConfig, CropConfig, ExportConfig]:
    info = VideoInfo(**payload["info"])
    sampled_df = pd.read_json(payload["sampled_df"])
    refined_df = pd.read_json(payload["refined_df"]) if payload.get("refined_df") else pd.DataFrame()
    events = [LightningEvent(**d) for d in payload.get("events", [])]
    det = DetectionConfig(**payload["det"])
    trim = TrimConfig(**payload["trim"])
    crop = CropConfig(**payload["crop"])
    export = ExportConfig(**payload["export"])
    return info, sampled_df, refined_df, events, det, trim, crop, export


def sidebar_configs() -> tuple[DetectionConfig, TrimConfig, CropConfig, ExportConfig, Optional[str]]:
    st.sidebar.header("1. Video")
    uploaded = st.sidebar.file_uploader("Upload thunderstorm video", type=["mp4", "mov", "m4v", "avi", "mkv"])
    local_path = st.sidebar.text_input("Or paste local video path", value="")

    video_path: Optional[str] = None
    if uploaded is not None:
        uploaded_path = workdir / safe_name(uploaded.name)
        if not uploaded_path.exists() or uploaded_path.stat().st_size != uploaded.size:
            with uploaded_path.open("wb") as f:
                f.write(uploaded.getbuffer())
        video_path = str(uploaded_path)
    elif local_path.strip():
        p = Path(local_path.strip()).expanduser()
        if p.exists():
            video_path = str(p)
        else:
            st.sidebar.warning("Local path does not exist on this machine.")

    if st.sidebar.button("Use built-in synthetic test video"):
        synth = write_synthetic_lightning_video(workdir / "synthetic_lightning_test.mp4")
        video_path = str(synth)
        st.session_state.last_video_path = video_path
        st.session_state.analysis = None
        st.session_state.edited_events_df = None

    if video_path:
        st.session_state.last_video_path = video_path
    elif st.session_state.last_video_path:
        video_path = st.session_state.last_video_path

    st.sidebar.header("2. Detection")
    detection_mode = st.sidebar.selectbox("Lightning type", DETECTION_MODES, index=0)
    profile = st.sidebar.selectbox("Starting preset", list(DETECTION_PROFILES.keys()), index=list(DETECTION_PROFILES.keys()).index(st.session_state.profile_applied) if st.session_state.profile_applied in DETECTION_PROFILES else 0)
    defaults, trim_defaults, crop_defaults, export_defaults = default_configs(profile)
    st.session_state.profile_applied = profile

    with st.sidebar.expander("Detection controls", expanded=True):
        sample_fps = st.slider("First-pass analysis FPS", 2.0, 30.0, float(defaults.sample_fps), 1.0)
        refine_full_fps = st.checkbox("Refine first/last light frames at full FPS", value=True)
        bright_threshold = st.slider("Bright pixel threshold", 120, 255, int(defaults.bright_threshold), 1)
        min_bright_pct = st.slider("Minimum bright pixels (%)", 0.01, 10.0, float(defaults.min_bright_pct), 0.01)
        delta_mean_threshold = st.slider("Sudden brightness jump", 1.0, 80.0, float(defaults.delta_mean_threshold), 0.5)
        delta_p99_threshold = st.slider("Highlight jump sensitivity", 1.0, 120.0, float(defaults.delta_p99_threshold), 0.5)
        merge_gap_sec = st.slider("Merge flashes within seconds", 0.05, 2.5, float(defaults.merge_gap_sec), 0.05)
        ignore_bottom_pct = st.slider("Ignore bottom of frame (%)", 0.0, 50.0, float(defaults.ignore_bottom_pct), 1.0, help="Useful if streetlights/traffic appear at the bottom.")
        suppress_static_lights = st.checkbox("Suppress static lights", value=bool(defaults.suppress_static_lights))

    det = DetectionConfig(
        detection_mode=detection_mode, profile=profile, sample_fps=sample_fps,
        refine_full_fps=refine_full_fps, bright_threshold=bright_threshold,
        min_bright_pct=min_bright_pct, delta_mean_threshold=delta_mean_threshold,
        delta_p99_threshold=delta_p99_threshold, merge_gap_sec=merge_gap_sec,
        search_margin_sec=max(0.35, min(1.6, merge_gap_sec + 0.35)),
        ignore_bottom_pct=ignore_bottom_pct, suppress_static_lights=suppress_static_lights,
    )

    st.sidebar.header("3. Trim")
    pre_buffer_sec = st.sidebar.slider("Keep before first light frame", 0.0, 3.0, trim_defaults.pre_buffer_sec, 0.05)
    post_buffer_sec = st.sidebar.slider("Keep after last light frame", 0.0, 5.0, trim_defaults.post_buffer_sec, 0.05)
    min_event_sec = st.sidebar.slider("Minimum clip length", 0.2, 6.0, trim_defaults.min_event_sec, 0.10)
    max_event_sec = st.sidebar.slider("Maximum clip length", 0.0, 15.0, trim_defaults.max_event_sec, 0.50, help="0 means no cap")
    trim = TrimConfig(pre_buffer_sec=pre_buffer_sec, post_buffer_sec=post_buffer_sec, min_event_sec=min_event_sec, max_event_sec=max_event_sec)

    st.sidebar.header("4. Crop")
    aspect_label = st.sidebar.selectbox("Output aspect", ASPECT_LABELS, index=0)
    crop_mode = st.sidebar.selectbox("Crop mode", ["Smart per-strike crop", "Center crop only", "Full frame / no smart crop"], index=0)
    crop_padding_pct = st.sidebar.slider("Padding around detected light", 0.0, 2.5, crop_defaults.crop_padding_pct, 0.05)
    min_crop_coverage = st.sidebar.slider("Minimum crop coverage of frame", 0.25, 1.0, crop_defaults.min_crop_coverage, 0.05)
    composition_bias = st.sidebar.selectbox("Composition bias", ["Center lightning", "Lightning slightly high", "Show more sky above", "Show more ground/context"], index=0)
    crop = CropConfig(aspect_label=aspect_label, crop_mode=crop_mode, crop_padding_pct=crop_padding_pct, min_crop_coverage=min_crop_coverage, composition_bias=composition_bias)

    st.sidebar.header("5. Viral export")
    ordering = st.sidebar.selectbox("Compilation order", ["Strongest strike first", "Timeline order"], index=0)
    keep_audio = st.sidebar.checkbox("Keep original thunder/audio", value=True)
    slow_motion = st.sidebar.checkbox("Slow motion exports", value=False)
    slow_motion_factor = st.sidebar.slider("Slow motion factor", 1.1, 3.0, export_defaults.slow_motion_factor, 0.1, disabled=not slow_motion)
    enhancement = st.sidebar.selectbox("Look enhancement", ["None", "Light contrast boost", "Moody storm boost", "Clean phone footage sharpen", "Flash clarity boost"], index=1)
    add_hook_text = st.sidebar.checkbox("Add hook text overlay", value=False)
    hook_text = st.sidebar.text_input("Hook text", value=export_defaults.hook_text)
    crf = st.sidebar.slider("Quality CRF", 14, 30, export_defaults.crf, 1)
    preset = st.sidebar.selectbox("Encoding preset", ["ultrafast", "veryfast", "faster", "fast", "medium", "slow"], index=4)
    target_compilation_sec = st.sidebar.slider("Target reel length seconds", 5.0, 60.0, export_defaults.target_compilation_sec, 1.0)
    export = ExportConfig(
        ordering=ordering, keep_audio=keep_audio, slow_motion=slow_motion,
        slow_motion_factor=slow_motion_factor, enhancement=enhancement,
        add_hook_text=add_hook_text, hook_text=hook_text, crf=crf, preset=preset,
        target_compilation_sec=target_compilation_sec,
    )
    return det, trim, crop, export, video_path


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------

st.title("⚡ Lightning Viral Studio")
st.caption(f"{APP_VERSION} — one-purpose tool: detect lightning, remove dead dark footage, crop strikes, and export a viral reel.")

with st.expander("Architecture guarantee for future builds", expanded=False):
    st.markdown(
        "This app is structured around versioned contracts: `DetectionConfig`, `TrimConfig`, `CropConfig`, "
        "`ExportConfig`, `LightningEvent`, and `ProjectManifest`. Future modules should register through the "
        "module registry and consume these contracts instead of rewriting the lightning workflow. Existing exports "
        "include `project_manifest.json` so later versions can reproduce or migrate the edit."
    )
    st.json(default_registry().as_dicts())

# -----------------------------------------------------------------------------
# Main flow
# -----------------------------------------------------------------------------

det, trim, crop, export, video_path = sidebar_configs()

if video_path:
    try:
        info = get_video_info(video_path)
        cols = st.columns(5)
        cols[0].metric("Resolution", f"{info.width}×{info.height}")
        cols[1].metric("FPS", f"{info.fps:.2f}")
        cols[2].metric("Duration", seconds_to_timestamp(info.duration))
        cols[3].metric("Frames", f"{info.frame_count:,}")
        cols[4].metric("FFmpeg", "OK" if has_ffmpeg() else "Missing")
        with st.expander("Original video", expanded=False):
            st.video(video_path)
    except Exception as exc:
        st.error(f"Could not read video: {exc}")
        video_path = None
else:
    st.info("Upload a video, paste a local path, or use the synthetic test video.")

analyze_col, clear_col = st.columns([1, 5])
with analyze_col:
    run_analysis = st.button("Analyze video", type="primary", disabled=not bool(video_path))
with clear_col:
    if st.button("Clear analysis"):
        st.session_state.analysis = None
        st.session_state.edited_events_df = None

if run_analysis and video_path:
    bar = st.progress(0.0, text="Preparing analysis…")
    try:
        info = get_video_info(video_path)
        def update_progress(p: float, label: str) -> None:
            bar.progress(min(1.0, max(0.0, p)), text=f"{label}… {int(p*100)}%")
        sampled_df, refined_df, events = analyze_video(video_path, info, det, trim, crop, progress=update_progress)
        payload = {
            "info": asdict(info),
            "sampled_df": sampled_df.to_json(orient="records"),
            "refined_df": refined_df.to_json(orient="records") if refined_df is not None and not refined_df.empty else "[]",
            "events": [asdict(e) for e in events],
            "det": asdict(det),
            "trim": asdict(trim),
            "crop": asdict(crop),
            "export": asdict(export),
        }
        st.session_state.analysis = payload
        st.session_state.edited_events_df = events_to_dataframe(events)
        bar.empty()
    except Exception as exc:
        bar.empty()
        st.error(f"Analysis failed: {exc}")

analysis = st.session_state.analysis
if analysis:
    info, sampled_df, refined_df, events, saved_det, saved_trim, saved_crop, saved_export = hydrate_analysis(analysis)

    st.header("Detected lightning events")
    if not events:
        st.warning("No lightning events found. Try `Very dark sky`, lower bright threshold, lower minimum bright pixels, or lower highlight/global jump thresholds.")
    else:
        selected_initial = [e for e in events if e.include]
        compilation_preview = plan_compilation(selected_initial, export.ordering, export.max_compilation_events, export.target_compilation_sec)
        cols = st.columns(5)
        cols[0].metric("Events", len(events))
        cols[1].metric("Top score", f"{max(e.score for e in events):.2f}")
        cols[2].metric("Best hook", f"{max(e.hook_score for e in events):.2f}")
        cols[3].metric("Selected seconds", f"{sum(e.duration for e in selected_initial):.1f}")
        cols[4].metric("Planned reel", f"{sum(e.duration for e in compilation_preview):.1f}s")

        st.subheader("Manual correction table")
        st.caption("Edit `start_time`, `end_time`, and crop values directly. Uncheck weak detections. The export uses this table, not hidden state.")
        editable_df = st.data_editor(
            st.session_state.edited_events_df,
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            disabled=[
                "index", "event_type", "score", "hook_score", "confidence", "crop_confidence",
                "first_light", "last_light", "cut_start", "cut_end", "duration", "crop",
                "peak_time", "peak_bright_pct", "peak_delta_mean", "peak_delta_p99", "structure_score", "light_frames",
            ],
            column_config={
                "include": st.column_config.CheckboxColumn("include", default=True),
                "start_time": st.column_config.NumberColumn("start_time", min_value=0.0, step=0.05, format="%.3f"),
                "end_time": st.column_config.NumberColumn("end_time", min_value=0.0, step=0.05, format="%.3f"),
                "crop_x": st.column_config.NumberColumn("crop_x", min_value=0, step=2),
                "crop_y": st.column_config.NumberColumn("crop_y", min_value=0, step=2),
                "crop_w": st.column_config.NumberColumn("crop_w", min_value=2, step=2),
                "crop_h": st.column_config.NumberColumn("crop_h", min_value=2, step=2),
            },
            key="events_editor",
        )
        st.session_state.edited_events_df = editable_df
        edited_events = dataframe_to_events(editable_df, events, info)
        selected_events = [e for e in edited_events if e.include]

        tabs = st.tabs(["Review", "Signal", "Viral plan", "Export"])
        with tabs[0]:
            st.markdown("#### Strike previews")
            preview_count = min(12, len(selected_events))
            cols = st.columns(3)
            for j, event in enumerate(selected_events[:preview_count]):
                with cols[j % 3]:
                    st.markdown(f"**Strike {event.index} — score {event.score:.2f}, hook {event.hook_score:.2f}**")
                    triptych = make_event_triptych(info.path, event)
                    if triptych is not None:
                        st.image(triptych, use_container_width=True)
                    else:
                        peak_frame = read_frame_at(info.path, event.peak_time)
                        if peak_frame is not None:
                            st.image(draw_crop_box(peak_frame, event), use_container_width=True)
                    st.caption(f"{seconds_to_timestamp(event.start_time)} → {seconds_to_timestamp(event.end_time)} | {event.event_type}")

            with st.expander("Contact sheet for one strike"):
                ids = [e.index for e in selected_events]
                if ids:
                    pick = st.selectbox("Event", ids, index=0)
                    if st.button("Make contact sheet"):
                        chosen = next(e for e in selected_events if e.index == pick)
                        sheet = make_contact_sheet(info.path, chosen)
                        if sheet is not None:
                            st.image(sheet, use_container_width=True)
                else:
                    st.info("No selected events.")

            with st.expander("Quick preview clip for one strike"):
                ids = [e.index for e in selected_events]
                if ids:
                    pick2 = st.selectbox("Preview event", ids, index=0, key="preview_event_id_final")
                    if st.button("Create preview clip", disabled=not has_ffmpeg()):
                        chosen = next(e for e in selected_events if e.index == pick2)
                        preview_export = ExportConfig(**asdict(export))
                        preview_export.keep_audio = False
                        preview_dir = workdir / "previews"
                        preview_dir.mkdir(exist_ok=True)
                        ok, output, preview_path = export_event_clip(info.path, chosen, info, crop, preview_export, preview_dir, suffix="preview", preview=True)
                        if ok and preview_path:
                            st.video(str(preview_path))
                        else:
                            st.error(output)
                else:
                    st.info("No selected events.")

        with tabs[1]:
            st.markdown("#### Detection signal")
            st.caption("Spikes correspond to lightning, exposure jumps, or strong reflections. Use this to tune thresholds.")
            cols = [c for c in ["time", "mean_luma", "p99_luma", "bright_pct", "delta_mean", "delta_p99"] if c in sampled_df.columns]
            if cols:
                chart_df = sampled_df[cols].copy().set_index("time")
                st.line_chart(chart_df)
            with st.expander("Sampled signal table"):
                st.dataframe(sampled_df.head(3000), use_container_width=True)
            if refined_df is not None and not refined_df.empty:
                with st.expander("Full-FPS refined rows"):
                    st.dataframe(refined_df.head(3000), use_container_width=True)

        with tabs[2]:
            st.markdown("#### Viral reel plan")
            max_compilation_events = st.slider("Max events in compilation", 1, max(1, len(selected_events)), min(len(selected_events), export.max_compilation_events), 1)
            export.max_compilation_events = max_compilation_events
            planned = plan_compilation(selected_events, export.ordering, export.max_compilation_events, export.target_compilation_sec)
            if planned:
                plan_df = pd.DataFrame([
                    {
                        "reel_order": i,
                        "strike": e.index,
                        "score": e.score,
                        "hook_score": e.hook_score,
                        "time": f"{seconds_to_timestamp(e.start_time)}–{seconds_to_timestamp(e.end_time)}",
                        "duration": e.duration,
                        "type": e.event_type,
                    }
                    for i, e in enumerate(planned, start=1)
                ])
                st.dataframe(plan_df, use_container_width=True, hide_index=True)
                total = sum(e.duration for e in planned)
                if total < 7:
                    st.warning("The planned reel is under 7 seconds. Consider adding a little buffer or including more strikes.")
                elif total > 22:
                    st.warning("The planned reel is long for a fast reel. Reduce max events or lower buffer durations.")
                else:
                    st.success("Pacing is in a strong range for a short-form post.")
            else:
                st.info("No selected events for a reel plan.")

        with tabs[3]:
            st.markdown("#### Export package")
            if not has_ffmpeg():
                st.error("FFmpeg is missing. Install FFmpeg and confirm `ffmpeg -version` works.")
            if not selected_events:
                st.warning("No events selected.")
            export.export_individual_clips = st.checkbox("Export individual strike clips", value=True)
            export.export_compilation = st.checkbox("Export final viral reel", value=True)
            if st.button("Export selected clips + reel", type="primary", disabled=(not selected_events or not has_ffmpeg())):
                bar = st.progress(0.0, text="Exporting package…")
                try:
                    zip_path, exported, compilation_path, logs = export_package(
                        info.path, info, selected_events, sampled_df, refined_df,
                        det, trim, crop, export, workdir,
                    )
                    bar.progress(1.0, text="Export complete")
                    st.success(f"Exported {len(exported)} clip file(s).")
                    if compilation_path and compilation_path.exists():
                        st.subheader("Final reel preview")
                        st.video(str(compilation_path))
                    elif exported:
                        st.subheader("First exported clip preview")
                        st.video(str(exported[0]))
                    with zip_path.open("rb") as f:
                        st.download_button("Download export ZIP", f, file_name="lightning_viral_studio_exports.zip", mime="application/zip")
                    if logs:
                        with st.expander("Export logs"):
                            st.text("\n\n".join(logs))
                except Exception as exc:
                    st.error(f"Export failed: {exc}")
                finally:
                    bar.empty()

st.divider()
st.caption("Best default: 0.50s before, 0.90s after, 9:16 smart crop, strongest strike first, light contrast boost. Future modules should be added through `src/lvs/registry.py` and stable contracts, not by rewriting the app.")
