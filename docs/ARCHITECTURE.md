# Architecture Notes

## Core principle

The app has one active product module: `lightning_reel`. The UI is allowed to change freely, but the core contracts should remain stable so future features do not break prior behavior.

## Stable contracts

Defined in `src/lvs/contracts.py`:

- `VideoInfo`
- `FrameSignal`
- `DetectionConfig`
- `TrimConfig`
- `CropConfig`
- `ExportConfig`
- `LightningEvent`
- `ProjectManifest`
- `ModuleManifest`

Breaking changes require a migration function in `src/lvs/config.py`.

## Current data flow

```text
Video -> VideoInfo
      -> sample_video_signals()
      -> group_candidate_windows()
      -> refine_window_full_fps()
      -> build_events()
      -> editable table
      -> plan_compilation()
      -> export_package()
      -> project_manifest.json
```

## Extension points

Future modules should:

1. Register a `ModuleManifest` in `registry.py`.
2. Consume `VideoInfo` and produce their own versioned event/config objects.
3. Avoid modifying lightning detection internals unless the feature directly improves lightning reels.
4. Export a manifest section so edits remain reproducible.

## Compatibility promise

- Existing exported `project_manifest.json` should remain readable.
- Existing event CSV columns should remain present.
- New fields should be additive.
- Existing Streamlit controls can move, but their effects should not silently change.
