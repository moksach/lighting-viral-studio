from __future__ import annotations

from dataclasses import asdict
from typing import Iterable, List

from .contracts import LightningEvent


def score_event(
    peak_bright_pct: float,
    peak_delta_mean: float,
    peak_delta_p99: float,
    peak_p99_luma: float,
    structure_score: float,
    light_frames: int,
    min_bright_pct: float,
    delta_mean_threshold: float,
) -> tuple[float, float, float]:
    """Return overall score, hook score, confidence seed.

    Score is a creator ranking metric, not a scientific probability.
    """
    brightness = min(1.0, peak_bright_pct / max(0.05, min_bright_pct * 12.0))
    exposure_jump = min(1.0, max(0.0, peak_delta_mean) / max(8.0, delta_mean_threshold * 4.0))
    highlight_jump = min(1.0, max(0.0, peak_delta_p99) / 80.0)
    highlights = min(1.0, peak_p99_luma / 255.0)
    structure = min(1.0, max(0.0, structure_score))
    duration_bonus = min(1.0, max(1, light_frames) / 6.0)

    score = (
        0.30 * brightness
        + 0.24 * exposure_jump
        + 0.18 * highlight_jump
        + 0.12 * highlights
        + 0.12 * structure
        + 0.04 * duration_bonus
    )
    hook_score = 0.60 * score + 0.40 * max(exposure_jump, structure)
    confidence_seed = 0.20 + 0.80 * score
    return round(float(score), 3), round(float(hook_score), 3), round(float(confidence_seed), 3)


def plan_compilation(events: Iterable[LightningEvent], ordering: str, max_events: int, target_seconds: float) -> List[LightningEvent]:
    selected = [e for e in events if e.include]
    if ordering == "Strongest strike first":
        selected.sort(key=lambda e: (e.hook_score, e.score, e.confidence), reverse=True)
    else:
        selected.sort(key=lambda e: e.start_time)

    out: List[LightningEvent] = []
    total = 0.0
    for e in selected:
        if len(out) >= max(1, int(max_events)):
            break
        if out and target_seconds > 0 and total + e.duration > target_seconds:
            # Include if still very high impact and the reel is too short otherwise.
            if total >= max(7.0, target_seconds * 0.65):
                continue
        out.append(e)
        total += e.duration
    return out


def caption_ideas(events: List[LightningEvent]) -> str:
    selected = [e for e in events if e.include]
    if selected:
        events = selected
    top = max([e.score for e in events], default=0.0)
    return "\n".join([
        "Caught this lightning on my phone during the storm.",
        "Wait for the sky to light up.",
        "The strongest strike is first - the rest kept coming.",
        f"Auto-cut from dark footage. Top strike score: {top:.2f}.",
        "#lightning #storm #thunderstorm #nature #caughtonphone #reels #shorts",
    ]) + "\n"


def manifest_rows(events: List[LightningEvent]) -> List[dict]:
    return [asdict(e) for e in events]
