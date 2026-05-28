from __future__ import annotations

from typing import Iterable, List

from .contracts import LightningEvent
from .utils import seconds_to_timestamp


def selected_lightning(events: Iterable[LightningEvent]) -> List[LightningEvent]:
    return sorted([e for e in events if e.include], key=lambda e: (e.hook_score, e.score), reverse=True)


def social_pack_text(events: Iterable[LightningEvent]) -> str:
    selected = selected_lightning(events)
    if not selected:
        return (
            "No export-ready lightning clips are selected yet.\n"
            "Review the candidates, include only real lightning, then come back here for hooks and captions.\n"
        )

    top = selected[0]
    total = sum(e.duration for e in selected)
    hook_time = seconds_to_timestamp(top.peak_time)
    count = len(selected)

    hooks = [
        f"Wait for the sky at {hook_time}.",
        "The storm went from pitch black to daylight.",
        "Caught the strongest strike first.",
        "This is why I kept recording in the rain.",
        "Phone camera, real storm, no edit trick.",
    ]
    captions = [
        f"Auto-cut the dead dark footage and kept the best {count} lightning moment(s). Strongest strike score: {top.score:.2f}.",
        f"The sky lit up out of nowhere. Best strike hits around {hook_time}.",
        f"Short storm reel from {total:.1f}s of selected lightning clips.",
    ]
    hashtags = [
        "#lightning", "#storm", "#thunderstorm", "#nature", "#caughtonphone",
        "#reels", "#shorts", "#viralvideo", "#weather",
    ]
    posting = [
        "Put the strongest strike first.",
        "Keep the first second quiet and dark only if the flash lands immediately after.",
        "Use original thunder audio when it is clean; otherwise export without audio and add a trending sound.",
        "Avoid exporting rejected candidates; headlights and reflections hurt retention.",
    ]

    sections = [
        ("Hooks", hooks),
        ("Caption Options", captions),
        ("Hashtags", [" ".join(hashtags)]),
        ("Posting Notes", posting),
    ]
    lines: List[str] = []
    for title, values in sections:
        lines.append(title)
        lines.extend(f"- {value}" for value in values)
        lines.append("")
    return "\n".join(lines).strip() + "\n"
