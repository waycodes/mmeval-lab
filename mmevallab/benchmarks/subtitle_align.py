"""Subtitle alignment to sampled video frames."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SubtitleSegment:
    """A subtitle segment with timing."""

    start_ms: float
    end_ms: float
    text: str


@dataclass
class AlignedSubtitle:
    """Subtitle aligned to a specific frame."""

    frame_idx: int
    timestamp_ms: float
    text: str


def parse_subtitles(subtitles: list[dict]) -> list[SubtitleSegment]:
    """Parse subtitle data into SubtitleSegment objects.

    Args:
        subtitles: List of subtitle dicts with start, end, text fields

    Returns:
        List of SubtitleSegment objects
    """
    segments = []
    for sub in subtitles:
        # Handle various timestamp formats
        start = sub.get("start_ms") or sub.get("start", 0) * 1000
        end = sub.get("end_ms") or sub.get("end", 0) * 1000
        text = sub.get("text", "")
        segments.append(SubtitleSegment(start_ms=float(start), end_ms=float(end), text=text))
    return segments


def align_subtitles_to_frames(
    subtitles: list[SubtitleSegment],
    frame_timestamps_ms: list[float],
    window_ms: float = 500.0,
) -> list[AlignedSubtitle]:
    """Align subtitles to sampled frame timestamps.

    For each frame, finds subtitles that are active within a time window
    around the frame timestamp.

    Args:
        subtitles: List of subtitle segments
        frame_timestamps_ms: List of frame timestamps in milliseconds
        window_ms: Time window before frame to include subtitles (default 500ms)

    Returns:
        List of AlignedSubtitle objects, one per frame with relevant text
    """
    aligned = []

    for frame_idx, frame_ts in enumerate(frame_timestamps_ms):
        # Find subtitles active around this frame
        # Include subtitles that:
        # 1. Are currently showing (start <= frame_ts <= end)
        # 2. Recently ended (within window_ms before frame)
        relevant_texts = []

        for sub in subtitles:
            # Subtitle is active at frame time
            if sub.start_ms <= frame_ts <= sub.end_ms:
                relevant_texts.append(sub.text)
            # Subtitle recently ended (within window)
            elif sub.end_ms <= frame_ts <= sub.end_ms + window_ms:
                relevant_texts.append(sub.text)

        # Combine relevant subtitles
        combined_text = " ".join(relevant_texts).strip()

        aligned.append(
            AlignedSubtitle(
                frame_idx=frame_idx,
                timestamp_ms=frame_ts,
                text=combined_text,
            )
        )

    return aligned


def format_subtitles_for_prompt(
    aligned_subtitles: list[AlignedSubtitle],
    include_timestamps: bool = False,
) -> str:
    """Format aligned subtitles for inclusion in a prompt.

    Args:
        aligned_subtitles: List of aligned subtitles
        include_timestamps: Whether to include timestamps in output

    Returns:
        Formatted subtitle string
    """
    if not aligned_subtitles:
        return ""

    # Deduplicate consecutive identical subtitles
    unique_texts = []
    prev_text = None
    for sub in aligned_subtitles:
        if sub.text and sub.text != prev_text:
            if include_timestamps:
                ts_sec = sub.timestamp_ms / 1000
                unique_texts.append(f"[{ts_sec:.1f}s] {sub.text}")
            else:
                unique_texts.append(sub.text)
            prev_text = sub.text

    return "\n".join(unique_texts)
