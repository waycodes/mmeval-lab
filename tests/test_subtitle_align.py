"""Tests for subtitle alignment."""

import pytest

from mmevallab.benchmarks.subtitle_align import (
    AlignedSubtitle,
    SubtitleSegment,
    align_subtitles_to_frames,
    format_subtitles_for_prompt,
    parse_subtitles,
)


class TestParseSubtitles:
    def test_parse_ms_format(self) -> None:
        subs = [{"start_ms": 0, "end_ms": 1000, "text": "Hello"}]
        result = parse_subtitles(subs)
        assert len(result) == 1
        assert result[0].start_ms == 0
        assert result[0].end_ms == 1000
        assert result[0].text == "Hello"

    def test_parse_seconds_format(self) -> None:
        subs = [{"start": 0, "end": 1, "text": "Hello"}]
        result = parse_subtitles(subs)
        assert result[0].start_ms == 0
        assert result[0].end_ms == 1000


class TestAlignSubtitles:
    def test_subtitle_active_at_frame(self) -> None:
        subs = [SubtitleSegment(start_ms=0, end_ms=2000, text="Hello")]
        frames = [500.0, 1500.0]
        result = align_subtitles_to_frames(subs, frames)

        assert len(result) == 2
        assert result[0].text == "Hello"
        assert result[1].text == "Hello"

    def test_subtitle_in_window(self) -> None:
        subs = [SubtitleSegment(start_ms=0, end_ms=1000, text="Hello")]
        frames = [1200.0]  # 200ms after subtitle ends
        result = align_subtitles_to_frames(subs, frames, window_ms=500)

        assert result[0].text == "Hello"

    def test_subtitle_outside_window(self) -> None:
        subs = [SubtitleSegment(start_ms=0, end_ms=1000, text="Hello")]
        frames = [2000.0]  # 1000ms after subtitle ends
        result = align_subtitles_to_frames(subs, frames, window_ms=500)

        assert result[0].text == ""

    def test_multiple_subtitles(self) -> None:
        subs = [
            SubtitleSegment(start_ms=0, end_ms=1000, text="First"),
            SubtitleSegment(start_ms=2000, end_ms=3000, text="Second"),
        ]
        frames = [500.0, 2500.0]
        result = align_subtitles_to_frames(subs, frames)

        assert result[0].text == "First"
        assert result[1].text == "Second"


class TestFormatSubtitles:
    def test_basic_format(self) -> None:
        aligned = [
            AlignedSubtitle(frame_idx=0, timestamp_ms=0, text="Hello"),
            AlignedSubtitle(frame_idx=1, timestamp_ms=1000, text="World"),
        ]
        result = format_subtitles_for_prompt(aligned)
        assert result == "Hello\nWorld"

    def test_with_timestamps(self) -> None:
        aligned = [
            AlignedSubtitle(frame_idx=0, timestamp_ms=1500, text="Hello"),
        ]
        result = format_subtitles_for_prompt(aligned, include_timestamps=True)
        assert "[1.5s]" in result

    def test_deduplication(self) -> None:
        aligned = [
            AlignedSubtitle(frame_idx=0, timestamp_ms=0, text="Same"),
            AlignedSubtitle(frame_idx=1, timestamp_ms=500, text="Same"),
            AlignedSubtitle(frame_idx=2, timestamp_ms=1000, text="Different"),
        ]
        result = format_subtitles_for_prompt(aligned)
        assert result == "Same\nDifferent"

    def test_empty_subtitles(self) -> None:
        result = format_subtitles_for_prompt([])
        assert result == ""
