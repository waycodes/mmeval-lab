# Video-MME Benchmark Contract

## Overview

Video-MME evaluates video understanding across diverse domains with multiple-choice questions requiring temporal reasoning.

## Dataset

- **Source**: Official Video-MME release
- **Format**: Videos (MP4) + JSON annotations
- **License**: Research use only; videos must not be redistributed

## Input Schema

| Field | Type | Description |
|-------|------|-------------|
| `video_id` | str | Video identifier |
| `video_path` | str | Path to video file |
| `question` | str | Question text |
| `options` | list[str] | Answer choices (A-D) |
| `subtitles` | list[dict] | Subtitle segments with timestamps |
| `answer` | str | Correct option letter |

## Metadata Fields

| Field | Values |
|-------|--------|
| `duration` | short (<2min), medium (2-15min), long (>15min) |
| `domain` | Knowledge, Film, Sports, Artistic, Life, etc. |
| `subcategory` | Fine-grained category |
| `task_type` | perception, reasoning, temporal |

## Frame Sampling

| Strategy | Description |
|----------|-------------|
| `uniform_n` | N frames uniformly sampled |
| `fps_based` | Sample at fixed FPS (e.g., 1 fps) |
| `keyframe` | Extract keyframes only |

Cache key: `(video_hash, strategy, params)`

## Subtitle Alignment

Subtitles are aligned to sampled frame timestamps:
- Each frame gets subtitles within ±window_ms
- Default window: 500ms before frame timestamp

## Output Schema

```json
{
  "example_id": "video_001_q1",
  "raw_output": "Based on the video, the answer is C...",
  "extracted_answer": "C",
  "is_correct": true,
  "num_frames": 16
}
```

## Scoring

- **Primary metric**: Accuracy (exact match)
- **Breakdowns**: By duration, domain, subcategory, task_type
- **With/without subtitles**: Separate evaluation tracks

## Prompt Template (v1)

```
Video frames are shown above.

{subtitles_if_enabled}

Question: {question}

Options:
{options}

Answer with the letter of the correct option.
```

## License Constraints

- Videos must not be redistributed
- Requires explicit license acceptance flag in config
- Frames may be cached locally but not shared
- Smoke subset uses only locally-available test videos
