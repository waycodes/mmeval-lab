"""Video frame sampling with caching."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image


@dataclass
class FrameManifest:
    """Manifest of sampled frames with timestamps."""

    video_path: str
    video_hash: str
    strategy: str
    params: dict
    frame_count: int
    timestamps_ms: list[float]
    cache_dir: str | None = None


def _compute_video_hash(video_path: Path) -> str:
    """Compute hash of video file for cache key."""
    h = hashlib.sha256()
    with open(video_path, "rb") as f:
        # Read first and last 1MB for speed
        h.update(f.read(1024 * 1024))
        f.seek(-min(1024 * 1024, f.seek(0, 2)), 2)
        h.update(f.read())
    return h.hexdigest()[:16]


def _get_cache_key(video_path: Path, strategy: str, params: dict) -> str:
    """Generate cache key for sampled frames."""
    video_hash = _compute_video_hash(video_path)
    params_str = json.dumps(params, sort_keys=True)
    params_hash = hashlib.sha256(params_str.encode()).hexdigest()[:8]
    return f"{video_hash}_{strategy}_{params_hash}"


class VideoFrameSampler:
    """Cacheable video frame sampler."""

    def __init__(self, cache_dir: Path | str | None = None) -> None:
        """Initialize sampler.

        Args:
            cache_dir: Directory for caching extracted frames (None = no caching)
        """
        self._cache_dir = Path(cache_dir) if cache_dir else None
        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    def sample_uniform(
        self,
        video_path: Path | str,
        num_frames: int = 16,
    ) -> tuple[list["Image.Image"], FrameManifest]:
        """Sample N frames uniformly from video.

        Args:
            video_path: Path to video file
            num_frames: Number of frames to sample

        Returns:
            Tuple of (list of PIL Images, FrameManifest)
        """
        return self._sample(
            video_path,
            strategy="uniform",
            params={"num_frames": num_frames},
        )

    def sample_fps(
        self,
        video_path: Path | str,
        target_fps: float = 1.0,
    ) -> tuple[list["Image.Image"], FrameManifest]:
        """Sample frames at target FPS.

        Args:
            video_path: Path to video file
            target_fps: Target frames per second

        Returns:
            Tuple of (list of PIL Images, FrameManifest)
        """
        return self._sample(
            video_path,
            strategy="fps",
            params={"target_fps": target_fps},
        )

    def _sample(
        self,
        video_path: Path | str,
        strategy: str,
        params: dict,
    ) -> tuple[list["Image.Image"], FrameManifest]:
        """Internal sampling implementation."""
        try:
            import av
            from PIL import Image  # noqa: F401 - used for type conversion
        except ImportError as e:
            raise ImportError("Install av and pillow: pip install av pillow") from e

        video_path = Path(video_path)
        cache_key = _get_cache_key(video_path, strategy, params)

        # Check cache
        if self._cache_dir:
            manifest_path = self._cache_dir / f"{cache_key}_manifest.json"
            if manifest_path.exists():
                return self._load_cached(cache_key)

        # Open video
        container = av.open(str(video_path))
        stream = container.streams.video[0]

        # Get video info
        duration_s = float(stream.duration * stream.time_base)
        fps = float(stream.average_rate)
        total_frames = stream.frames or int(duration_s * fps)

        # Calculate frame indices based on strategy
        if strategy == "uniform":
            num_frames = params["num_frames"]
            if total_frames <= num_frames:
                indices = list(range(total_frames))
            else:
                step = total_frames / num_frames
                indices = [int(i * step) for i in range(num_frames)]
        elif strategy == "fps":
            target_fps = params["target_fps"]
            step = fps / target_fps
            indices = [int(i * step) for i in range(int(total_frames / step))]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Extract frames
        frames = []
        timestamps_ms = []
        frame_idx = 0
        target_set = set(indices)

        for frame in container.decode(video=0):
            if frame_idx in target_set:
                img = frame.to_image()
                frames.append(img)
                timestamps_ms.append(float(frame.pts * stream.time_base) * 1000)
            frame_idx += 1
            if len(frames) >= len(indices):
                break

        container.close()

        # Create manifest
        manifest = FrameManifest(
            video_path=str(video_path),
            video_hash=_compute_video_hash(video_path),
            strategy=strategy,
            params=params,
            frame_count=len(frames),
            timestamps_ms=timestamps_ms,
            cache_dir=str(self._cache_dir) if self._cache_dir else None,
        )

        # Cache results
        if self._cache_dir:
            self._save_cached(cache_key, frames, manifest)

        return frames, manifest

    def _save_cached(
        self,
        cache_key: str,
        frames: list["Image.Image"],
        manifest: FrameManifest,
    ) -> None:
        """Save frames and manifest to cache."""
        if not self._cache_dir:
            return

        # Save manifest
        manifest_path = self._cache_dir / f"{cache_key}_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(
                {
                    "video_path": manifest.video_path,
                    "video_hash": manifest.video_hash,
                    "strategy": manifest.strategy,
                    "params": manifest.params,
                    "frame_count": manifest.frame_count,
                    "timestamps_ms": manifest.timestamps_ms,
                },
                f,
            )

        # Save frames
        for i, frame in enumerate(frames):
            frame_path = self._cache_dir / f"{cache_key}_frame_{i:04d}.png"
            frame.save(frame_path, "PNG")

    def _load_cached(
        self,
        cache_key: str,
    ) -> tuple[list["Image.Image"], FrameManifest]:
        """Load frames and manifest from cache."""
        from PIL import Image

        if not self._cache_dir:
            raise ValueError("No cache directory configured")

        # Load manifest
        manifest_path = self._cache_dir / f"{cache_key}_manifest.json"
        with open(manifest_path) as f:
            data = json.load(f)

        manifest = FrameManifest(
            video_path=data["video_path"],
            video_hash=data["video_hash"],
            strategy=data["strategy"],
            params=data["params"],
            frame_count=data["frame_count"],
            timestamps_ms=data["timestamps_ms"],
            cache_dir=str(self._cache_dir),
        )

        # Load frames
        frames = []
        for i in range(manifest.frame_count):
            frame_path = self._cache_dir / f"{cache_key}_frame_{i:04d}.png"
            frames.append(Image.open(frame_path))

        return frames, manifest
