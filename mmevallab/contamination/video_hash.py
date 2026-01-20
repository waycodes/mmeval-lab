"""Video hashing using frame-based PDQ fallback."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from PIL import Image

from mmevallab.contamination.pdq import compute_pdq_hash


def extract_key_frames(video_path: str, num_frames: int = 8) -> Iterator["Image.Image"]:
    """Extract key frames from video."""
    try:
        import av

        container = av.open(video_path)
        stream = container.streams.video[0]
        total_frames = stream.frames or 1000
        step = max(1, total_frames // num_frames)

        for i, frame in enumerate(container.decode(video=0)):
            if i % step == 0:
                yield frame.to_image()
                if i // step >= num_frames - 1:
                    break
    except ImportError:
        return


def compute_video_hash(video_path: str, num_frames: int = 8) -> str:
    """Compute video hash from key frame PDQ hashes."""
    frame_hashes = []
    for frame in extract_key_frames(video_path, num_frames):
        frame_hashes.append(compute_pdq_hash(frame))

    if not frame_hashes:
        # Fallback to file hash
        with open(video_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    combined = ":".join(frame_hashes)
    return hashlib.sha256(combined.encode()).hexdigest()


def video_similarity(hash1: str, hash2: str) -> float:
    """Compute similarity between video hashes."""
    if hash1 == hash2:
        return 1.0
    # For SHA256 hashes, only exact match is meaningful
    return 0.0
