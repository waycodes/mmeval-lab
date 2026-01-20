"""PDQ perceptual hashing for images/PDFs."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image


def compute_pdq_hash(image: "Image.Image", hash_size: int = 16) -> str:
    """Compute PDQ-like perceptual hash for image.

    Simplified implementation using DCT-like approach.
    """
    # Resize to small square
    img = image.convert("L").resize((hash_size, hash_size))
    pixels = list(img.getdata())

    # Compute mean
    mean = sum(pixels) / len(pixels)

    # Generate hash bits
    bits = "".join("1" if p > mean else "0" for p in pixels)

    # Convert to hex
    hash_int = int(bits, 2)
    return f"{hash_int:0{hash_size * hash_size // 4}x}"


def hamming_distance(hash1: str, hash2: str) -> int:
    """Compute Hamming distance between two hashes."""
    if len(hash1) != len(hash2):
        raise ValueError("Hashes must have same length")
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))


def pdq_similarity(hash1: str, hash2: str) -> float:
    """Compute similarity (0-1) between two PDQ hashes."""
    dist = hamming_distance(hash1, hash2)
    max_dist = len(hash1)
    return 1.0 - (dist / max_dist)


def hash_pdf_page(pdf_path: str, page_num: int = 0) -> str:
    """Compute perceptual hash for a PDF page."""
    try:
        import fitz  # pymupdf

        doc = fitz.open(pdf_path)
        page = doc[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(0.5, 0.5))

        from PIL import Image

        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return compute_pdq_hash(img)
    except ImportError:
        # Fallback to file hash
        with open(pdf_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:64]
