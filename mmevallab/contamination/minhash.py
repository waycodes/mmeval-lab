"""MinHash/LSH for text near-duplicate detection."""

from __future__ import annotations

import hashlib
import re
from typing import Iterator


def _shingle(text: str, k: int = 5) -> Iterator[str]:
    """Generate k-shingles from text."""
    text = re.sub(r"\s+", " ", text.lower().strip())
    words = text.split()
    for i in range(len(words) - k + 1):
        yield " ".join(words[i : i + k])


def _hash_shingle(shingle: str, seed: int) -> int:
    """Hash a shingle with a seed."""
    data = f"{seed}:{shingle}".encode()
    return int(hashlib.md5(data).hexdigest(), 16)


class MinHash:
    """MinHash signature for text."""

    def __init__(self, num_hashes: int = 128) -> None:
        self.num_hashes = num_hashes
        self.signature: list[int] = []

    def compute(self, text: str, shingle_size: int = 5) -> "MinHash":
        """Compute MinHash signature for text."""
        shingles = set(_shingle(text, shingle_size))
        if not shingles:
            self.signature = [0] * self.num_hashes
            return self

        self.signature = []
        for seed in range(self.num_hashes):
            min_hash = min(_hash_shingle(s, seed) for s in shingles)
            self.signature.append(min_hash)
        return self

    def jaccard(self, other: "MinHash") -> float:
        """Estimate Jaccard similarity with another MinHash."""
        if len(self.signature) != len(other.signature):
            raise ValueError("Signatures must have same length")
        matches = sum(1 for a, b in zip(self.signature, other.signature) if a == b)
        return matches / len(self.signature)


class LSHIndex:
    """Locality-Sensitive Hashing index for near-duplicate detection."""

    def __init__(self, num_bands: int = 16, rows_per_band: int = 8) -> None:
        self.num_bands = num_bands
        self.rows_per_band = rows_per_band
        self._buckets: list[dict[int, list[str]]] = [{} for _ in range(num_bands)]
        self._signatures: dict[str, MinHash] = {}

    def add(self, doc_id: str, minhash: MinHash) -> None:
        """Add document to index."""
        self._signatures[doc_id] = minhash
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band = tuple(minhash.signature[start:end])
            bucket_key = hash(band)
            if bucket_key not in self._buckets[band_idx]:
                self._buckets[band_idx][bucket_key] = []
            self._buckets[band_idx][bucket_key].append(doc_id)

    def query(self, minhash: MinHash, threshold: float = 0.8) -> list[tuple[str, float]]:
        """Find near-duplicates above threshold."""
        candidates: set[str] = set()
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band = tuple(minhash.signature[start:end])
            bucket_key = hash(band)
            if bucket_key in self._buckets[band_idx]:
                candidates.update(self._buckets[band_idx][bucket_key])

        results = []
        for doc_id in candidates:
            sim = minhash.jaccard(self._signatures[doc_id])
            if sim >= threshold:
                results.append((doc_id, sim))
        return sorted(results, key=lambda x: -x[1])
