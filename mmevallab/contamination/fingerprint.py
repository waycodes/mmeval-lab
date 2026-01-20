"""Text fingerprinting for contamination detection."""

from __future__ import annotations

import hashlib
import re
import unicodedata


def normalize_text(text: str) -> str:
    """Normalize text for fingerprinting."""
    # Lowercase
    text = text.lower()
    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove punctuation (keep alphanumeric and spaces)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def compute_text_hash(text: str, normalize: bool = True) -> str:
    """Compute SHA256 hash of text."""
    if normalize:
        text = normalize_text(text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_ngram_hashes(text: str, n: int = 5, normalize: bool = True) -> set[str]:
    """Compute hashes of word n-grams for partial matching."""
    if normalize:
        text = normalize_text(text)
    words = text.split()
    if len(words) < n:
        return {compute_text_hash(text, normalize=False)}
    hashes = set()
    for i in range(len(words) - n + 1):
        ngram = " ".join(words[i : i + n])
        hashes.add(hashlib.sha256(ngram.encode()).hexdigest()[:16])
    return hashes


class TextFingerprintIndex:
    """Index for exact and near-duplicate text matching."""

    def __init__(self) -> None:
        self._exact: dict[str, list[str]] = {}  # hash -> sample_ids
        self._ngrams: dict[str, list[str]] = {}  # ngram_hash -> sample_ids

    def add(self, sample_id: str, text: str) -> None:
        """Add a text sample to the index."""
        exact_hash = compute_text_hash(text)
        self._exact.setdefault(exact_hash, []).append(sample_id)

        for ngram_hash in compute_ngram_hashes(text):
            self._ngrams.setdefault(ngram_hash, []).append(sample_id)

    def find_exact(self, text: str) -> list[str]:
        """Find exact matches for text."""
        h = compute_text_hash(text)
        return self._exact.get(h, [])

    def find_near(self, text: str, threshold: float = 0.5) -> list[tuple[str, float]]:
        """Find near-duplicate matches based on n-gram overlap."""
        query_ngrams = compute_ngram_hashes(text)
        if not query_ngrams:
            return []

        # Count matches per sample
        matches: dict[str, int] = {}
        for ngram_hash in query_ngrams:
            for sample_id in self._ngrams.get(ngram_hash, []):
                matches[sample_id] = matches.get(sample_id, 0) + 1

        # Compute similarity scores
        results = []
        for sample_id, match_count in matches.items():
            similarity = match_count / len(query_ngrams)
            if similarity >= threshold:
                results.append((sample_id, similarity))

        return sorted(results, key=lambda x: -x[1])

    def __len__(self) -> int:
        return len(self._exact)
