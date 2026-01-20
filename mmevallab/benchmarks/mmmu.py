"""MMMU benchmark adapter."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from typing import Any

from mmevallab.core.datamodel import Example, MediaRef, Prediction
from mmevallab.core.registry import Benchmark, register_benchmark

MMMU_SUBJECTS = [
    "Accounting", "Agriculture", "Architecture_and_Engineering", "Art", "Art_Theory",
    "Basic_Medical_Science", "Biology", "Chemistry", "Clinical_Medicine", "Computer_Science",
    "Design", "Diagnostics_and_Laboratory_Medicine", "Economics", "Electronics", "Energy_and_Power",
    "Finance", "Geography", "History", "Literature", "Manage", "Marketing", "Materials",
    "Math", "Mechanical_Engineering", "Music", "Pharmacy", "Physics", "Psychology",
    "Public_Health", "Sociology",
]

# Subject to discipline mapping
SUBJECT_TO_DISCIPLINE = {
    "Art": "Art & Design",
    "Art_Theory": "Art & Design",
    "Design": "Art & Design",
    "Music": "Art & Design",
    "Accounting": "Business",
    "Economics": "Business",
    "Finance": "Business",
    "Manage": "Business",
    "Marketing": "Business",
    "Basic_Medical_Science": "Health & Medicine",
    "Clinical_Medicine": "Health & Medicine",
    "Diagnostics_and_Laboratory_Medicine": "Health & Medicine",
    "Pharmacy": "Health & Medicine",
    "Public_Health": "Health & Medicine",
    "Geography": "Humanities & Social Science",
    "History": "Humanities & Social Science",
    "Literature": "Humanities & Social Science",
    "Psychology": "Humanities & Social Science",
    "Sociology": "Humanities & Social Science",
    "Agriculture": "Science",
    "Biology": "Science",
    "Chemistry": "Science",
    "Math": "Science",
    "Physics": "Science",
    "Architecture_and_Engineering": "Tech & Engineering",
    "Computer_Science": "Tech & Engineering",
    "Electronics": "Tech & Engineering",
    "Energy_and_Power": "Tech & Engineering",
    "Materials": "Tech & Engineering",
    "Mechanical_Engineering": "Tech & Engineering",
}


def _load_mmmu_dataset(split: str) -> Any:
    """Load MMMU dataset from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("Install datasets: pip install datasets") from e

    # MMMU has subject-specific configs
    all_data = []
    for subject in MMMU_SUBJECTS:
        try:
            ds = load_dataset("MMMU/MMMU", subject, split=split, trust_remote_code=True)
            all_data.extend(ds)
        except Exception:
            # Some subjects may not exist in all splits
            continue
    return all_data


def _compute_content_hash(data: list[Any]) -> str:
    """Compute hash of dataset content for reproducibility."""
    ids = sorted(item.get("id", str(i)) for i, item in enumerate(data))
    return hashlib.sha256("|".join(ids).encode()).hexdigest()[:16]


@register_benchmark("mmmu")
class MMMUBenchmark(Benchmark):
    """MMMU: Massive Multi-discipline Multimodal Understanding benchmark."""

    @property
    def name(self) -> str:
        return "mmmu"

    def load(self, split: str, **kwargs: Any) -> Iterator[Example]:
        """Load MMMU examples.

        Args:
            split: One of 'dev', 'validation', 'test'
            **kwargs: Additional options (e.g., limit for testing)

        Yields:
            Example objects with stable example_id
        """
        data = _load_mmmu_dataset(split)
        limit = kwargs.get("limit")

        for i, item in enumerate(data):
            if limit and i >= limit:
                break

            example_id = item.get("id", f"mmmu_{split}_{i}")

            # Collect images
            media = []
            img_keys = [f"image_{i}" for i in range(1, 8)]
            for img_key in img_keys:
                img = item.get(img_key)
                if img is not None:
                    media.append(MediaRef(type="image", path=None))

            # Build inputs
            options = []
            for opt_key in ["A", "B", "C", "D", "E", "F", "G"]:
                opt = item.get(opt_key)
                if opt:
                    options.append(f"{opt_key}. {opt}")

            inputs = {
                "question": item.get("question", ""),
                "options": options,
                "question_type": item.get("question_type", "multiple-choice"),
            }

            # Store raw images in metadata for later access
            raw_images = {}
            for img_key in img_keys:
                img = item.get(img_key)
                if img is not None:
                    raw_images[img_key] = img

            subject = item.get("subject", "")
            discipline = SUBJECT_TO_DISCIPLINE.get(subject, "Unknown")

            metadata = {
                "discipline": discipline,
                "subject": subject,
                "subfield": item.get("subfield", ""),
                "topic_difficulty": item.get("topic_difficulty", ""),
                "image_type": item.get("image_type", ""),
                "num_images": len(media),
                "split": split,
                "_raw_images": raw_images,
            }

            # Ground truth (None for test split)
            ground_truth = item.get("answer") if split != "test" else None

            yield Example(
                example_id=example_id,
                inputs=inputs,
                media=media,
                metadata=metadata,
                ground_truth=ground_truth,
            )

    def score(self, example: Example, prediction: Prediction) -> dict[str, Any]:
        """Score prediction against ground truth."""
        if example.ground_truth is None:
            return {"is_correct": None, "accuracy": None}

        # Normalize answers for comparison
        gt = example.ground_truth.strip().upper()
        pred = (prediction.extracted_answer or "").strip().upper()

        is_correct = pred == gt
        return {
            "is_correct": is_correct,
            "accuracy": 1.0 if is_correct else 0.0,
            "ground_truth": gt,
            "predicted": pred,
        }
