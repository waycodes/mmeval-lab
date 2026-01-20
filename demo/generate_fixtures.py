#!/usr/bin/env python3
"""Generate synthetic demo fixtures."""

import json
import random
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURES.mkdir(exist_ok=True)

# Disciplines and image types for MMMU-like data
DISCIPLINES = ["Science", "Engineering", "Business", "Health", "Humanities"]
IMAGE_TYPES = ["diagram", "chart", "photo", "table", "equation"]
SUBJECTS = {
    "Science": ["Physics", "Chemistry", "Biology"],
    "Engineering": ["Computer Science", "Electrical", "Mechanical"],
    "Business": ["Accounting", "Economics", "Marketing"],
    "Health": ["Medicine", "Nursing", "Pharmacy"],
    "Humanities": ["History", "Literature", "Philosophy"],
}

QUESTIONS = [
    "What is the primary function shown in the diagram?",
    "Based on the chart, which value is highest?",
    "What conclusion can be drawn from this data?",
    "Which option best describes the relationship shown?",
    "What is the correct interpretation of this figure?",
]


def generate_mmmu_samples(n: int = 50, seed: int = 42) -> list[dict]:
    """Generate synthetic MMMU-like samples."""
    random.seed(seed)
    samples = []

    for i in range(n):
        discipline = random.choice(DISCIPLINES)
        subject = random.choice(SUBJECTS[discipline])
        image_type = random.choice(IMAGE_TYPES)

        # Baseline accuracy ~85%, but varies by discipline
        base_prob = {
            "Science": 0.75,  # Harder
            "Engineering": 0.80,
            "Business": 0.90,
            "Health": 0.85,
            "Humanities": 0.92,
        }[discipline]

        # Candidate has regression in Science+diagram
        if discipline == "Science" and image_type == "diagram":
            cand_prob = 0.50  # Significant regression
        elif discipline == "Science":
            cand_prob = base_prob - 0.10  # Slight regression
        else:
            cand_prob = base_prob + 0.02  # Slight improvement elsewhere

        is_correct_baseline = random.random() < base_prob
        is_correct_candidate = random.random() < cand_prob

        samples.append({
            "id": f"mmmu_{i:04d}",
            "question": random.choice(QUESTIONS),
            "options": ["A. Option 1", "B. Option 2", "C. Option 3", "D. Option 4"],
            "answer": random.choice(["A", "B", "C", "D"]),
            "metadata": {
                "discipline": discipline,
                "subject": subject,
                "image_type": image_type,
            },
            "is_correct_baseline": is_correct_baseline,
            "is_correct_candidate": is_correct_candidate,
        })

    return samples


def generate_training_manifest(
    test_samples: list[dict], n_train: int = 100, contamination_rate: float = 0.05
) -> list[dict]:
    """Generate training manifest with some contamination."""
    random.seed(43)
    train_samples = []

    # Add some contaminated samples (exact copies of test questions)
    n_contaminated = int(len(test_samples) * contamination_rate)
    contaminated_tests = random.sample(test_samples, n_contaminated)

    for i, test in enumerate(contaminated_tests):
        train_samples.append({
            "id": f"train_contam_{i:04d}",
            "text": test["question"],  # Exact copy
            "source": "web_scrape",
        })

    # Add clean training samples
    clean_questions = [
        "Explain the concept of thermodynamics.",
        "What are the principles of supply and demand?",
        "Describe the structure of DNA.",
        "How does machine learning work?",
        "What is the significance of the Renaissance?",
        "Calculate the derivative of x^2.",
        "What are the symptoms of diabetes?",
        "Explain Newton's laws of motion.",
        "What is the role of mitochondria?",
        "Describe the water cycle.",
    ]

    for i in range(n_train - n_contaminated):
        train_samples.append({
            "id": f"train_clean_{i:04d}",
            "text": random.choice(clean_questions) + f" (variant {i})",
            "source": "textbook",
        })

    random.shuffle(train_samples)
    return train_samples


def main():
    # Generate test samples
    samples = generate_mmmu_samples(50)
    with open(FIXTURES / "mmmu_samples.json", "w") as f:
        json.dump(samples, f, indent=2)
    print(f"Generated {len(samples)} MMMU samples")

    # Generate training manifest
    train = generate_training_manifest(samples, n_train=100, contamination_rate=0.06)
    with open(FIXTURES / "training_manifest.jsonl", "w") as f:
        for sample in train:
            f.write(json.dumps(sample) + "\n")
    n_contam = sum(1 for t in train if "contam" in t["id"])
    print(f"Generated {len(train)} training samples ({n_contam} contaminated)")


if __name__ == "__main__":
    main()
