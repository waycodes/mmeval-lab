"""Tests for MCQ answer extraction."""

import pytest

from mmevallab.eval.extraction import extract_mcq_answer, extract_mcq_answer_strict


class TestExtractMCQAnswer:
    """Tests for the flexible MCQ extractor."""

    # Basic cases
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("A", "A"),
            ("B", "B"),
            ("C", "C"),
            ("D", "D"),
            ("E", "E"),
            ("a", "A"),
            ("b", "B"),
        ],
    )
    def test_single_letter(self, text: str, expected: str) -> None:
        assert extract_mcq_answer(text) == expected

    # Letter with punctuation
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("A.", "A"),
            ("B)", "B"),
            ("C:", "C"),
            ("(D)", "D"),
            ("[E]", "E"),
            ("A. ", "A"),
        ],
    )
    def test_letter_with_punctuation(self, text: str, expected: str) -> None:
        assert extract_mcq_answer(text) == expected

    # Common answer patterns
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("The answer is B", "B"),
            ("The answer is B.", "B"),
            ("Answer is C", "C"),
            ("Answer: D", "D"),
            ("the answer is A", "A"),
            ("The correct answer is B", "B"),
            ("correct answer is C", "C"),
        ],
    )
    def test_answer_patterns(self, text: str, expected: str) -> None:
        assert extract_mcq_answer(text) == expected

    # Verbose outputs
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("The answer is B because the image shows...", "B"),
            ("Based on my analysis, the correct answer is C. Here's why...", "C"),
            ("Looking at the options, A is correct since...", "A"),
            ("I believe the answer is D. The reasoning is...", "D"),
        ],
    )
    def test_verbose_outputs(self, text: str, expected: str) -> None:
        assert extract_mcq_answer(text) == expected

    # Adversarial cases
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("B\n\nExplanation: The image shows...", "B"),
            ("**B**", "B"),
            ("Option B is correct", "B"),
            ("The best choice is option C", "C"),
            ("A) is the right answer", "A"),
            ("I would select B", "B"),
            ("My answer: C", "C"),
        ],
    )
    def test_adversarial_formats(self, text: str, expected: str) -> None:
        assert extract_mcq_answer(text) == expected

    # Edge cases
    def test_empty_string(self) -> None:
        assert extract_mcq_answer("") is None

    def test_none_input(self) -> None:
        assert extract_mcq_answer(None) is None  # type: ignore

    def test_no_valid_answer(self) -> None:
        assert extract_mcq_answer("I don't know") is None
        assert extract_mcq_answer("The image is unclear") is None

    def test_whitespace(self) -> None:
        assert extract_mcq_answer("  B  ") == "B"
        assert extract_mcq_answer("\nA\n") == "A"

    # Custom valid options
    def test_custom_valid_options(self) -> None:
        assert extract_mcq_answer("F", valid_options="ABCDEF") == "F"
        assert extract_mcq_answer("F", valid_options="ABCD") is None
        assert extract_mcq_answer("G", valid_options="ABCDEFG") == "G"

    # Tricky cases with multiple letters
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("A and B are both possible, but A is correct", "A"),
            ("Between A and C, I choose A", "A"),
            ("Not B, the answer is C", "C"),  # Should get C from "answer is C"
        ],
    )
    def test_multiple_letters(self, text: str, expected: str) -> None:
        assert extract_mcq_answer(text) == expected


class TestExtractMCQAnswerStrict:
    """Tests for the strict MCQ extractor."""

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("A", "A"),
            ("B.", "B"),
            ("(C)", "C"),
            ("[D]", "D"),
            ("The answer is B", "B"),
            ("Answer is C", "C"),
        ],
    )
    def test_clean_formats(self, text: str, expected: str) -> None:
        assert extract_mcq_answer_strict(text) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "The answer is B because...",
            "I think B is correct",
            "B, since the image shows...",
            "Looking at this, B",
        ],
    )
    def test_rejects_verbose(self, text: str) -> None:
        assert extract_mcq_answer_strict(text) is None

    def test_empty_and_none(self) -> None:
        assert extract_mcq_answer_strict("") is None
        assert extract_mcq_answer_strict(None) is None  # type: ignore
