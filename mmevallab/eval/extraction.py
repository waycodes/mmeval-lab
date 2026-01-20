"""MCQ answer extraction utilities."""

from __future__ import annotations

import re

# Valid MCQ answer letters
VALID_ANSWERS = frozenset("ABCDEFG")


def extract_mcq_answer(text: str, valid_options: str = "ABCDE") -> str | None:
    """Extract MCQ answer letter from model output.

    Handles various output formats:
    - Direct letter: "B"
    - With period: "B."
    - Parenthesized: "(B)"
    - Bracketed: "[B]"
    - "The answer is B"
    - "Answer: B"
    - Verbose: "The correct answer is B because..."

    Args:
        text: Model output text
        valid_options: String of valid option letters (default "ABCDE")

    Returns:
        Extracted letter (uppercase) or None if not found
    """
    if not text:
        return None

    text = text.strip()
    valid_set = frozenset(valid_options.upper())

    # Strategy 1: Check if entire response is just a letter
    if len(text) == 1 and text.upper() in valid_set:
        return text.upper()

    # Strategy 2: Check for letter with period at start
    if len(text) >= 1 and text[0].upper() in valid_set:
        if len(text) == 1 or text[1] in ".):] \n":
            return text[0].upper()

    # Strategy 3: Look for common patterns
    patterns = [
        # "The answer is B" / "Answer is B" / "answer: B"
        r"(?:the\s+)?answer\s*(?:is|:)\s*\(?([A-G])\)?",
        # "correct answer is B"
        r"correct\s+(?:answer|option)\s*(?:is|:)\s*\(?([A-G])\)?",
        # "select B" / "choose B"
        r"(?:select|choose|pick)\s+\(?([A-G])\)?",
        # "Option B" / "option (B)"
        r"option\s*\(?([A-G])\)?",
        # "(B)" or "[B]" standalone
        r"[\(\[]([A-G])[\)\]]",
        # "B." or "B)" at start of line
        r"^([A-G])[.\):]",
        # Letter followed by period/paren at word boundary
        r"\b([A-G])[.\):]",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            letter = match.group(1).upper()
            if letter in valid_set:
                return letter

    # Strategy 4: Look for standalone letter (not part of a word)
    # This is more conservative than finding any letter
    standalone_match = re.search(r"(?<![a-zA-Z])([A-G])(?![a-zA-Z])", text)
    if standalone_match:
        letter = standalone_match.group(1).upper()
        if letter in valid_set:
            return letter

    return None


def extract_mcq_answer_strict(text: str, valid_options: str = "ABCDE") -> str | None:
    """Strict MCQ extraction - only accepts clean answers.

    Only extracts if the answer is clearly stated without ambiguity.
    Returns None for messy outputs that might be misinterpreted.

    Args:
        text: Model output text
        valid_options: String of valid option letters

    Returns:
        Extracted letter or None
    """
    if not text:
        return None

    text = text.strip()
    valid_set = frozenset(valid_options.upper())

    # Only accept very clean formats
    # Single letter
    if len(text) == 1 and text.upper() in valid_set:
        return text.upper()

    # Letter with period/paren
    if len(text) <= 3:
        clean = text.strip("().[]:")
        if len(clean) == 1 and clean.upper() in valid_set:
            return clean.upper()

    # "The answer is X" pattern
    match = re.match(
        r"^(?:the\s+)?(?:correct\s+)?answer\s*(?:is|:)\s*\(?([A-G])\)?\.?$",
        text,
        re.IGNORECASE,
    )
    if match and match.group(1).upper() in valid_set:
        return match.group(1).upper()

    return None
