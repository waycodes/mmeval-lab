"""Tests for attribution sanity checks."""

import pytest

from mmevallab.attribution.sanity import (
    SanityCheckResult,
    check_attribution_coverage,
    check_self_attribution,
    run_sanity_checks,
)
from mmevallab.attribution.similarity import Attribution


class TestSelfAttribution:
    """Tests for self-attribution check."""

    def test_no_self_attribution(self) -> None:
        attrs = {
            "test1": [Attribution("test1", "train1", 0.8, "jaccard")],
            "test2": [Attribution("test2", "train2", 0.7, "jaccard")],
        }
        result = check_self_attribution(attrs, {"test1", "test2"})
        assert result.passed

    def test_detects_self_attribution(self) -> None:
        attrs = {
            "test1": [Attribution("test1", "test2", 0.9, "jaccard")],  # test2 is in test set
        }
        result = check_self_attribution(attrs, {"test1", "test2"})
        assert not result.passed
        assert "1" in result.message


class TestCoverage:
    """Tests for attribution coverage check."""

    def test_full_coverage(self) -> None:
        attrs = {
            "test1": [Attribution("test1", "train1", 0.8, "jaccard")],
            "test2": [Attribution("test2", "train2", 0.7, "jaccard")],
        }
        result = check_attribution_coverage(attrs, min_coverage=0.5)
        assert result.passed

    def test_low_coverage(self) -> None:
        attrs = {
            "test1": [Attribution("test1", "train1", 0.8, "jaccard")],
            "test2": [],  # No attributions
            "test3": [],
            "test4": [],
        }
        result = check_attribution_coverage(attrs, min_coverage=0.5)
        assert not result.passed


class TestRunAllChecks:
    """Tests for running all sanity checks."""

    def test_all_pass(self) -> None:
        attrs = {
            "test1": [Attribution("test1", "train1", 0.8, "jaccard")],
        }
        results = run_sanity_checks(attrs, {"test1"})
        assert len(results) == 2
        assert all(r.passed for r in results)
