"""Tests for deterministic run-id hashing."""

from mmevallab.core.hashing import compute_dataset_id, compute_run_id


def test_run_id_deterministic() -> None:
    """Run ID should be deterministic for same inputs."""
    config = {"benchmark": "mmmu", "model": "test", "split": "val"}
    dataset_id = "MMMU:1.0:val:abc12345"
    code_version = "deadbeef1234"

    id1 = compute_run_id(config, dataset_id, code_version)
    id2 = compute_run_id(config, dataset_id, code_version)

    assert id1 == id2
    assert len(id1) == 12


def test_run_id_changes_with_config() -> None:
    """Run ID should change when config changes."""
    config1 = {"benchmark": "mmmu", "model": "test"}
    config2 = {"benchmark": "mmmu", "model": "other"}
    dataset_id = "MMMU:1.0:val:abc12345"
    code_version = "deadbeef1234"

    id1 = compute_run_id(config1, dataset_id, code_version)
    id2 = compute_run_id(config2, dataset_id, code_version)

    assert id1 != id2


def test_run_id_golden() -> None:
    """Golden test for stable run ID computation."""
    config = {"benchmark": "mmmu", "model": "test", "split": "val"}
    dataset_id = "MMMU:1.0:val:abc12345"
    code_version = "deadbeef1234"

    run_id = compute_run_id(config, dataset_id, code_version)
    # This is the expected stable hash - if this changes, hashing is broken
    assert run_id == "4ef55ffb34ec"


def test_dataset_id_format() -> None:
    """Dataset ID should have expected format."""
    did = compute_dataset_id("MMMU", "1.0", "val", "abc123456789")
    assert did == "MMMU:1.0:val:abc12345"
