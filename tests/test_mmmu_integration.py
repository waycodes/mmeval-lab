"""Integration test: MMMU smoke end-to-end."""

import pytest

from mmevallab.core.registry import benchmark_registry, model_registry


class TestMMMUIntegration:
    """MMMU end-to-end integration tests."""

    def test_registries_exist(self) -> None:
        """Test that registries are properly initialized."""
        # Import to trigger registration
        import mmevallab.benchmarks  # noqa: F401
        import mmevallab.models  # noqa: F401

        # Verify registration via list method
        benchmarks = benchmark_registry.list()
        models = model_registry.list()

        assert "mmmu" in benchmarks
        assert "dummy" in models

    def test_benchmark_creation(self) -> None:
        """Test that MMMU benchmark can be created."""
        import mmevallab.benchmarks  # noqa: F401

        benchmark = benchmark_registry.create("mmmu")
        assert benchmark.name == "mmmu"

    def test_model_creation(self) -> None:
        """Test that dummy model can be created."""
        import mmevallab.models  # noqa: F401

        model = model_registry.create("dummy")
        assert model.name == "dummy"

    def test_dummy_model_generate(self) -> None:
        """Test dummy model generation."""
        import mmevallab.models  # noqa: F401
        from mmevallab.core.datamodel import Example

        model = model_registry.create("dummy")
        example = Example(
            example_id="test1",
            inputs={"question": "What is 5+1?"},
            ground_truth="6",
        )
        pred = model.generate(example)
        assert pred.example_id == "test1"
        assert pred.extracted_answer == "6"
