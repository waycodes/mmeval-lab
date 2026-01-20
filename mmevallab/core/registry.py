"""Plugin registry for benchmarks and models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any, Callable, Generic, TypeVar

from mmevallab.core.datamodel import Example, Prediction

T = TypeVar("T")


class Benchmark(ABC):
    """Base class for benchmark adapters."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Benchmark name."""
        ...

    @abstractmethod
    def load(self, split: str, **kwargs: Any) -> Iterator[Example]:
        """Load examples from the benchmark."""
        ...

    @abstractmethod
    def score(self, example: Example, prediction: Prediction) -> dict[str, Any]:
        """Score a single prediction against ground truth."""
        ...


class ModelRunner(ABC):
    """Base class for model runners."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Model name."""
        ...

    @abstractmethod
    def generate(self, example: Example) -> Prediction:
        """Generate prediction for an example."""
        ...


BenchmarkFactory = Callable[..., Benchmark]
ModelFactory = Callable[..., ModelRunner]


class Registry(Generic[T]):
    """Generic registry for plugins."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._registry: dict[str, Callable[..., T]] = {}

    def register(self, name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Decorator to register a factory."""

        def decorator(factory: Callable[..., T]) -> Callable[..., T]:
            if name in self._registry:
                raise ValueError(f"{self._name} '{name}' already registered")
            self._registry[name] = factory
            return factory

        return decorator

    def get(self, name: str) -> Callable[..., T]:
        """Get a registered factory by name."""
        if name not in self._registry:
            available = ", ".join(sorted(self._registry.keys())) or "(none)"
            raise KeyError(f"Unknown {self._name}: '{name}'. Available: {available}")
        return self._registry[name]

    def create(self, name: str, **kwargs: Any) -> T:
        """Create an instance from a registered factory."""
        return self.get(name)(**kwargs)

    def list(self) -> list[str]:
        """List all registered names."""
        return sorted(self._registry.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._registry


# Global registries
benchmark_registry: Registry[Benchmark] = Registry("benchmark")
model_registry: Registry[ModelRunner] = Registry("model")


def register_benchmark(name: str) -> Callable[[BenchmarkFactory], BenchmarkFactory]:
    """Register a benchmark factory."""
    return benchmark_registry.register(name)


def register_model(name: str) -> Callable[[ModelFactory], ModelFactory]:
    """Register a model factory."""
    return model_registry.register(name)
