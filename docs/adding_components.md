# Adding Benchmarks and Models

## Adding a New Benchmark

1. Create a new file in `mmevallab/benchmarks/`:

```python
from mmevallab.core.datamodel import Example, Prediction
from mmevallab.core.registry import Benchmark, register_benchmark

@register_benchmark("my_benchmark")
class MyBenchmark(Benchmark):
    def __init__(self, data_dir: str | None = None) -> None:
        self._data_dir = data_dir

    @property
    def name(self) -> str:
        return "my_benchmark"

    def load(self, split: str, limit: int | None = None):
        # Load and yield Example objects
        for item in load_data(split):
            yield Example(
                example_id=item["id"],
                inputs={"question": item["question"], "images": item["images"]},
                ground_truth=item["answer"],
                metadata=item.get("metadata", {}),
            )

    def score(self, example: Example, prediction: Prediction) -> dict:
        is_correct = prediction.extracted_answer == example.ground_truth
        return {"is_correct": is_correct}
```

2. Import in `mmevallab/benchmarks/__init__.py`

## Adding a New Model

1. Create or extend `mmevallab/models/__init__.py`:

```python
from mmevallab.core.datamodel import Example, Prediction
from mmevallab.core.registry import ModelRunner, register_model

@register_model("my_model")
class MyModel(ModelRunner):
    def __init__(self, model_path: str, **kwargs) -> None:
        self.model_path = model_path
        self._model = None

    @property
    def name(self) -> str:
        return f"my_model:{self.model_path}"

    def generate(self, example: Example) -> Prediction:
        # Load model lazily
        if self._model is None:
            self._model = load_model(self.model_path)

        # Generate prediction
        output = self._model.generate(example.inputs)

        return Prediction(
            example_id=example.example_id,
            raw_output=output,
            extracted_answer=extract_answer(output),
            latency_ms=elapsed_ms,
        )
```

## Using Custom Components

```bash
mmeval run --benchmark my_benchmark --model my_model --split test
```

Or in config:

```yaml
benchmark: my_benchmark
model:
  name: my_model
  model_path: /path/to/model
```
