"""Base runtime adapter abstraction for capability-first benchmarking scripts."""

from dataclasses import dataclass
from typing import Any


@dataclass
class AdapterMetadata:
    name: str
    runtime: str
    precision: str


class InferenceAdapter:
    """Minimal adapter contract used by this public artifact."""

    metadata: AdapterMetadata

    def load(self) -> None:
        raise NotImplementedError

    def infer(self, input_data: Any) -> Any:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError
