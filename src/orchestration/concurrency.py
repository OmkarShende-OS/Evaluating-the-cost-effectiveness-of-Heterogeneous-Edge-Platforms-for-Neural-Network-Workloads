"""Simple config-driven concurrency experiment scaffolds."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ConcurrencyResult:
    strategy: str
    baseline_latency_ms: float
    concurrent_latency_ms: float

    @property
    def speedup(self) -> float:
        if self.concurrent_latency_ms <= 0:
            return 0.0
        return self.baseline_latency_ms / self.concurrent_latency_ms
