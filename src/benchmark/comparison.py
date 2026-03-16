"""Device/runtime comparison utilities."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Iterable, List


@dataclass
class BenchmarkRow:
    device: str
    runtime: str
    latency_ms: float
    energy_j: float
    cost_usd: float

    @property
    def fps(self) -> float:
        return 1000.0 / self.latency_ms if self.latency_ms > 0 else 0.0


def summarize_latency(rows: Iterable[BenchmarkRow]) -> dict:
    values: List[float] = [r.latency_ms for r in rows]
    if not values:
        return {"mean_ms": 0.0}
    return {"mean_ms": round(mean(values), 4)}
