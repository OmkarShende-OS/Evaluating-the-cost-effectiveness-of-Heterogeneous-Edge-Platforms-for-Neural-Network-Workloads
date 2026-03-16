"""EDCP metric implementation and helpers.

Formula:
    EDCP = Energy * Delay * Cost
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional
import csv


@dataclass
class EDCPRecord:
    device: str
    runtime: str
    energy_j: float
    delay_s: float
    cost_usd: float
    edcp: float
    edcp_normalized: Optional[float] = None


def compute_edcp(energy_j: float, delay_s: float, cost_usd: float) -> float:
    if energy_j <= 0 or delay_s <= 0 or cost_usd <= 0:
        raise ValueError("EDCP inputs must be positive")
    return energy_j * delay_s * cost_usd


def normalize(values: Iterable[float], baseline: float) -> List[float]:
    if baseline <= 0:
        raise ValueError("baseline must be positive")
    return [v / baseline for v in values]


def records_from_csv(path: Path) -> List[EDCPRecord]:
    """Load benchmark rows and compute EDCP.

    Required CSV columns:
    - device
    - runtime
    - energy_j
    - latency_ms
    - cost_usd
    """
    rows: List[EDCPRecord] = []
    with path.open("r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for r in reader:
            latency_ms = float(r["latency_ms"])
            delay_s = latency_ms / 1000.0
            energy_j = float(r["energy_j"])
            cost_usd = float(r["cost_usd"])
            rows.append(
                EDCPRecord(
                    device=r["device"],
                    runtime=r["runtime"],
                    energy_j=energy_j,
                    delay_s=delay_s,
                    cost_usd=cost_usd,
                    edcp=compute_edcp(energy_j, delay_s, cost_usd),
                )
            )
    return rows


def apply_normalization(records: List[EDCPRecord], baseline_device: str) -> List[EDCPRecord]:
    baseline = next((r.edcp for r in records if r.device == baseline_device), None)
    if baseline is None:
        raise ValueError(f"baseline_device '{baseline_device}' not found")
    for r in records:
        r.edcp_normalized = r.edcp / baseline
    return records
