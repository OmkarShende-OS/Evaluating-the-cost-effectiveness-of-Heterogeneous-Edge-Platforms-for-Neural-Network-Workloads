"""Runtime selection demo: choose runtime by weighted objective."""

from __future__ import annotations

from pathlib import Path
import csv


def score(latency_ms: float, energy_j: float, weight_latency: float = 0.6) -> float:
    return weight_latency * latency_ms + (1.0 - weight_latency) * (energy_j * 100)


def main() -> None:
    rows = []
    path = Path("results/sample_outputs/tables/device_comparison_sample.csv")
    with path.open("r", newline="", encoding="utf-8") as fp:
        for r in csv.DictReader(fp):
            r["score"] = score(float(r["latency_ms"]), float(r["energy_j"]))
            rows.append(r)
    best = min(rows, key=lambda x: x["score"])
    print(f"Selected runtime={best['runtime']} on device={best['device']} with score={best['score']:.4f}")


if __name__ == "__main__":
    main()
