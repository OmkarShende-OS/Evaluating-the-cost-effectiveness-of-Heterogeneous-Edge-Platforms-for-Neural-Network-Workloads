"""Run simple concurrency strategy comparisons from config-like arguments."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.orchestration.concurrency import ConcurrencyResult


def main() -> None:
    parser = argparse.ArgumentParser(description="Run concurrency experiment scaffold")
    parser.add_argument("--strategy", default="dwa")
    parser.add_argument("--baseline-latency-ms", type=float, default=20.0)
    parser.add_argument("--concurrent-latency-ms", type=float, default=15.0)
    args = parser.parse_args()

    res = ConcurrencyResult(
        strategy=args.strategy,
        baseline_latency_ms=args.baseline_latency_ms,
        concurrent_latency_ms=args.concurrent_latency_ms,
    )
    print(
        {
            "strategy": res.strategy,
            "baseline_latency_ms": res.baseline_latency_ms,
            "concurrent_latency_ms": res.concurrent_latency_ms,
            "speedup": round(res.speedup, 4),
        }
    )


if __name__ == "__main__":
    main()
