"""Compare runtimes by mean latency using benchmark CSV input."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Runtime comparison from benchmark CSV")
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f"Input file does not exist: {path}")

    table = defaultdict(list)
    with path.open("r", newline="", encoding="utf-8") as fp:
        for row in csv.DictReader(fp):
            table[row["runtime"]].append(float(row["latency_ms"]))

    ranking = sorted(
        ((runtime, sum(vals) / len(vals)) for runtime, vals in table.items()),
        key=lambda x: x[1],
    )

    print("Runtime ranking (lower latency is better):")
    for i, (runtime, mean_ms) in enumerate(ranking, start=1):
        print(f"{i}. {runtime}: {mean_ms:.3f} ms")


if __name__ == "__main__":
    main()
