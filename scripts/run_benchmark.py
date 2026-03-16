"""Run benchmark summarization from CSV input.

This script keeps the public workflow simple and path-agnostic.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize benchmark CSV rows")
    parser.add_argument("--input", required=True, help="Path to benchmark CSV")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file does not exist: {input_path}")

    with input_path.open("r", newline="", encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))

    print(f"Loaded {len(rows)} benchmark rows from {input_path}")


if __name__ == "__main__":
    main()
