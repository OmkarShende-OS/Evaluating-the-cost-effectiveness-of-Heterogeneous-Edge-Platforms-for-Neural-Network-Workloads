"""Quickstart GPU-style runtime comparison flow."""

from pathlib import Path
import csv


def main() -> None:
    path = Path("results/sample_outputs/tables/device_comparison_sample.csv")
    with path.open("r", newline="", encoding="utf-8") as fp:
        rows = [r for r in csv.DictReader(fp) if "gpu" in r["runtime"].lower() or "tensorrt" in r["runtime"].lower()]
    print(f"GPU-flavored rows: {len(rows)}")


if __name__ == "__main__":
    main()
