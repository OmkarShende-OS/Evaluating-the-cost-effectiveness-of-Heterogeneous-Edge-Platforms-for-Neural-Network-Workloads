"""Quickstart CPU-oriented EDCP flow."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics.edcp import records_from_csv, apply_normalization


def main() -> None:
    path = Path("results/sample_outputs/tables/device_comparison_sample.csv")
    records = records_from_csv(path)
    apply_normalization(records, baseline_device="Odroid-M1")
    print(f"Loaded {len(records)} records and computed normalized EDCP.")


if __name__ == "__main__":
    main()
