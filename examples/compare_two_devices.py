"""Compare two devices using sample CSV and EDCP."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics.edcp import records_from_csv


def main() -> None:
    records = records_from_csv(Path("results/sample_outputs/tables/device_comparison_sample.csv"))
    d1 = [r for r in records if r.device == "Odroid-M1"]
    d2 = [r for r in records if r.device == "Jetson-AGX"]
    print(f"Odroid-M1 rows: {len(d1)} | Jetson-AGX rows: {len(d2)}")


if __name__ == "__main__":
    main()
