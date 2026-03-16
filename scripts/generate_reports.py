"""Generate markdown EDCP comparison report from benchmark CSV input."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics.edcp import apply_normalization, records_from_csv
from src.reporting.report import write_edcp_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark markdown reports")
    parser.add_argument("--input", required=True)
    parser.add_argument("--baseline-device", default="Odroid-M1")
    parser.add_argument(
        "--output",
        default="results/sample_outputs/reports/edcp_report.md",
    )
    args = parser.parse_args()

    records = records_from_csv(Path(args.input))
    apply_normalization(records, args.baseline_device)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_edcp_markdown(records, out_path)
    print(f"Wrote report to {out_path}")


if __name__ == "__main__":
    main()
