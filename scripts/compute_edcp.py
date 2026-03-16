"""Compute EDCP and optional normalized EDCP from benchmark CSV input."""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics.edcp import apply_normalization, records_from_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute EDCP from benchmark rows")
    parser.add_argument("--input", required=True, help="CSV with device/runtime/energy/latency/cost")
    parser.add_argument("--baseline-device", default="Odroid-M1")
    parser.add_argument("--output", default="results/sample_outputs/tables/edcp_output.json")
    args = parser.parse_args()

    records = records_from_csv(Path(args.input))
    apply_normalization(records, baseline_device=args.baseline_device)

    out = [
        {
            "device": r.device,
            "runtime": r.runtime,
            "energy_j": r.energy_j,
            "delay_s": r.delay_s,
            "cost_usd": r.cost_usd,
            "edcp": r.edcp,
            "edcp_normalized": r.edcp_normalized,
        }
        for r in records
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote EDCP output to {output_path}")


if __name__ == "__main__":
    main()
