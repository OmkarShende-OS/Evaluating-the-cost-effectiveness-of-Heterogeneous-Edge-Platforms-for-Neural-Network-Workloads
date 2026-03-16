"""Compare devices by latency/FPS/EDCP from benchmark CSV."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics.edcp import compute_edcp


def main() -> None:
    parser = argparse.ArgumentParser(description="Device comparison from benchmark CSV")
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f"Input file does not exist: {path}")

    rows = defaultdict(list)
    with path.open("r", newline="", encoding="utf-8") as fp:
        for row in csv.DictReader(fp):
            rows[row["device"]].append(row)

    print("Device comparison summary:")
    for device, drows in rows.items():
        latency = sum(float(r["latency_ms"]) for r in drows) / len(drows)
        energy = sum(float(r["energy_j"]) for r in drows) / len(drows)
        cost = sum(float(r["cost_usd"]) for r in drows) / len(drows)
        edcp = compute_edcp(energy, latency / 1000.0, cost)
        fps = 1000.0 / latency if latency > 0 else 0.0
        print(f"- {device}: latency={latency:.3f}ms, fps={fps:.2f}, edcp={edcp:.6f}")


if __name__ == "__main__":
    main()
