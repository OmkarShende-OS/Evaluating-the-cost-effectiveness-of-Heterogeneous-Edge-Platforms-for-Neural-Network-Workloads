"""Profile power and temperature summary from synthetic or measured CSV columns."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.profiling.power_thermal import summarize_power_temperature


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize power/temperature metrics")
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f"Input file does not exist: {path}")

    power = []
    temp = []
    with path.open("r", newline="", encoding="utf-8") as fp:
        for row in csv.DictReader(fp):
            if "power_w" in row and row["power_w"]:
                power.append(float(row["power_w"]))
            if "temperature_c" in row and row["temperature_c"]:
                temp.append(float(row["temperature_c"]))

    print(summarize_power_temperature(power, temp))


if __name__ == "__main__":
    main()
