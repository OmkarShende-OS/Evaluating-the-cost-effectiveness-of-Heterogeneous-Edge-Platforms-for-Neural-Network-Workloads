"""Quickstart NPU/VPU-style row selection demo."""

from pathlib import Path
import csv


def main() -> None:
    path = Path("results/sample_outputs/tables/device_comparison_sample.csv")
    with path.open("r", newline="", encoding="utf-8") as fp:
        rows = [
            r
            for r in csv.DictReader(fp)
            if any(x in r["runtime"].lower() for x in ["npu", "vpu", "rknn", "ksnn", "openvino-vpu"])
        ]
    print(f"NPU/VPU-flavored rows: {len(rows)}")


if __name__ == "__main__":
    main()
