"""Collect host system information into JSON output."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect basic system info")
    parser.add_argument("--output", default="results/sample_outputs/tables/system_info.json")
    args = parser.parse_args()

    payload = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "system": platform.system(),
        "release": platform.release(),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote system info to {out}")


if __name__ == "__main__":
    main()
