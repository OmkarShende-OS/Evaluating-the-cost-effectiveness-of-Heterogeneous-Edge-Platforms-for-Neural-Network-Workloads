# Setup Guide

## Environment

Use either:

- `pip install -r requirements.txt`
- or `conda env create -f environment.yml` (if maintained in your workflow)

## Quick verification

1. `python examples/quickstart_cpu.py`
2. `python scripts/compute_edcp.py --input results/sample_outputs/tables/device_comparison_sample.csv`
3. `python scripts/generate_reports.py --input results/sample_outputs/tables/device_comparison_sample.csv`

## Optional dependencies

Some runtime backends (e.g., TensorRT/OpenVINO/vendor NPUs) require platform-specific SDK installation. The generic scripts degrade gracefully and will emit capability warnings if optional dependencies are unavailable.
