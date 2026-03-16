# Reproducibility

## Environment

- Python 3.9+
- Dependencies from `requirements.txt`

## Minimal reproducible path

1. Run `python examples/quickstart_cpu.py`.
2. Run `python scripts/compute_edcp.py --input results/sample_outputs/tables/device_comparison_sample.csv`.
3. Run `python scripts/generate_reports.py --input results/sample_outputs/tables/device_comparison_sample.csv`.

## Output conventions

- CSV: tabular benchmark results
- JSON: structured summary and normalized metrics
- Markdown report: human-readable comparison

## Determinism notes

- Synthetic sample outputs are fixed and versioned.
- Real hardware measurements naturally vary; use repeated runs and percentile stats.

## What is intentionally not included

- Private raw datasets
- Host-specific paths and credentials
- Uncurated local backup trees
