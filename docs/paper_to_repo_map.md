# Paper-to-Repository Map

| Paper Area | Repository Mapping |
|---|---|
| Motivation and framing | `README.md`, `docs/overview.md` |
| Methodology | `docs/methodology.md`, `docs/reproducibility.md` |
| Hardware/platform analysis | `docs/hardware_matrix.md`, `1_roofline_analysis/` |
| Runtime/software stack analysis | `docs/runtime_matrix.md`, `4_sw_framework_comparison/`, `scripts/compare_runtimes.py` |
| Metrics and EDCP | `docs/metrics.md`, `docs/edcp.md`, `src/metrics/edcp.py`, `scripts/compute_edcp.py` |
| Benchmarking/profiling pipeline | `2_benchmark_harness/`, `3_metrics/`, `scripts/run_benchmark.py`, `scripts/profile_power_temperature.py` |
| Concurrency/workload allocation | `docs/concurrency_and_workload_allocation.md`, `5_concurrent_execution/`, `scripts/run_concurrency_experiments.py` |
| Real-world application case study | `6_application_case_study/`, `docs/case_study.md` |
| Reporting and artifacts | `scripts/generate_reports.py`, `src/reporting/`, `results/sample_outputs/` |
