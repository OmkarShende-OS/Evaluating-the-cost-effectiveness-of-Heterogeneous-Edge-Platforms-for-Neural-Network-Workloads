# Internal Inventory (Keep / Refactor / Ignore)

## Candidate source pools scanned

- `bang-for-the-buck/` (current repository)
- Reference alignment docs from workspace root:
  - `README.md`
  - `REPOSITORY_SUITE_SUMMARY.md`
  - `UPLOAD_CHECKLIST.md`

## Inventory decisions

| Path | Decision | Rationale |
|---|---|---|
| `1_roofline_analysis/` | keep | Direct paper capability and model-based performance framing |
| `2_benchmark_harness/` | keep + refactor wrappers | Strong core benchmarking loop; exposed via new `scripts/` and `src/` |
| `3_metrics/` | keep + refactor wrappers | Contains EDCP/energy/temp logic relevant to public artifact |
| `4_sw_framework_comparison/` | keep + refactor wrappers | Runtime comparison capability is central to paper and repo goals |
| `5_concurrent_execution/` | keep + refactor wrappers | Concurrency/workload allocation is required by paper and deliverables |
| `6_application_case_study/` | keep | Real-world deployment framing |
| `visualizations/` | keep | Figure reproduction and reporting support |
| `results/` | replace with sample outputs + generated outputs | Empty previously; now structured for reproducibility |

## Public-facing additions

- Capability-first docs layer in `docs/`
- Modular source package in `src/`
- CLI script layer in `scripts/`
- Reproducible sample outputs in `results/sample_outputs/`
- Tests in `tests/`
