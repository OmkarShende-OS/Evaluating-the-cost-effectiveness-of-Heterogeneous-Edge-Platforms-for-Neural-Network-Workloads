# Architecture

## Layers

1. **Config layer** (`configs/`): declared devices/runtimes/models/experiments.
2. **Core source layer** (`src/`): metrics, benchmarking, profiling, orchestration, reporting.
3. **CLI workflow layer** (`scripts/`): user-facing execution entrypoints.
4. **Examples + artifacts layer** (`examples/`, `results/sample_outputs/`): onboarding and reproducibility.

## Design principles

- Capability-first organization
- Stable output schema
- No machine-specific hardcoded assumptions
- Graceful degradation when optional dependencies are missing
