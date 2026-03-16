# Evaluating the Cost-Effectiveness of Heterogeneous Edge Platforms for Neural Network Workloads

Capability-first research-engineering repository accompanying the paper **"Evaluating the Cost-Effectiveness of Heterogeneous Edge Platforms for Neural Network Workloads"**.

This repository demonstrates practical systems work across heterogeneous edge hardware, runtime stacks, profiling pipelines, and reproducible evaluation artifacts—not just paper figures.

## Why this repo exists

Most edge-AI publications share conclusions but not end-to-end engineering pipelines. This repository focuses on that missing layer: how to benchmark, profile, compare, and reason about edge inference deployments across CPUs, GPUs, NPUs, and VPUs.

## What this repository demonstrates

- Cross-device inference benchmarking for heterogeneous edge systems.
- Runtime/backend comparison across ONNX Runtime, TensorRT, OpenVINO, TFLite, ARMNN-style flows.
- Latency, throughput, energy, power, and temperature profiling.
- Cost-aware evaluation with **EDCP** (Energy × Delay × Cost Product).
- Concurrent workload allocation experiments (DWA/LWA-inspired pipelines).
- Capability-oriented scripts for device and runtime comparison.
- Reproducible reporting with structured CSV/JSON outputs.
- Practical example workflows for CPU/GPU/NPU/VPU-oriented evaluation.

## Paper summary (short)

The paper evaluates eight heterogeneous edge platforms and multiple inference runtimes, studies software-stack effects, introduces EDCP for cost-effectiveness, and analyzes concurrent workload allocation across co-processors.

## Architecture

1. **Configs** define devices, runtimes, models, and experiment profiles.
2. **Adapters + benchmarking modules** execute model inference runs.
3. **Profiling modules** collect latency/power/thermal data.
4. **Metrics modules** compute EDCP and comparative summaries.
5. **Reporting modules** export tables and markdown reports.

## Supported hardware classes

- ARM edge SBCs (e.g., Odroid/Rockchip class)
- NVIDIA Jetson class (CPU/GPU/DLA-style workflows)
- x86 + accelerator class (e.g., NCS2/VPU-like)
- Heterogeneous CPU + accelerator combinations

## Supported runtime/backend categories

- ONNX Runtime (CPU/CUDA/TensorRT EP style)
- TensorRT
- OpenVINO
- TensorFlow Lite
- ARM NN / PyARMNN-style flows
- Vendor NPUs (KSNN / RKNN class)

## Metrics collected

- Inference latency (mean, median, p90, p99)
- End-to-end latency
- Throughput (FPS)
- Energy per inference
- Average power
- Temperature
- Cost and normalized cost-effectiveness
- EDCP and optional normalized EDCP

## EDCP at a glance

Core formula used in this repository:

$$
EDCP = Energy \times Delay \times Cost
$$

Lower EDCP indicates better cost-effectiveness for practical edge deployment.

## Quickstart

Install dependencies:

- Python 3.9+
- `pip install -r requirements.txt`

Run examples:

- `python examples/quickstart_cpu.py`
- `python examples/compare_two_devices.py`

Run core scripts:

- `python scripts/compute_edcp.py --input results/sample_outputs/tables/device_comparison_sample.csv`
- `python scripts/compare_devices.py --input results/sample_outputs/tables/device_comparison_sample.csv`
- `python scripts/generate_reports.py --input results/sample_outputs/tables/device_comparison_sample.csv`

## Example outputs

Sample public-safe outputs are included under:

- `results/sample_outputs/tables/`
- `results/sample_outputs/plots/`
- `results/sample_outputs/reports/`

## Repository structure

```text
heterogeneous-edge-ml-benchmarks/
├── docs/
├── configs/
├── src/
├── scripts/
├── examples/
├── results/sample_outputs/
├── tests/
└── assets/
```

## Documentation map

- Methodology: `docs/methodology.md`
- Reproducibility: `docs/reproducibility.md`
- Hardware matrix: `docs/hardware_matrix.md`
- Runtime matrix: `docs/runtime_matrix.md`
- Metrics and EDCP: `docs/metrics.md`, `docs/edcp.md`
- Concurrency/workload allocation: `docs/concurrency_and_workload_allocation.md`
- Skill-focused narrative: `docs/skill_showcase.md`
- Paper-to-code traceability: `docs/paper_to_repo_map.md`

## Reproducibility

This repository includes a reproducibility guide, sample inputs/outputs, and deterministic output schema conventions for CSV/JSON report generation.

## Citation

See `CITATION.cff`.

## Author

Omkar B Shende  
Edge AI Systems Research
