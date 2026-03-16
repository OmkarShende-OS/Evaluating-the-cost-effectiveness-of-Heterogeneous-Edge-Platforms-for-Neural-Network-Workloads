# Concurrency and Workload Allocation

## Scope

The repository includes concurrency-focused evaluation inspired by data-wise and layer-wise allocation studies across heterogeneous co-processors.

## Included capability

- Config-driven workload split experiments.
- Comparative reporting for single-device vs concurrent allocation.
- Script entrypoint: `scripts/run_concurrency_experiments.py`.

## Practical interpretation

Not all concurrency splits produce speedups. Interconnect and transfer overhead can erase theoretical gains, especially in CPU + external accelerator pipelines.
