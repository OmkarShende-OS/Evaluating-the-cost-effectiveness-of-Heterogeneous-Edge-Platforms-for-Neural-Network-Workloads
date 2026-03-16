# Methodology

## Evaluation philosophy

Measure systems as deployed, not isolated kernels only. We use both inference-only and end-to-end timing to capture preprocessing/postprocessing/host bottlenecks.

## Experimental flow

1. Select model/device/runtime/precision.
2. Warmup for stable kernel/runtime behavior.
3. Run repeated measurements.
4. Collect latency, throughput, energy/power, and thermal metrics.
5. Compute EDCP and normalized ranking.
6. Generate tables and reports.

## Measurement conventions

- Warmup iterations are excluded from final statistics.
- Reported latency metrics include mean, median, p90, p99.
- Throughput reported as FPS from latency.
- Energy uses Joules per inference where available.
- Temperature is tracked as average and peak during run windows.

## Fairness notes

- Identical model and input shape per comparison group.
- Explicit runtime and precision annotations.
- Cost is declared and versioned via config files.
