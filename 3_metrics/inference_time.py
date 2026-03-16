"""
inference_time.py — Pure inference latency measurement (§5.2).

Measures ONLY the accelerator execution time (no I/O, no preprocessing).
This is the metric most commonly reported in inference benchmarks — but
the paper shows it is misleading in isolation (see total_time.py).

Provides:
  • run_inference_benchmark  — the core timing loop
  • warmup_and_measure       — standard warmup + measurement protocol
  • Statistical utilities: mean, p50, p90, p99, coefficient of variation
"""

import time
from typing import Callable, Dict, List, Optional

import numpy as np


def warmup_and_measure(
    infer_fn: Callable[[], None],
    warmup_iters: int = 20,
    measure_iters: int = 100,
    sleep_between_ms: float = 0.0,
) -> Dict[str, float]:
    """
    Standard benchmark protocol: warmup then measure.

    This matches the paper's §5.1 setup:
      - 20 warmup iterations (discarded, allow JIT / kernel cache warm-up)
      - 100 measurement iterations per run

    Args:
        infer_fn:        Zero-argument callable that executes one inference
        warmup_iters:    Number of discarded warmup runs
        measure_iters:   Number of timed measurement runs
        sleep_between_ms: Optional sleep between inferences (0 = no sleep)

    Returns:
        dict with latency statistics (ms) matching paper reporting format
    """
    # Warmup
    for _ in range(warmup_iters):
        infer_fn()

    # Measurement
    latencies_ms: List[float] = []
    for _ in range(measure_iters):
        t_start = time.perf_counter()
        infer_fn()
        t_end   = time.perf_counter()
        latencies_ms.append((t_end - t_start) * 1000.0)
        if sleep_between_ms > 0:
            time.sleep(sleep_between_ms / 1000.0)

    return compute_latency_stats(latencies_ms)


def compute_latency_stats(latencies_ms: List[float]) -> Dict[str, float]:
    """
    Compute summary statistics from a list of per-inference latencies.

    Returns:
        {
            "mean_ms", "median_ms", "p90_ms", "p99_ms",
            "std_ms", "cv", "min_ms", "max_ms",
            "fps",     # throughput from mean latency
            "count"
        }
    """
    if not latencies_ms:
        return {}

    arr = np.array(latencies_ms)
    mean_ms = float(np.mean(arr))

    return {
        "mean_ms":   round(mean_ms, 3),
        "median_ms": round(float(np.median(arr)), 3),
        "p90_ms":    round(float(np.percentile(arr, 90)), 3),
        "p99_ms":    round(float(np.percentile(arr, 99)), 3),
        "std_ms":    round(float(np.std(arr)), 3),
        "cv":        round(float(np.std(arr) / mean_ms), 4),  # coefficient of variation
        "min_ms":    round(float(np.min(arr)), 3),
        "max_ms":    round(float(np.max(arr)), 3),
        "fps":       round(1000.0 / mean_ms, 2),
        "count":     len(latencies_ms),
    }


def benchmark_runner(
    adapter,
    input_data: np.ndarray,
    warmup_iters: int = 20,
    measure_iters: int = 100,
    synchronize_cuda: bool = False,
) -> Dict[str, float]:
    """
    Benchmark an inference adapter as used in the paper.

    Args:
        adapter:           Object with .infer(np.ndarray) → np.ndarray method
        input_data:        Pre-cached input batch (already preprocessed)
        warmup_iters:      Paper uses 20
        measure_iters:     Paper uses 100
        synchronize_cuda:  If True, add CUDA sync before/after for GPU accuracy

    Returns:
        Latency statistics dict
    """
    # Warmup
    for _ in range(warmup_iters):
        _ = adapter.infer(input_data)

    # Optional CUDA synchronization for precise GPU timing
    if synchronize_cuda:
        try:
            import torch
            torch.cuda.synchronize()
        except ImportError:
            pass

    # Measurement
    latencies_ms = []
    for _ in range(measure_iters):
        if synchronize_cuda:
            try:
                import torch
                torch.cuda.synchronize()
            except ImportError:
                pass

        t0 = time.perf_counter()
        _ = adapter.infer(input_data)
        if synchronize_cuda:
            try:
                torch.cuda.synchronize()
            except ImportError:
                pass
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    return compute_latency_stats(latencies_ms)


# ---------------------------------------------------------------------------
# Paper normalised inference-time orderings (from Figure 2)
# Reproduced for validation and visualisation
# ---------------------------------------------------------------------------

PAPER_INFERENCE_ORDERING = {
    "CPU_FP32": {
        "desc": "Relative inference time (CPU FP32), baseline = M1_CPU",
        "ordering": [
            ("AGX_CPU",   3.94),
            ("H2_CPU",    2.08),
            ("NX_CPU",    1.98),
            ("TX2_CPU",   1.80),
            ("VIM3_Big",  1.10),
            ("M1_CPU",    1.00),
            ("Nano_CPU",  0.98),
        ]
    },
    "GPU_FP16": {
        "desc": "Relative inference time (GPU/Accelerator FP16), baseline = M1_NPU",
        "ordering": [
            ("AGX_GPU",   24.61),
            ("NX_GPU",    14.74),
            ("AGX_DLA",    8.45),
            ("NX_DLA",     7.62),
            ("TX2_GPU",    5.64),
            ("Nano_GPU",   1.89),
            ("H2_GPU",     1.74),
            ("NCS2_VPU",   1.64),
            ("M1_NPU",     1.00),
        ]
    },
    "INT8": {
        "desc": "Relative inference time (INT8), baseline = M1_NPU",
        "ordering": [
            ("AGX_GPU",   12.61),
            ("NX_GPU",     7.81),
            ("AGX_DLA",    3.89),
            ("NX_DLA",     3.69),
            ("VIM3_NPU",   1.29),
            ("M1_NPU",     1.00),
            ("M1_GPU",     0.089),
            ("VIM3_GPU",   0.057),
        ]
    },
}


def print_inference_ordering():
    """Print the normalized inference time ordering from Figure 2."""
    print("=== Inference Time Performance Ordering (Figure 2, Bang for the Buck) ===\n")
    for mode, data in PAPER_INFERENCE_ORDERING.items():
        print(f"  {data['desc']}:")
        for proc, rel in data["ordering"]:
            bar = "█" * max(1, int(rel * 2))
            print(f"    {proc:<15} {rel:7.2f}×  {bar}")
        print()


if __name__ == "__main__":
    print_inference_ordering()

    # Quick local demo with synthetic data
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    print("=== Synthetic benchmark demo (sleeping to simulate inference) ===")

    def fake_inference():
        time.sleep(0.010)  # 10ms fake inference

    stats = warmup_and_measure(fake_inference, warmup_iters=5, measure_iters=20)
    for k, v in stats.items():
        print(f"  {k}: {v}")
