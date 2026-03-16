"""
data_wise_allocation.py — Data-Wise Workload Allocation orchestrator (§6).

DWA splits a batch of input images between two processors running in parallel,
proportional to their solo inference speed. Both processors run the full
(identical) model — the data is divided, not the model.

Paper §6 key results:
  • Odroid H2 (Celeron J4125 CPU + Intel NCS2 VPU):
      DWA via OpenVINO MULTI plugin → 43% latency reduction (Table 5)
  • Khadas VIM3 (Big CPU cluster + Amlogic NPU):
      Manual DWA (ARMNN + KSNN split) → ~11% improvement only
      Reason: NPU is so fast that CPU is always the bottleneck

Allocation formula (§6.1):
  Let the solo speeds be T_A (ms) and T_B (ms).
  Optimal split ratio: r_A = T_B / (T_A + T_B)
  i.e., give more data to the faster processor.

  Normalised batch split for N frames:
    n_A = round(N × r_A)
    n_B = N − n_A
"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DWAConfig:
    """Configuration for data-wise allocation between two processors."""
    proc_a_name:  str         # e.g. "CPU", "GPU"
    proc_b_name:  str         # e.g. "NPU", "VPU"
    t_solo_a_ms:  float       # Solo inference time (ms) for one frame on proc A
    t_solo_b_ms:  float       # Solo inference time (ms) for one frame on proc B

    @property
    def split_ratio_a(self) -> float:
        """Fraction of data allocated to processor A."""
        total = self.t_solo_a_ms + self.t_solo_b_ms
        if total == 0:
            return 0.5
        # Faster processor gets proportionally MORE data
        # If B is faster, t_solo_b < t_solo_a → r_a = t_solo_b/(t_solo_a+t_solo_b) < 0.5
        return self.t_solo_b_ms / total

    @property
    def split_ratio_b(self) -> float:
        return 1.0 - self.split_ratio_a

    @property
    def theoretical_speedup(self) -> float:
        """
        Theoretical speedup from DWA.
        Eq(1) from paper: speedup = (T_A + T_B) / max(T_A * r_A, T_B * r_B)
        """
        t_a_split = self.t_solo_a_ms * self.split_ratio_a
        t_b_split = self.t_solo_b_ms * self.split_ratio_b
        # Ideally both finish at the same time → speedup = T_fast / T_parallel
        t_parallel = max(t_a_split, t_b_split)
        t_faster   = min(self.t_solo_a_ms, self.t_solo_b_ms)
        if t_parallel == 0:
            return 1.0
        return t_faster / t_parallel


def compute_optimal_split(
    n_frames: int,
    t_solo_a_ms: float,
    t_solo_b_ms: float,
) -> Tuple[int, int]:
    """
    Compute optimal frame counts for processors A and B.

    Args:
        n_frames:     Total number of frames to process
        t_solo_a_ms:  Solo latency for one frame on proc A (ms)
        t_solo_b_ms:  Solo latency for one frame on proc B (ms)

    Returns:
        (n_a, n_b) where n_a + n_b == n_frames
    """
    if t_solo_a_ms <= 0 or t_solo_b_ms <= 0:
        raise ValueError("Solo latencies must be positive")

    total = t_solo_a_ms + t_solo_b_ms
    r_b   = t_solo_a_ms / total   # faster → more data (lower solo time = more allocation)
    n_b   = round(n_frames * r_b)
    n_a   = n_frames - n_b
    return int(n_a), int(n_b)


class DWARunner:
    """
    Parallel DWA runner — splits a batch between two processors in parallel threads.

    Both processor functions run concurrently. DWA is equivalent to the
    OpenVINO MULTI plugin on Odroid H2 but also works manually for other platforms.

    Usage:
        def cpu_infer(batch): return model_cpu.infer(batch)
        def npu_infer(batch): return model_npu.infer(batch)

        runner = DWARunner(cpu_infer, npu_infer, t_solo_cpu=28.4, t_solo_npu=9.8)
        outputs = runner.run(input_batch_of_100_frames)
    """

    def __init__(
        self,
        proc_a_fn: Callable[[np.ndarray], np.ndarray],
        proc_b_fn: Callable[[np.ndarray], np.ndarray],
        t_solo_a_ms: float,
        t_solo_b_ms: float,
        proc_a_name: str = "ProcA",
        proc_b_name: str = "ProcB",
    ):
        self.proc_a_fn   = proc_a_fn
        self.proc_b_fn   = proc_b_fn
        self.config      = DWAConfig(proc_a_name, proc_b_name, t_solo_a_ms, t_solo_b_ms)

    def run(self, batch: np.ndarray) -> np.ndarray:
        """
        Run DWA on a batch. Returns combined output in original order.

        Args:
            batch: Array of shape (N, H, W, C) or (N, C, H, W)

        Returns:
            Combined inference outputs (N, num_classes) or similar
        """
        n = len(batch)
        n_a, n_b = compute_optimal_split(
            n, self.config.t_solo_a_ms, self.config.t_solo_b_ms
        )
        logger.debug(
            "DWA split: %s=%d frames, %s=%d frames (ratio=%.2f/%.2f)",
            self.config.proc_a_name, n_a,
            self.config.proc_b_name, n_b,
            self.config.split_ratio_a, self.config.split_ratio_b,
        )

        batch_a = batch[:n_a]
        batch_b = batch[n_a:]

        result_a: Optional[np.ndarray] = None
        result_b: Optional[np.ndarray] = None
        exc_a = exc_b = None

        def run_a():
            nonlocal result_a, exc_a
            try:
                result_a = self.proc_a_fn(batch_a)
            except Exception as e:
                exc_a = e

        def run_b():
            nonlocal result_b, exc_b
            try:
                result_b = self.proc_b_fn(batch_b)
            except Exception as e:
                exc_b = e

        t0 = time.perf_counter()
        thread_a = threading.Thread(target=run_a, daemon=True)
        thread_b = threading.Thread(target=run_b, daemon=True)
        thread_a.start(); thread_b.start()
        thread_a.join();  thread_b.join()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if exc_a:
            raise RuntimeError(f"{self.config.proc_a_name} failed: {exc_a}") from exc_a
        if exc_b:
            raise RuntimeError(f"{self.config.proc_b_name} failed: {exc_b}") from exc_b

        logger.debug("DWA wall time: %.1f ms", elapsed_ms)

        # Concatenate in original order (A first, B second)
        if result_a is None:
            return result_b
        if result_b is None:
            return result_a
        return np.concatenate([result_a, result_b], axis=0)

    def benchmark(self, batch: np.ndarray, n_runs: int = 50) -> dict:
        """Measure DWA latency vs solo latency."""
        latencies = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            self.run(batch)
            latencies.append((time.perf_counter() - t0) * 1000)

        mean_dwa = float(np.mean(latencies))
        solo_a   = self.config.t_solo_a_ms * len(batch)
        solo_b   = self.config.t_solo_b_ms * len(batch)
        solo_best = min(solo_a, solo_b)

        return {
            "dwa_mean_ms":         round(mean_dwa, 2),
            "solo_a_estimated_ms": round(solo_a, 2),
            "solo_b_estimated_ms": round(solo_b, 2),
            "speedup_vs_a":        round(solo_a / mean_dwa, 3),
            "speedup_vs_b":        round(solo_b / mean_dwa, 3),
            "speedup_vs_best_solo":round(solo_best / mean_dwa, 3),
            "theoretical_speedup": round(self.config.theoretical_speedup, 3),
            "split_a":             self.config.split_ratio_a,
            "split_b":             self.config.split_ratio_b,
        }


# ---------------------------------------------------------------------------
# Paper §6 DWA results (Table 5)
# ---------------------------------------------------------------------------

PAPER_DWA_RESULTS = {
    "Odroid_H2_CPU_NCS2": {
        "description": (
            "Odroid H2: Celeron J4125 CPU + NCS2 VPU DWA via OpenVINO MULTI. "
            "Best DWA case in paper — 43% latency reduction."
        ),
        "MobileNetV2": {
            "solo_CPU_ms": 28.4,  "solo_NCS2_ms": 9.8,
            "DWA_ms": 17.0,       "speedup_vs_CPU": 1.67, "speedup_vs_NCS2": 0.58,
            "latency_reduction_pct": 43,
        },
        "ResNet101V2": {
            "solo_CPU_ms": 210.0, "solo_NCS2_ms": 74.0,
            "DWA_ms": 126.0,      "speedup_vs_CPU": 1.67, "speedup_vs_NCS2": 0.59,
            "latency_reduction_pct": 40,
        },
    },
    "Khadas_VIM3_Big_KSNN": {
        "description": (
            "VIM3: Big CPU cluster (Cortex-A73) + Amlogic NPU (KSNN). "
            "Modest improvement — NPU is bottleneck, CPU is always waiting."
        ),
        "MobileNetV2": {
            "solo_CPU_ms": 38.2,  "solo_KSNN_ms": 5.4,
            "DWA_ms": 34.8,       "speedup_vs_CPU": 1.10, "speedup_vs_KSNN": 0.16,
            "latency_reduction_pct": 11,
        },
        "note": (
            "KSNN NPU is 7× faster than VIM3 CPU. "
            "CPU only processes 13% of frames — very few frames per CPU thread. "
            "No meaningful parallelism improvement."
        ),
    },
}


if __name__ == "__main__":
    print("Data-Wise Allocation (DWA) — Bang for the Buck §6")
    print()
    print("Key insight: DWA is only effective when both processors have similar speed")
    print()
    for platform, data in PAPER_DWA_RESULTS.items():
        if "description" in data:
            print(f"Platform: {platform}")
            print(f"  {data['description']}")
            for model, vals in data.items():
                if isinstance(vals, dict) and "DWA_ms" in vals:
                    print(f"  {model}: solo_CPU={vals['solo_CPU_ms']:.0f}ms  "
                          f"solo_FAST={list(vals.values())[1]:.0f}ms  "
                          f"DWA={vals['DWA_ms']:.0f}ms  "
                          f"reduction={vals['latency_reduction_pct']}%")
            print()

    # Demonstrate compute_optimal_split
    print("DWA split example — Odroid H2 MobileNetV2 (100 frames):")
    n_a, n_b = compute_optimal_split(100, t_solo_a_ms=28.4, t_solo_b_ms=9.8)
    print(f"  CPU frames: {n_a}  NCS2 frames: {n_b}  (ratio {n_a}:{n_b})")
    cfg = DWAConfig("CPU", "NCS2", 28.4, 9.8)
    print(f"  Theoretical speedup: {cfg.theoretical_speedup:.2f}×  (paper: 1.67×)")
