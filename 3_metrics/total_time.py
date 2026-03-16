"""
total_time.py — End-to-end latency measurement and CPU bottleneck analysis.

"Bang for the Buck" (SEC '23) §5.2 distinguishes two latency metrics:

  1. Inference Time  — time for the GPU/NPU/CPU execute operation alone
  2. Total Time      — full pipeline: load → preprocess → infer → postprocess

The paper's key finding (CPU Bottleneck):
  • Jetson AGX GPU shows 24.61× speedup over baseline in inference time
  • But only 11.64× speedup in total (end-to-end) time
  → The CPU handling I/O is the bottleneck, not the accelerator

This module implements:
  • TotalTimeProfiler — measure the full 4-stage pipeline
  • CPUBottleneckAnalyzer — quantify how much the driving CPU limits the accelerator
  • Stage breakdown: how much time each pipeline stage takes
"""

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Pipeline stage definitions
# ---------------------------------------------------------------------------

@dataclass
class PipelineStageTimings:
    """Timings for each stage of the NN inference pipeline."""
    load_ms:         float = 0.0   # Image loading from disk
    preprocess_ms:   float = 0.0   # Resize, normalise, type convert
    inference_ms:    float = 0.0   # GPU/NPU/CPU inference only
    postprocess_ms:  float = 0.0   # Softmax, NMS, top-k

    @property
    def total_ms(self) -> float:
        return (self.load_ms + self.preprocess_ms +
                self.inference_ms + self.postprocess_ms)

    @property
    def non_inference_ms(self) -> float:
        """Time spent outside the accelerator (CPU overhead)."""
        return self.load_ms + self.preprocess_ms + self.postprocess_ms

    @property
    def cpu_fraction(self) -> float:
        """Fraction of total time that runs on CPU (not on accelerator)."""
        return self.non_inference_ms / max(self.total_ms, 1e-9)

    def as_dict(self) -> Dict[str, float]:
        return {
            "load_ms":        round(self.load_ms, 3),
            "preprocess_ms":  round(self.preprocess_ms, 3),
            "inference_ms":   round(self.inference_ms, 3),
            "postprocess_ms": round(self.postprocess_ms, 3),
            "total_ms":       round(self.total_ms, 3),
            "cpu_fraction":   round(self.cpu_fraction, 4),
        }


class TotalTimeProfiler:
    """
    Profiles the full end-to-end inference pipeline, broken into 4 stages.

    Each stage is timed separately so that the CPU bottleneck effect can be
    quantified (as shown in Figure 3 of the paper).
    """

    def __init__(self, runner, data_loader):
        """
        Args:
            runner:      An inference runner with .infer(batch) method
            data_loader: Iterable of (raw_image_array, path) tuples
                         (not yet preprocessed)
        """
        self.runner = runner
        self.loader = data_loader
        self._measurements: List[PipelineStageTimings] = []

    def _load_image(self, path: str, target_h: int, target_w: int) -> np.ndarray:
        """Stage 1: Load raw image from disk."""
        from imagenet_loader import _resize_and_load
        return _resize_and_load(path, target_h, target_w)

    def _preprocess(self, raw: np.ndarray, preprocess_fn: Callable) -> np.ndarray:
        """Stage 2: Preprocess (normalise, cast, add batch dim)."""
        proc  = preprocess_fn(raw)
        return proc[np.newaxis, ...]   # (1, H, W, C)

    def _postprocess_classification(self, output: np.ndarray) -> int:
        """Stage 4: Compute top-1 class index from logits."""
        return int(np.argmax(output.flatten()))

    def profile(
        self,
        preprocess_fn: Callable,
        target_h: int = 224,
        target_w: int = 224,
        num_images: int = 100,
        warmup: int = 20,
    ) -> List[PipelineStageTimings]:
        """
        Run the full profiling pipeline.

        Args:
            preprocess_fn: Preprocessing function (from imagenet_loader)
            target_h/w:    Image target size
            num_images:    Number of measurement iterations
            warmup:        Number of iterations to discard

        Returns:
            List of PipelineStageTimings, one per image
        """
        self._measurements = []
        all_images = list(self.loader)

        # Warmup
        for batch, _ in all_images[:warmup]:
            _ = self.runner.infer(batch)

        for raw_batch, path in all_images[warmup:warmup + num_images]:
            timing = PipelineStageTimings()

            # Stage 1: Load
            t0 = time.perf_counter()
            raw = self._load_image(path, target_h, target_w)
            timing.load_ms = (time.perf_counter() - t0) * 1000.0

            # Stage 2: Preprocess
            t1 = time.perf_counter()
            batch = self._preprocess(raw, preprocess_fn)
            timing.preprocess_ms = (time.perf_counter() - t1) * 1000.0

            # Stage 3: Inference
            t2 = time.perf_counter()
            output = self.runner.infer(batch)
            timing.inference_ms = (time.perf_counter() - t2) * 1000.0

            # Stage 4: Postprocess
            t3 = time.perf_counter()
            _ = self._postprocess_classification(output)
            timing.postprocess_ms = (time.perf_counter() - t3) * 1000.0

            self._measurements.append(timing)

        return self._measurements

    def summary(self) -> Dict[str, float]:
        """Return mean values for each pipeline stage."""
        if not self._measurements:
            return {}
        fields = ["load_ms", "preprocess_ms", "inference_ms",
                  "postprocess_ms", "total_ms", "cpu_fraction"]
        result = {}
        for f in fields:
            vals = [getattr(m, f) for m in self._measurements]
            result[f"mean_{f}"] = round(float(np.mean(vals)), 3)
            if f in ("inference_ms", "total_ms"):
                result[f"p90_{f}"] = round(float(np.percentile(vals, 90)), 3)
        return result


# ---------------------------------------------------------------------------
# CPU Bottleneck Analyser
# ---------------------------------------------------------------------------

class CPUBottleneckAnalyzer:
    """
    Quantifies the CPU bottleneck effect observed in the paper.

    The paper shows:
      AGX GPU inference speedup  = 24.61×  (relative to M1 baseline)
      AGX GPU total time speedup =  11.64×  (relative to M1 baseline)
      Reduction factor           = 11.64 / 24.61 = 0.47

    This loss (53%!) is due to the AGX CPU handling I/O for the GPU.

    Additionally demonstrates the big.LITTLE effect on VIM3:
      VIM3 GPU total time (little core) = 1.12× slower than with big core
      VIM3 NPU total time (little core) = 1.30× slower than with big core
    """

    # From paper §5.2: (inference_time_speedup, total_time_speedup) per platform
    PAPER_SPEEDUPS = {
        # (inference_relative, total_time_relative)  — baseline M1
        "AGX_GPU_FP16":  (24.61, 11.64),
        "NX_GPU_FP16":   (14.74,  9.66),
        "AGX_DLA_INT8":  ( 8.45,  4.69),
        "NX_DLA_INT8":   ( 7.62,  4.03),
        "TX2_GPU_FP16":  ( 5.64,  4.62),
        "Nano_GPU_FP16": ( 1.89,  1.76),
        "H2_GPU_FP16":   ( 1.74,  1.60),
        "NCS2_VPU_FP16": ( 1.64,  1.56),
    }

    # VIM3 big vs little CPU driving the GPU/NPU (paper §5.2)
    VIM3_BIGLITTLE_EFFECT = {
        "VIM3_GPU":  {"big_total_ms": 100, "little_total_ms": 112, "ratio": 1.12},
        "VIM3_NPU":  {"big_total_ms": 100, "little_total_ms": 130, "ratio": 1.30},
    }

    @staticmethod
    def bottleneck_ratio(infer_speedup: float, total_speedup: float) -> float:
        """
        Compute how much of the accelerator speedup is lost to CPU bottleneck.

            bottleneck_ratio = total_speedup / infer_speedup

        A ratio of 1.0 means no bottleneck. Lower = more CPU-limited.

        Args:
            infer_speedup: Speedup relative to baseline, inference only
            total_speedup: Speedup relative to baseline, end-to-end

        Returns:
            Ratio in (0, 1] — lower means worse CPU bottleneck
        """
        return total_speedup / infer_speedup

    def print_bottleneck_analysis(self):
        """
        Print the CPU bottleneck analysis matching §5.2 of the paper.
        Shows how end-to-end speedup is always much less than inference speedup.
        """
        print("=" * 70)
        print("CPU Bottleneck Analysis — Bang for the Buck (SEC '23, §5.2)")
        print("=" * 70)
        print(f"\n  {'Processor':<22} {'Infer×':>8} {'Total×':>8} {'Ratio':>8} {'Lost%':>8}")
        print("  " + "-" * 60)
        for proc, (infer, total) in self.PAPER_SPEEDUPS.items():
            ratio = self.bottleneck_ratio(infer, total)
            lost  = (1.0 - ratio) * 100.0
            bar   = "⚠ " if lost > 40 else "  "
            print(f"  {bar}{proc:<20} {infer:>8.2f} {total:>8.2f} {ratio:>8.2f} {lost:>7.1f}%")
        print()
        print("  ⚠  AGX loses 53% of GPU speedup to CPU I/O bottleneck!")
        print()
        print("  VIM3 big.LITTLE effect:")
        for proc, data in self.VIM3_BIGLITTLE_EFFECT.items():
            print(f"    {proc}: little CPU is {data['ratio']}× slower in total time "
                  f"vs. big CPU driving the same accelerator")
        print()
        print("  Takeaway: End-to-end latency (Total Time) is the correct metric")
        print("  for application deployment, NOT inference-only time.")

    def compute_effective_speedup_loss(
        self,
        infer_speedup: float,
        total_speedup: float,
    ) -> Dict[str, float]:
        """
        Quantify effective speedup loss due to CPU bottleneck.

        Returns:
            dict with:
              - effective_ratio: total / inference speedup  (ideally 1.0)
              - lost_pct: percentage of speedup lost to CPU overhead
              - cpu_bound_fraction: estimated fraction of total time on CPU
        """
        ratio = self.bottleneck_ratio(infer_speedup, total_speedup)
        return {
            "effective_ratio": round(ratio, 3),
            "lost_pct": round((1.0 - ratio) * 100.0, 1),
            "interpretation": (
                "severe CPU bottleneck" if ratio < 0.5
                else "moderate CPU bottleneck" if ratio < 0.75
                else "acceptable CPU overhead"
            ),
        }


# ---------------------------------------------------------------------------
# Paper-published latency orderings (for validation/reproduction)
# ---------------------------------------------------------------------------

PAPER_INFERENCE_TIME_ORDERING = {
    "CPU_FP32": [
        ("AGX_CPU", 3.94), ("H2_CPU", 2.08), ("NX_CPU", 1.98), ("TX2_CPU", 1.80),
        ("VIM3_Big", 1.10), ("M1_CPU", 1.0),  ("Nano_CPU", 0.98),
    ],
    "GPU_FP16": [
        ("AGX_GPU", 24.61), ("NX_GPU", 14.74), ("AGX_DLA", 8.45), ("NX_DLA", 7.62),
        ("TX2_GPU", 5.64), ("Nano_GPU", 1.89), ("H2_GPU", 1.74), ("NCS2", 1.64),
        ("M1_NPU", 1.0),
    ],
    "INT8": [
        ("AGX_GPU", 12.61), ("NX_GPU", 7.81), ("AGX_DLA", 3.89), ("NX_DLA", 3.69),
        ("VIM3_NPU", 1.29), ("M1_NPU", 1.0), ("M1_GPU", 0.089), ("VIM3_GPU", 0.057),
    ],
}

PAPER_TOTAL_TIME_ORDERING = {
    "CPU_FP32": [
        ("AGX_CPU", 3.64), ("NX_CPU", 2.04), ("H2_CPU", 2.02), ("TX2_CPU", 1.80),
        ("Nano_CPU", 1.03), ("VIM3_Big", 1.01), ("M1_CPU", 1.0),
    ],
    "GPU_FP16": [
        ("AGX_GPU", 11.64), ("NX_GPU", 9.66), ("AGX_DLA", 4.69), ("TX2_GPU", 4.62),
        ("NX_DLA", 4.03), ("Nano_GPU", 1.76), ("H2_GPU", 1.60), ("NCS2", 1.56),
        ("M1_NPU", 1.0),
    ],
    "INT8": [
        ("AGX_GPU", 3.62), ("NX_GPU", 2.75), ("AGX_DLA", 2.38), ("NX_DLA", 2.08),
        ("VIM3_NPU", 1.78), ("M1_NPU", 1.0), ("M1_GPU", 0.13), ("VIM3_GPU", 0.08),
    ],
}


if __name__ == "__main__":
    analyzer = CPUBottleneckAnalyzer()
    analyzer.print_bottleneck_analysis()

    print("\n=== Total Time vs Inference Time deviation (example: AGX GPU FP16) ===")
    result = analyzer.compute_effective_speedup_loss(24.61, 11.64)
    for k, v in result.items():
        print(f"  {k}: {v}")
