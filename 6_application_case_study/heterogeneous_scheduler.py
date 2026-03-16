"""
heterogeneous_scheduler.py — Adaptive CPU→GPU/NPU inference router (§7).

This is the central dispatch component of the application case study.

Architecture:
  ┌──────────────┐      ┌──────────────────┐
  │  Video frame  │─────▶│ TrafficDensity   │
  │  (from cam)  │      │ Estimator (CPU)  │
  └──────────────┘      └────────┬─────────┘
                                  │
              ┌────────────────────┤
              │  LIGHT/MODERATE    │  HEAVY
              ▼                    ▼
  ┌──────────────────┐   ┌──────────────────┐
  │  NPU INT8        │   │  GPU FP16        │
  │  YOLOv3 fast     │   │  YOLOv3 accurate │
  │  AP=90.1%  38fps │   │  AP=96.3%  14fps │
  └──────────────────┘   └──────────────────┘
              │                    │
              └──────────┬─────────┘
                          ▼
                  Aggregated detections

Paper §7 result (Table 7):
  Adaptive scheduler AP  = 93.9%   (vs 96.3% GPU-only, vs 90.1% NPU-only)
  Adaptive scheduler FPS = 28.4    (vs 14.2  GPU-only, vs 38.7 NPU-only)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class InferenceRoute(Enum):
    GPU_FP16 = "gpu_fp16"
    NPU_INT8 = "npu_int8"
    SKIP     = "skip"


@dataclass
class SchedulerStats:
    """Aggregated statistics from an adaptive scheduling session."""
    n_frames:     int = 0
    n_gpu_frames: int = 0
    n_npu_frames: int = 0
    n_skip:       int = 0
    total_ms:     float = 0.0
    latencies_ms: List[float] = field(default_factory=list)

    @property
    def fps(self) -> float:
        return self.n_frames / (self.total_ms / 1000) if self.total_ms > 0 else 0.0

    @property
    def gpu_fraction(self) -> float:
        return self.n_gpu_frames / self.n_frames if self.n_frames > 0 else 0.0

    @property
    def mean_latency_ms(self) -> float:
        return float(np.mean(self.latencies_ms)) if self.latencies_ms else 0.0

    def report(self) -> str:
        lines = [
            "=== Adaptive Scheduler Stats ===",
            f"  Total frames:  {self.n_frames}",
            f"  GPU FP16 path: {self.n_gpu_frames} ({self.gpu_fraction*100:.0f}%)",
            f"  NPU INT8 path: {self.n_npu_frames} ({(1-self.gpu_fraction)*100:.0f}%)",
            f"  Skipped:       {self.n_skip}",
            f"  FPS:           {self.fps:.1f}",
            f"  Mean lat (ms): {self.mean_latency_ms:.1f}",
        ]
        return "\n".join(lines)


class HeterogeneousScheduler:
    """
    Adaptive inference scheduler routing frames to GPU or NPU based on traffic density.

    This implements the decision loop from §7:
      1. Run TrafficDensityEstimator on each frame (lightweight, CPU only)
      2. if density == HEAVY → dispatch to GPU FP16 detector
         if density < HEAVY  → dispatch to NPU INT8 detector
         if density == EMPTY → skip detection (no vehicles)

    The GPU and NPU detectors run as separate paths — no concurrent execution here
    (that's covered in Section 6). This is SEQUENTIAL, frame-by-frame routing.

    Usage:
        scheduler = HeterogeneousScheduler(
            gpu_detector=YOLOv3VehicleDetector("gpu_fp16", onnx_path="yolov3.onnx"),
            npu_detector=YOLOv3VehicleDetector("npu_int8", rknn_path="yolov3.rknn"),
        )
        scheduler.start()
        for frame in video:
            result = scheduler.process(frame)
        stats = scheduler.stop()
        print(stats.report())
    """

    def __init__(
        self,
        gpu_detector,              # YOLOv3VehicleDetector(backend="gpu_fp16")
        npu_detector,              # YOLOv3VehicleDetector(backend="npu_int8")
        density_estimator=None,    # TrafficDensityEstimator instance (created if None)
        heavy_threshold: int = 9,  # Contour count above which GPU is used
    ):
        self.gpu_detector    = gpu_detector
        self.npu_detector    = npu_detector
        self.heavy_threshold = heavy_threshold
        self._stats = SchedulerStats()

        if density_estimator is None:
            from traffic_density import TrafficDensityEstimator
            self._density_est = TrafficDensityEstimator()
        else:
            self._density_est = density_estimator

        self._t_session_start: Optional[float] = None

    def start(self) -> None:
        """Initialize both detectors and density estimator."""
        self.gpu_detector.load()
        self.npu_detector.load()
        self._density_est.reset()
        self._stats  = SchedulerStats()
        self._t_session_start = time.perf_counter()
        logger.info("Heterogeneous scheduler started")

    def process(self, frame: np.ndarray) -> dict:
        """
        Route and run inference on a single frame.

        Args:
            frame: BGR uint8 image

        Returns:
            dict with keys: route, density, boxes, inference_ms, frame_idx
        """
        t0 = time.perf_counter()
        self._stats.n_frames += 1
        frame_idx = self._stats.n_frames

        # Step 1: fast density classification (CPU only)
        from traffic_density import TrafficDensity
        density_est = self._density_est.estimate(frame, frame_idx=frame_idx)

        # Step 2: route
        if density_est.density == TrafficDensity.EMPTY:
            route  = InferenceRoute.SKIP
            result = None
            self._stats.n_skip += 1
        elif density_est.density == TrafficDensity.HEAVY or \
             density_est.n_contours > self.heavy_threshold:
            route  = InferenceRoute.GPU_FP16
            result = self.gpu_detector.detect(frame)
            self._stats.n_gpu_frames += 1
        else:
            route  = InferenceRoute.NPU_INT8
            result = self.npu_detector.detect(frame)
            self._stats.n_npu_frames += 1

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._stats.latencies_ms.append(elapsed_ms)

        return {
            "frame_idx":    frame_idx,
            "route":        route.value,
            "density":      density_est.density.name,
            "n_contours":   density_est.n_contours,
            "boxes":        result.boxes if result is not None else [],
            "inference_ms": result.inference_ms if result is not None else 0.0,
            "total_ms":     elapsed_ms,
        }

    def stop(self) -> SchedulerStats:
        """Finalize session and return statistics."""
        if self._t_session_start is not None:
            self._stats.total_ms = (time.perf_counter() - self._t_session_start) * 1000
        self.gpu_detector.close()
        self.npu_detector.close()
        return self._stats

    def validate_against_paper(self) -> dict:
        """
        Compare scheduler stats against paper §7 expected values.
        Returns validation report dict.
        """
        paper_fps = 28.4
        paper_gpu_frac = 0.30

        fps_ok      = abs(self._stats.fps - paper_fps) / paper_fps < 0.15
        gpu_frac_ok = abs(self._stats.gpu_fraction - paper_gpu_frac) < 0.10

        return {
            "measured_fps":         round(self._stats.fps, 1),
            "paper_fps":            paper_fps,
            "fps_ok":               fps_ok,
            "measured_gpu_frac":    round(self._stats.gpu_fraction, 2),
            "paper_gpu_frac":       paper_gpu_frac,
            "gpu_fraction_ok":      gpu_frac_ok,
            "overall_pass":         fps_ok and gpu_frac_ok,
        }


# ---------------------------------------------------------------------------
# Paper §7 scheduling policy table
# ---------------------------------------------------------------------------

PAPER_SCHEDULING_POLICY = {
    "description": (
        "Adaptive scheduling policy from §7 case study. "
        "CPU background subtraction runs on every frame in ~2ms."
    ),
    "EMPTY":    "SKIP (no inference, save power)",
    "LIGHT":    "NPU INT8 (1–3 vehicles, AP=90.1% acceptable)",
    "MODERATE": "NPU INT8 (4–8 vehicles, AP=90.1% acceptable)",
    "HEAVY":    "GPU FP16 (9+ vehicles, AP=96.3% needed for overlapping boxes)",
    "routing_time_ms": 1.8,   # CPU density estimation latency
    "gpu_activation_pct": 30,  # % of frames routed to GPU
}


if __name__ == "__main__":
    print("Heterogeneous Scheduler — Bang for the Buck §7")
    print()
    print("Scheduling policy:")
    for k, v in PAPER_SCHEDULING_POLICY.items():
        if isinstance(v, str) and k not in ("description",):
            print(f"  {k:<12} → {v}")
    print()
    print("Paper §7 results (Table 7):")
    print("  GPU only:   AP=96.3%  FPS=14.2")
    print("  NPU only:   AP=90.1%  FPS=38.7")
    print("  ADAPTIVE:   AP=93.9%  FPS=28.4  ← adaptive scheduler")
    print()
    print("Insight: 30% GPU frames preserve accuracy on dense scenes;")
    print("         70% NPU frames provide 2× throughput when not needed.")
