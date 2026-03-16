"""
traffic_density.py — CPU-based traffic density estimation (§7 Application Case Study).

The paper's Section 7 presents a real-world application:
  Heterogeneous edge inference for traffic monitoring using YOLOv3.

  The main insight is the adaptive scheduler:
    - CPU estimates TRAFFIC DENSITY (light/moderate/heavy) via background subtraction
    - LOW density  → send to NPU (INT8) — faster, lower power, acceptable accuracy
    - HIGH density → send to GPU (FP16) — higher AP, needed when many vehicles overlap

This module implements the CPU traffic density classifier.
It requires only OpenCV (no NN accelerator) — lightweight monitoring loop.

Paper §7 result (Table 7):
  GPU FP16:    AP = 96.3%  (dense traffic — maximum accuracy needed)
  NPU INT8:    AP = 90.1%  (sparse traffic — throughput prioritized)
  Adaptive:    AP = 93.9%  FPS = 28.4  (best accuracy-throughput trade-off)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.debug("OpenCV not available — traffic density estimation disabled")


class TrafficDensity(Enum):
    EMPTY     = 0  # No vehicles — skip inference
    LIGHT     = 1  # 1–3 vehicles — NPU path
    MODERATE  = 2  # 4–8 vehicles — NPU path
    HEAVY     = 3  # 9+ vehicles — GPU path (high accuracy needed)


@dataclass
class DensityEstimate:
    """Output of one density estimation pass."""
    density:       TrafficDensity
    motionpx_ratio: float     # Fraction of pixels with significant motion
    n_contours:    int        # Number of detected vehicle-sized blobs
    frame_idx:     int        # Frame index in the video stream


class TrafficDensityEstimator:
    """
    Lightweight traffic density classifier using background subtraction.

    Runs on CPU only, used to gate which inference backend is used:
      HEAVY → heterogeneous_scheduler routes to GPU (FP16, high AP)
      LIGHT/MODERATE → scheduler routes to NPU (INT8, high throughput)

    Algorithm:
      1. MOG2 background subtractor extracts foreground mask
      2. Morphological open/close to remove noise
      3. Find contours and filter by area (vehicle-sized blobs)
      4. Classify density by contour count and motion ratio

    Usage:
        estimator = TrafficDensityEstimator(history=50, var_threshold=50)
        estimator.reset()
        for frame in video_frames:
            density_est = estimator.estimate(frame, frame_idx=i)
            print(density_est.density)
    """

    # Vehicle-sized blob area thresholds (pixels²) — tuned for 640×480 input
    MIN_BLOB_AREA = 800
    MAX_BLOB_AREA = 50_000

    # Density thresholds (number of valid contours)
    LIGHT_THRESHOLD    = 3
    MODERATE_THRESHOLD = 8

    def __init__(
        self,
        history: int = 50,
        var_threshold: float = 50.0,
        detect_shadows: bool = False,
    ):
        """
        Args:
            history:         MOG2 history length (frames)
            var_threshold:   MOG2 variance threshold — higher = less sensitive
            detect_shadows:  Include shadow detection (slower, not needed here)
        """
        self.history       = history
        self.var_threshold = var_threshold
        self.detect_shadows = detect_shadows

        self._subtractor = None
        self._kernel     = None

    def reset(self) -> None:
        """Initialize/reset background model."""
        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV (cv2) is required for traffic density estimation")

        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.history,
            varThreshold=self.var_threshold,
            detectShadows=self.detect_shadows,
        )
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        logger.debug("Traffic density estimator reset (history=%d)", self.history)

    def estimate(self, frame: np.ndarray, frame_idx: int = 0) -> DensityEstimate:
        """
        Estimate traffic density from a single video frame.

        Args:
            frame:     BGR frame as uint8 NumPy array (H, W, 3)
            frame_idx: Frame number (used for tracking)

        Returns:
            DensityEstimate with density class and motion statistics
        """
        if self._subtractor is None:
            self.reset()

        # Background subtraction
        fg_mask = self._subtractor.apply(frame)

        # Clean up noise with morphological operations
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  self._kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self._kernel)

        # Binarize (threshold out shadows if any)
        _, fg_binary = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)

        # Motion pixel ratio
        motion_ratio = float(np.count_nonzero(fg_binary)) / fg_binary.size

        # Find contours
        contours, _ = cv2.findContours(fg_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter by area
        valid = [c for c in contours
                 if self.MIN_BLOB_AREA <= cv2.contourArea(c) <= self.MAX_BLOB_AREA]
        n_valid = len(valid)

        # Classify
        if n_valid == 0:
            density = TrafficDensity.EMPTY
        elif n_valid <= self.LIGHT_THRESHOLD:
            density = TrafficDensity.LIGHT
        elif n_valid <= self.MODERATE_THRESHOLD:
            density = TrafficDensity.MODERATE
        else:
            density = TrafficDensity.HEAVY

        return DensityEstimate(
            density         = density,
            motionpx_ratio  = round(motion_ratio, 4),
            n_contours      = n_valid,
            frame_idx       = frame_idx,
        )

    def classify_batch(self, frames: np.ndarray) -> list:
        """Estimate density for every frame in a batch."""
        return [self.estimate(frames[i], i) for i in range(len(frames))]


# ---------------------------------------------------------------------------
# Synthetic test for environments without a camera
# ---------------------------------------------------------------------------

def generate_synthetic_traffic_frame(
    density: str = "heavy",
    frame_size: tuple = (480, 640, 3),
    n_vehicles: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a synthetic traffic frame with random rectangular "vehicles".

    Args:
        density:     "empty", "light", "moderate", "heavy"
        frame_size:  (H, W, C) shape
        n_vehicles:  Override number of vehicles

    Returns:
        BGR uint8 synthetic frame
    """
    rng = np.random.default_rng(42)
    frame = np.zeros(frame_size, dtype=np.uint8)

    # Road texture: grey asphalt
    frame[:] = rng.integers(60, 80, frame_size, dtype=np.uint8)

    n_map = {"empty": 0, "light": 2, "moderate": 6, "heavy": 12}
    n = n_vehicles if n_vehicles is not None else n_map.get(density, 6)

    h, w = frame_size[:2]
    for _ in range(n):
        vw = rng.integers(30, 80)
        vh = rng.integers(20, 50)
        x  = int(rng.integers(0, max(1, w - vw)))
        y  = int(rng.integers(0, max(1, h - vh)))
        color = tuple(int(c) for c in rng.integers(100, 255, 3).tolist())
        frame[y:y+vh, x:x+vw] = color

    return frame


# ---------------------------------------------------------------------------
# Paper §7 density distribution data
# ---------------------------------------------------------------------------

PAPER_TRAFFIC_DENSITY_DISTRIBUTION = {
    "description": (
        "Empirical traffic density distribution over 10 minutes of highway footage "
        "used in §7 case study. GPU path activated for ~30% of frames."
    ),
    "EMPTY":    {"pct": 5,  "npu_route": True},
    "LIGHT":    {"pct": 35, "npu_route": True},
    "MODERATE": {"pct": 30, "npu_route": True},
    "HEAVY":    {"pct": 30, "npu_route": False},   # → GPU
}


if __name__ == "__main__":
    print("Traffic Density Estimator — Bang for the Buck §7")
    print()
    print("Traffic distribution in paper evaluation:")
    for cat, d in PAPER_TRAFFIC_DENSITY_DISTRIBUTION.items():
        if "pct" in d:
            route = "NPU (INT8)" if d["npu_route"] else "GPU (FP16)"
            bar = "█" * (d["pct"] // 2)
            print(f"  {cat:<10} {d['pct']:>3}%  → {route}  {bar}")
    print()
    print("Paper result:")
    print("  GPU only:   AP=96.3%  FPS=14.2")
    print("  NPU only:   AP=90.1%  FPS=38.7")
    print("  Adaptive:   AP=93.9%  FPS=28.4  ← best trade-off")

    if CV2_AVAILABLE:
        print()
        print("Running synthetic density estimation demo...")
        est = TrafficDensityEstimator()
        frame = generate_synthetic_traffic_frame("heavy")
        result = est.estimate(frame)
        print(f"  Synthetic heavy frame: density={result.density.name}  "
              f"n_contours={result.n_contours}  motion={result.motionpx_ratio:.3f}")
    else:
        print("\n(OpenCV not available — install with: pip install opencv-python)")
