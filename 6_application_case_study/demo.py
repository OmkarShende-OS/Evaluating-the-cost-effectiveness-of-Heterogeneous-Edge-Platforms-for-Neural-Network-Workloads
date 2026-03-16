"""
demo.py — End-to-end application case study demo (§7).

Reproduces the traffic monitoring scenario from Section 7 of the paper.

Modes:
  1. demo_with_synthetic — generates synthetic traffic frames, runs adaptive scheduler
  2. demo_with_video     — processes a real video file (path required)
  3. demo_paper_results  — prints Table 7 results without running inference

The demo shows the key contribution of §7:
  Adaptive heterogeneous scheduling achieves a better accuracy-throughput
  trade-off than either GPU-only or NPU-only inference.

Requirements:
  Full demo: OpenCV, plus appropriate hardware (Jetson for GPU, M1/VIM3 for NPU)
  Paper results print: no hardware required
"""

import logging
import sys
import time
import os
from typing import List, Optional

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger(__name__)


def demo_paper_results() -> None:
    """Print Table 7 results from the paper without running any inference."""
    from vehicle_detector import PAPER_YOLOV3_ACCURACY
    from heterogeneous_scheduler import PAPER_SCHEDULING_POLICY

    print("=" * 60)
    print("  Bang for the Buck — §7 Application Case Study")
    print("  Traffic Vehicle Detection — Adaptive Scheduler")
    print("=" * 60)
    print()
    print("Platform: Odroid M1 (Cortex-A55 CPU + RK3568 NPU) + Odroid H2 GPU")
    print()
    print("Table 7: Inference Strategy Comparison")
    print(f"  {'Strategy':<12} {'AP %':>7} {'FPS':>7} {'Lat (ms)':>10}  Notes")
    print(f"  {'─'*12} {'─'*7} {'─'*7} {'─'*10}  {'─'*30}")
    for mode, d in PAPER_YOLOV3_ACCURACY.items():
        if "AP_pct" in d:
            print(f"  {mode:<12} {d['AP_pct']:>7.1f} {d['FPS']:>7.1f} {d['avg_latency_ms']:>10.1f}  {d['note']}")
    print()
    print("Scheduling policy:")
    for k, v in PAPER_SCHEDULING_POLICY.items():
        if isinstance(v, str) and k != "description":
            print(f"  {k:<12} → {v}")
    print()
    print(f"CPU density estimation latency: {PAPER_SCHEDULING_POLICY['routing_time_ms']} ms/frame")
    print(f"GPU path activated for {PAPER_SCHEDULING_POLICY['gpu_activation_pct']}% of frames")


def demo_with_synthetic(
    n_frames: int = 200,
    dist: Optional[dict] = None,
) -> None:
    """
    Run adaptive scheduler on synthetic traffic frames.

    Args:
        n_frames: Total number of frames to simulate
        dist:     Traffic density distribution (fraction of each class).
                  Default: matches paper §7 test set distribution.
    """
    from traffic_density import (
        TrafficDensityEstimator,
        TrafficDensity,
        generate_synthetic_traffic_frame,
        PAPER_TRAFFIC_DENSITY_DISTRIBUTION,
    )
    from vehicle_detector import YOLOv3VehicleDetector
    from heterogeneous_scheduler import HeterogeneousScheduler, SchedulerStats

    if dist is None:
        dist = {
            "empty":    0.05,
            "light":    0.35,
            "moderate": 0.30,
            "heavy":    0.30,
        }

    print("=" * 60)
    print("  Synthetic traffic demo — adaptive scheduler")
    print(f"  {n_frames} frames, density dist: {dist}")
    print("=" * 60)

    # Generate frames matching distribution
    rng = np.random.default_rng(42)
    densities = rng.choice(
        list(dist.keys()),
        size=n_frames,
        p=list(dist.values()),
    )
    frames = [generate_synthetic_traffic_frame(d, frame_size=(480, 640, 3)) for d in densities]
    frames = np.stack(frames)
    print(f"Generated {n_frames} synthetic frames.")

    # Create stub detectors (no hardware required)
    class _StubGPUDetector:
        def load(self): logger.debug("Stub GPU detector loaded")
        def detect(self, frame):
            from vehicle_detector import DetectionResult
            time.sleep(0.070)  # Simulate 70ms GPU inference
            return DetectionResult(boxes=[], inference_ms=70.0, backend="gpu_fp16")
        def close(self): pass

    class _StubNPUDetector:
        def load(self): logger.debug("Stub NPU detector loaded")
        def detect(self, frame):
            from vehicle_detector import DetectionResult
            time.sleep(0.026)  # Simulate 26ms NPU inference
            return DetectionResult(boxes=[], inference_ms=26.0, backend="npu_int8")
        def close(self): pass

    scheduler = HeterogeneousScheduler(
        gpu_detector=_StubGPUDetector(),
        npu_detector=_StubNPUDetector(),
    )
    scheduler.start()

    print("Processing frames...")
    route_counts = {"gpu_fp16": 0, "npu_int8": 0, "skip": 0}
    for i, frame in enumerate(frames):
        result = scheduler.process(frame)
        route_counts[result["route"]] += 1
        if (i + 1) % 50 == 0:
            print(f"  Frame {i+1}/{n_frames}  density={result['density']}  "
                  f"route={result['route']}  lat={result['total_ms']:.1f}ms")

    stats = scheduler.stop()
    print()
    print(stats.report())
    print()
    print(f"Route distribution: GPU={route_counts['gpu_fp16']}  "
          f"NPU={route_counts['npu_int8']}  SKIP={route_counts['skip']}")
    print()
    print("Paper expected (Table 7):  FPS≈28.4  GPU_frac≈30%")


def demo_with_video(video_path: str, max_frames: int = 500) -> None:
    """
    Run adaptive scheduler on a real video file.

    Args:
        video_path: Path to video file (MP4, AVI, etc.)
        max_frames: Maximum frames to process
    """
    try:
        import cv2
    except ImportError:
        print("OpenCV is required for video demo: pip install opencv-python")
        return

    from traffic_density import TrafficDensityEstimator
    from vehicle_detector import YOLOv3VehicleDetector
    from heterogeneous_scheduler import HeterogeneousScheduler

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return

    fps_native = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {video_path}  |  FPS: {fps_native:.1f}  |  Frames: {total_frames}")

    # Build scheduler with real or stub detectors based on what's available
    try:
        gpu_det = YOLOv3VehicleDetector("gpu_fp16")
        npu_det = YOLOv3VehicleDetector("npu_int8")
        scheduler = HeterogeneousScheduler(gpu_det, npu_det)
        scheduler.start()
    except Exception as e:
        print(f"Could not load real detectors ({e}), demo requires hardware")
        cap.release()
        return

    results = []
    frame_count = 0
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        result = scheduler.process(frame)
        results.append(result)
        frame_count += 1

    cap.release()
    stats = scheduler.stop()

    print()
    print(stats.report())
    validation = scheduler.validate_against_paper()
    print()
    print(f"Paper validation: FPS {validation['measured_fps']} vs expected {validation['paper_fps']}  "
          f"{'✓' if validation['fps_ok'] else '✗'}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    import argparse
    parser = argparse.ArgumentParser(
        description="Bang for the Buck §7 — Adaptive Traffic Detection Demo"
    )
    parser.add_argument("--mode", choices=["paper", "synthetic", "video"],
                        default="paper", help="Demo mode")
    parser.add_argument("--video", type=str, help="Video file path (for --mode=video)")
    parser.add_argument("--frames", type=int, default=200, help="Number of frames (synthetic)")
    args = parser.parse_args()

    if args.mode == "paper":
        demo_paper_results()
    elif args.mode == "synthetic":
        demo_with_synthetic(n_frames=args.frames)
    elif args.mode == "video":
        if not args.video:
            print("--video path required for video mode")
            sys.exit(1)
        demo_with_video(args.video, max_frames=args.frames)
