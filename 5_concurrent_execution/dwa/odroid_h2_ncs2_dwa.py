"""
odroid_h2_ncs2_dwa.py — Odroid H2 (Celeron + NCS2) DWA implementation (§6).

This module implements the best-performing DWA configuration from the paper:
Odroid H2 Celeron J4125 CPU + Intel NCS2 VPU using the OpenVINO MULTI plugin.

Two approaches:
  1. OpenVINO MULTI plugin — single model loaded into MULTI:CPU,MYRIAD
     OpenVINO automatically dispatches requests to keep both devices busy.
  2. Manual threading — two separate OpenVINO runtimes, one per device,
     with frame splitting handled by data_wise_allocation.DWARunner.

Paper finding (§6, Table 5):
  MULTI plugin approach achieves 43% latency reduction vs solo NCS2 baseline
  for MobileNetV2 (100 frames).

  The improvement comes from:
    - CPU handling ~25% of frames while NCS2 handles ~75%
    - Both devices run simultaneously — wall time ≈ max(CPU_time, NCS2_time)
    - NCS2 becomes the bottleneck but CPU contribution reduces total time.
"""

import logging
import time
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class OdroidH2NCS2DWA:
    """
    Odroid H2 + NCS2 Data-Wise Allocation.

    Method: OpenVINO MULTI plugin (single compiled model for both devices).
    The MULTI plugin handles internal load balancing between CPU and MYRIAD.

    This is more efficient than manual threading because:
    - No Python GIL contention
    - OpenVINO's async queue keeps both devices saturated
    - Works transparently with OpenVINO API 2.0 asynchronous inference

    Usage:
        dwa = OdroidH2NCS2DWA(model_xml="mobilenet_fp16.xml", n_infer_requests=4)
        dwa.load()
        results = dwa.run_batch(frames_100)   # List of 100 preprocessed images
        print(dwa.benchmark(frames_100))
    """

    DEVICES = "MULTI:CPU,MYRIAD"

    def __init__(
        self,
        model_xml: str,
        n_infer_requests: int = 4,
        cpu_threads: int = 4,
    ):
        """
        Args:
            model_xml:         OpenVINO IR .xml model (FP16, required for MYRIAD)
            n_infer_requests:  Number of async inference requests in the queue
            cpu_threads:       CPU thread count for the CPU backend
        """
        self.model_xml        = model_xml
        self.n_infer_requests = n_infer_requests
        self.cpu_threads      = cpu_threads

        self._core     = None
        self._compiled = None
        self._requests: List = []

    def load(self) -> None:
        """Compile model for MULTI:CPU,MYRIAD."""
        try:
            from openvino.runtime import Core
        except ImportError:
            raise RuntimeError("OpenVINO not installed. pip install openvino-dev")

        self._core = Core()

        config = {
            "MULTI_DEVICE_PRIORITIES": "MYRIAD,CPU",  # Prefer NCS2, fall back to CPU
            "CPU_THREADS_NUM": str(self.cpu_threads),
            "MYRIAD_ENABLE_HW_ACCELERATION": "YES",
        }

        model = self._core.read_model(model=self.model_xml)
        self._compiled = self._core.compile_model(
            model=model,
            device_name=self.DEVICES,
            config=config,
        )

        # Pre-create inference requests for async pipelining
        self._requests = [
            self._compiled.create_infer_request()
            for _ in range(self.n_infer_requests)
        ]

        input_node = self._compiled.inputs[0]
        self._input_name  = input_node.get_any_name()
        self._output_name = self._compiled.outputs[0].get_any_name()
        self._input_shape = tuple(input_node.shape)
        logger.info("H2 DWA loaded: %s → %s (%d async requests)",
                    self.model_xml, self.DEVICES, self.n_infer_requests)

    def run_batch(self, frames: np.ndarray) -> np.ndarray:
        """
        Run async batch inference distributing frames across CPU and NCS2.

        OpenVINO MULTI plugin handles the distribution internally.
        We pipeline multiple inference requests to keep both devices busy.

        Args:
            frames: (N, H, W, C) or (N, C, H, W) float32 array

        Returns:
            (N, num_classes) output array
        """
        if self._compiled is None:
            raise RuntimeError("Call load() first")

        n = len(frames)
        outputs = [None] * n
        req_idx = 0

        for i, frame in enumerate(frames):
            req = self._requests[req_idx % self.n_infer_requests]
            req.set_tensor(self._input_name,
                           frame.reshape(self._input_shape).astype(np.float16))
            req.start_async()
            req_idx += 1

            # Drain completed requests every N_REQUESTS to avoid queue overflow
            if req_idx % self.n_infer_requests == 0:
                for j in range(max(0, i - self.n_infer_requests + 1), i + 1):
                    self._requests[j % self.n_infer_requests].wait()
                    outputs[j] = self._requests[j % self.n_infer_requests].get_tensor(
                        self._output_name
                    ).data.copy()

        # Drain remaining requests
        for remaining in range(req_idx - (req_idx % self.n_infer_requests), n):
            self._requests[remaining % self.n_infer_requests].wait()
            outputs[remaining] = self._requests[remaining % self.n_infer_requests].get_tensor(
                self._output_name
            ).data.copy()

        # Filter None entries (shouldn't happen)
        outputs = [o for o in outputs if o is not None]
        return np.stack(outputs, axis=0) if outputs else np.array([])

    def benchmark(
        self,
        frames: np.ndarray,
        n_runs: int = 50,
        warmup: int = 10,
    ) -> Dict[str, float]:
        """
        Measure DWA throughput and compare with paper results.

        Returns timing metrics dict.
        """
        # Warmup
        for _ in range(warmup):
            self.run_batch(frames[:1])

        latencies_ms = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            self.run_batch(frames)
            latencies_ms.append((time.perf_counter() - t0) * 1000)

        mean_ms  = float(np.mean(latencies_ms))
        n_frames = len(frames)

        return {
            "platform":    "Odroid_H2",
            "method":      "OpenVINO_MULTI_CPU_NCS2",
            "n_frames":    n_frames,
            "mean_ms":     round(mean_ms, 2),
            "fps":         round(n_frames / (mean_ms / 1000), 2),
            "per_frame_ms":round(mean_ms / n_frames, 3),
        }

    def close(self) -> None:
        self._requests.clear()
        self._compiled = None
        self._core     = None


# ---------------------------------------------------------------------------
# Manual DWA (alternative to MULTI plugin) using two separate runtimes
# ---------------------------------------------------------------------------

class OdroidH2ManualDWA:
    """
    Manual DWA using two separate OpenVINO runtimes + threading.

    Use this when you need explicit control over the split ratio or when
    the MULTI plugin is not available. Less efficient than MULTI plugin.
    """

    def __init__(self, model_xml: str, n_frames: int = 100):
        self.model_xml = model_xml
        self.n_frames  = n_frames
        self._cpu_runner = None
        self._ncs2_runner = None

    def load(self) -> None:
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
        from bang_for_the_buck.four_sw_framework_comparison.runners.openvino_runner import (
            OpenVINORunner,
        )
        self._cpu_runner  = OpenVINORunner(self.model_xml, device="CPU",    precision="fp32")
        self._ncs2_runner = OpenVINORunner(self.model_xml, device="MYRIAD", precision="fp16")
        self._cpu_runner.load()
        self._ncs2_runner.load()

    def run(self, frames: np.ndarray) -> np.ndarray:
        from data_wise_allocation import DWARunner
        dwa = DWARunner(
            proc_a_fn=lambda b: np.stack([self._cpu_runner.infer(f) for f in b]),
            proc_b_fn=lambda b: np.stack([self._ncs2_runner.infer(f) for f in b]),
            t_solo_a_ms=28.4,   # Celeron J4125 solo estimate
            t_solo_b_ms=9.8,    # NCS2 solo estimate
            proc_a_name="CPU",
            proc_b_name="NCS2",
        )
        return dwa.run(frames)

    def close(self) -> None:
        if self._cpu_runner:
            self._cpu_runner.close()
        if self._ncs2_runner:
            self._ncs2_runner.close()


# ---------------------------------------------------------------------------
# Standalone demo / validation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Odroid H2 + NCS2 DWA — Bang for the Buck §6")
    print()
    print("Paper result (Table 5):")
    print("  MobileNetV2 solo NCS2:        9.8 ms/frame  → 100 frames = 980 ms")
    print("  MobileNetV2 DWA CPU+NCS2:     5.6 ms/frame  → 100 frames = 560 ms")
    print("  Latency reduction: 43%  (wall time: 560ms vs 980ms)")
    print()
    print("Split: CPU ~25 frames, NCS2 ~75 frames (proportional to solo speed)")
    from data_wise_allocation import compute_optimal_split
    n_a, n_b = compute_optimal_split(100, t_solo_a_ms=28.4, t_solo_b_ms=9.8)
    print(f"  Optimal split from formula: CPU={n_a}, NCS2={n_b} frames")
