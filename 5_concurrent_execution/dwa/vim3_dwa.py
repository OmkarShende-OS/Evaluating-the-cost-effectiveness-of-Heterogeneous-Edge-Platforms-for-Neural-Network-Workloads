"""
vim3_dwa.py — Khadas VIM3 (Big CPU + Amlogic NPU) DWA implementation (§6).

On VIM3, the Big CPU cluster (Cortex-A73×4) and the dedicated Amlogic NPU
are used in a manual DWA setup. Unlike Odroid H2, there is NO hardware
MULTI plugin for the Amlogic NPU — split must be done manually.

Paper finding (§6):
  VIM3 DWA shows only ~11% improvement vs solo KSNN NPU.
  Root cause: KSNN NPU is 7× faster than VIM3 Big CPU.
  Optimal split gives CPU only ~12.5% of frames.
  For 100 frames: CPU handles 13, NPU handles 87.
  CPU still finishes after NPU for many models → little wall-time gain.

  This demonstrates the paper's core DWA insight:
    DWA is only effective when the two processors have similar speed.
    High speed asymmetry → minimal improvement.
"""

import logging
import threading
import time
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class VIM3DWA:
    """
    Khadas VIM3 Big CPU + Amlogic KSNN NPU manual Data-Wise Allocation.

    Uses threading to run ARM NN (Big CPU) and KSNN (NPU) simultaneously.
    The optimal split assigns ~12–13% of frames to CPU.

    Usage:
        dwa = VIM3DWA(
            tflite_model="mobilenetv2.tflite",
            ksnn_model="mobilenetv2_int8.nb",
        )
        dwa.load()
        result = dwa.run(input_batch_100)
        print(dwa.benchmark(input_batch_100))
    """

    # Solo inference times on VIM3 for MobileNetV2 (from paper §5.3)
    SOLO_TIMES_MS = {
        "MobileNetV2": {"CPU_Big_ms": 38.2, "KSNN_NPU_ms": 5.4},
        "ResNet101V2": {"CPU_Big_ms": 280.0, "KSNN_NPU_ms": 40.2},
        "DenseNet121": {"CPU_Big_ms": 260.0, "KSNN_NPU_ms": 37.8},
        "Xception":    {"CPU_Big_ms": 430.0, "KSNN_NPU_ms": 63.0},
    }

    def __init__(
        self,
        tflite_model: str,
        ksnn_model:   str,
        model_name:   str = "MobileNetV2",
        num_cpu_threads: int = 4,
    ):
        """
        Args:
            tflite_model:      TFLite .tflite model for ARM NN Big CPU
            ksnn_model:        KSNN .nb model for Amlogic NPU
            model_name:        Used to look up solo times (for split calculation)
            num_cpu_threads:   CPU thread count for ARM NN
        """
        self.tflite_model    = tflite_model
        self.ksnn_model      = ksnn_model
        self.model_name      = model_name
        self.num_cpu_threads = num_cpu_threads

        self._cpu_runner  = None
        self._ksnn_runner = None

        # Look up solo times from paper data
        model_times = self.SOLO_TIMES_MS.get(model_name, {"CPU_Big_ms": 38.2, "KSNN_NPU_ms": 5.4})
        self._t_cpu_ms  = model_times["CPU_Big_ms"]
        self._t_ksnn_ms = model_times["KSNN_NPU_ms"]

    def load(self) -> None:
        """Load both CPU (TFLite) and NPU (KSNN) runtimes."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

        try:
            from bang_for_the_buck.four_sw_framework_comparison.runners.tflite_runner import TFLiteRunner
            self._cpu_runner = TFLiteRunner(
                self.tflite_model,
                num_threads=self.num_cpu_threads,
                use_xnnpack=True,
            )
            self._cpu_runner.load()
            logger.info("VIM3 CPU runner loaded (TFLite+XNNPACK)")
        except ImportError:
            logger.warning("TFLite not available — CPU DWA branch disabled")

        try:
            from bang_for_the_buck.four_sw_framework_comparison.runners.ksnn_runner import KSNNRunner
            self._ksnn_runner = KSNNRunner(self.ksnn_model, precision="int8")
            self._ksnn_runner.load()
            logger.info("VIM3 KSNN NPU runner loaded")
        except (ImportError, RuntimeError) as e:
            logger.warning("KSNN not available: %s", e)

    def compute_split(self, n_frames: int) -> Tuple[int, int]:
        """
        Compute optimal frame split given solo inference times.

        Returns:
            (n_cpu, n_npu) where n_cpu + n_npu == n_frames
        """
        from data_wise_allocation import compute_optimal_split
        return compute_optimal_split(n_frames, self._t_cpu_ms, self._t_ksnn_ms)

    def run(self, frames: np.ndarray) -> Optional[np.ndarray]:
        """
        Run DWA: CPU handles small fraction, NPU handles the rest.

        Args:
            frames: (N, H, W, C) uint8 or float32 array

        Returns:
            Combined output array (N, num_classes)
        """
        n = len(frames)
        n_cpu, n_npu = self.compute_split(n)

        logger.debug("VIM3 DWA: CPU=%d frames, NPU=%d frames", n_cpu, n_npu)

        result_cpu:  Optional[np.ndarray] = None
        result_npu:  Optional[np.ndarray] = None
        exc_cpu = exc_npu = None

        def run_cpu():
            nonlocal result_cpu, exc_cpu
            if self._cpu_runner is None or n_cpu == 0:
                result_cpu = np.array([])
                return
            try:
                out = [self._cpu_runner.infer(frames[i]) for i in range(n_cpu)]
                result_cpu = np.stack(out, axis=0)
            except Exception as e:
                exc_cpu = e

        def run_npu():
            nonlocal result_npu, exc_npu
            if self._ksnn_runner is None or n_npu == 0:
                result_npu = np.array([])
                return
            try:
                out = [self._ksnn_runner.infer(frames[n_cpu + i]) for i in range(n_npu)]
                result_npu = np.stack(out, axis=0)
            except Exception as e:
                exc_npu = e

        t0 = time.perf_counter()
        t_cpu = threading.Thread(target=run_cpu, daemon=True)
        t_npu = threading.Thread(target=run_npu, daemon=True)
        t_cpu.start(); t_npu.start()
        t_cpu.join();  t_npu.join()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if exc_cpu:
            raise RuntimeError(f"CPU thread failed: {exc_cpu}") from exc_cpu
        if exc_npu:
            raise RuntimeError(f"NPU thread failed: {exc_npu}") from exc_npu

        logger.debug("VIM3 DWA wall time: %.1f ms", elapsed_ms)

        parts = [r for r in [result_cpu, result_npu] if r is not None and r.size > 0]
        return np.concatenate(parts, axis=0) if len(parts) > 1 else (parts[0] if parts else None)

    def benchmark(self, frames: np.ndarray, n_runs: int = 50) -> Dict[str, float]:
        """Benchmark DWA vs solo NPU baseline."""
        for _ in range(5):  # warmup
            self.run(frames[:1])

        lats = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            self.run(frames)
            lats.append((time.perf_counter() - t0) * 1000)

        mean_ms    = float(np.mean(lats))
        n          = len(frames)
        solo_cpu   = self._t_cpu_ms * n
        solo_npu   = self._t_ksnn_ms * n
        n_cpu, n_npu = self.compute_split(n)

        return {
            "platform":           "Khadas_VIM3",
            "model":              self.model_name,
            "n_frames":           n,
            "split_cpu_n":        n_cpu,
            "split_npu_n":        n_npu,
            "dwa_mean_ms":        round(mean_ms, 2),
            "solo_cpu_est_ms":    round(solo_cpu, 2),
            "solo_npu_est_ms":    round(solo_npu, 2),
            "speedup_vs_npu":     round(solo_npu / mean_ms, 3) if mean_ms else 0,
            "paper_improvement_pct": 11.0,  # as reported in paper
        }

    def close(self) -> None:
        if self._cpu_runner:
            self._cpu_runner.close()
        if self._ksnn_runner:
            self._ksnn_runner.close()


# ---------------------------------------------------------------------------
# DWA insight analysis — reproduce paper's asymmetry argument
# ---------------------------------------------------------------------------

def dwa_gap_analysis() -> None:
    """
    Reproduce paper §6 analysis: why VIM3 DWA has limited improvement.

    The key metric is speed_ratio = T_slow / T_fast.
    Large ratio → only small fraction of work goes to slow processor
                → CPU finishes quickly but idle most of the time
                → wall time dominated by NPU
    """
    print("DWA Gap Analysis — VIM3 vs H2 speed asymmetry (§6)\n")
    platforms = {
        "Odroid H2 (CPU + NCS2)": {
            "T_CPU": 28.4,  "T_NPU": 9.8,
            "note": "Moderate asymmetry → good DWA"
        },
        "Khadas VIM3 (BigCPU + KSNN)": {
            "T_CPU": 38.2,  "T_NPU": 5.4,
            "note": "High asymmetry → limited DWA"
        },
    }

    for plat, vals in platforms.items():
        t_a, t_b = vals["T_CPU"], vals["T_NPU"]
        total = t_a + t_b
        r_a = t_b / total   # CPU fraction
        r_b = t_a / total   # NPU fraction
        t_parallel = max(t_a * r_a, t_b * r_b)
        speedup = min(t_a, t_b) / t_parallel if t_parallel else 0
        ratio = t_a / t_b

        print(f"Platform: {plat}")
        print(f"  Solo CPU: {t_a:.1f}ms  Solo NPU: {t_b:.1f}ms  ratio: {ratio:.1f}×")
        print(f"  Optimal split: CPU gives {r_a*100:.0f}% to CPU, {r_b*100:.0f}% to NPU")
        print(f"  Theoretical DWA speedup: {speedup:.2f}×  | {vals['note']}")
        print()


if __name__ == "__main__":
    dwa_gap_analysis()
