"""
framework_runner.py — Unified SW framework comparison orchestrator (§5.3).

This module provides a single entry point to run inference across all
eight frameworks evaluated in the paper:

  Platform       Frameworks evaluated
  ─────────────  ────────────────────────────────────────────────
  Jetson AGX     TensorRT FP32/FP16/INT8, AutoScheduler
  Jetson NX      TensorRT FP32/FP16/INT8, AutoScheduler
  Jetson TX2     TensorRT FP32/FP16, AutoScheduler
  Jetson Nano    TensorRT FP32/FP16, TFLite CPU
  Odroid H2      OpenVINO CPU FP32, OpenVINO GPU FP16, NCS2 VPU FP16
  Odroid M1      ARM NN CPU/GPU, PyARMNN, RKNN INT8, TFLite CPU
  Khadas VIM3    ARM NN CPU/GPU, PyARMNN, KSNN INT8, TFLite CPU

Usage:
    runner = FrameworkRunner(
        model_path={"tflite": "mobilenet.tflite", "onnx": "mobilenet.onnx"},
        platform="vim3",
        framework="ksnn",
        precision="int8",
    )
    runner.load()
    results = runner.sweep(input_data, n_iters=100)
"""

import importlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Framework registry: maps (platform, framework) → runner module + class
# ---------------------------------------------------------------------------

FRAMEWORK_RUNNER_MAP = {
    # NVIDIA platforms
    ("agx",    "tensorrt"):      ("runners.tensorrt_runner",     "TensorRTRunner"),
    ("nx",     "tensorrt"):      ("runners.tensorrt_runner",     "TensorRTRunner"),
    ("tx2",    "tensorrt"):      ("runners.tensorrt_runner",     "TensorRTRunner"),
    ("nano",   "tensorrt"):      ("runners.tensorrt_runner",     "TensorRTRunner"),
    ("agx",    "autoschedule"):  ("runners.autoschedule_runner", "AutoScheduleRunner"),
    ("tx2",    "autoschedule"):  ("runners.autoschedule_runner", "AutoScheduleRunner"),
    # Intel / Odroid H2
    ("h2",     "openvino"):      ("runners.openvino_runner",     "OpenVINORunner"),
    ("h2",     "openvino_ncs2"): ("runners.openvino_runner",     "OpenVINORunner"),
    ("h2",     "openvino_multi"):("runners.openvino_runner",     "OpenVINOMultiDeviceRunner"),
    # ARM NN platforms
    ("vim3",   "armnn"):         ("runners.armnn_runner",        "ArmNNRunner"),
    ("vim3",   "pyarmnn"):       ("runners.pyarmnn_runner",      "PyArmNNRunner"),
    ("vim3",   "tflite"):        ("runners.tflite_runner",       "TFLiteRunner"),
    ("vim3",   "ksnn"):          ("runners.ksnn_runner",         "KSNNRunner"),
    ("m1",     "armnn"):         ("runners.armnn_runner",        "ArmNNRunner"),
    ("m1",     "pyarmnn"):       ("runners.pyarmnn_runner",      "PyArmNNRunner"),
    ("m1",     "tflite"):        ("runners.tflite_runner",       "TFLiteRunner"),
    ("m1",     "rknn"):          ("runners.rknn_runner",         "RKNNRunner"),
    # TFLite on any platform
    ("nano",   "tflite"):        ("runners.tflite_runner",       "TFLiteRunner"),
}

PLATFORM_ALIASES = {
    "jetson_agx": "agx", "xavier": "agx",
    "jetson_nx":  "nx",
    "jetson_tx2": "tx2",
    "jetson_nano": "nano",
    "odroid_h2":  "h2",
    "odroid_m1":  "m1",
    "khadas_vim3": "vim3",
}


@dataclass
class FrameworkRunResult:
    """Structured result from a single framework benchmark run."""
    platform:   str
    framework:  str
    precision:  str
    model_name: str
    n_iters:    int
    latencies_ms: List[float] = field(default_factory=list)

    @property
    def mean_ms(self) -> float:
        return float(np.mean(self.latencies_ms)) if self.latencies_ms else 0.0

    @property
    def median_ms(self) -> float:
        return float(np.median(self.latencies_ms)) if self.latencies_ms else 0.0

    @property
    def p90_ms(self) -> float:
        return float(np.percentile(self.latencies_ms, 90)) if self.latencies_ms else 0.0

    @property
    def fps(self) -> float:
        return 1000.0 / self.mean_ms if self.mean_ms > 0 else 0.0

    def summary(self) -> Dict[str, Any]:
        return {
            "platform":   self.platform,
            "framework":  self.framework,
            "precision":  self.precision,
            "model":      self.model_name,
            "mean_ms":    round(self.mean_ms, 3),
            "median_ms":  round(self.median_ms, 3),
            "p90_ms":     round(self.p90_ms, 3),
            "fps":        round(self.fps, 2),
            "n_iters":    self.n_iters,
        }


class FrameworkRunner:
    """
    Factory + runner for any (platform, framework, precision) combination.

    Dynamically loads the correct runner class from the registry.

    Usage:
        fr = FrameworkRunner(
            model_paths={"tflite": "mob.tflite", "onnx": "mob.onnx", "rknn": "mob.rknn"},
            platform="m1",
            framework="rknn",
            precision="int8",
            model_name="MobileNetV2",
        )
        fr.load()
        result = fr.sweep(preloaded_input, warmup=20, measure=100)
        print(result.summary())
    """

    def __init__(
        self,
        model_paths: Dict[str, str],
        platform: str,
        framework: str,
        precision: str = "fp32",
        model_name: str = "unknown",
        extra_kwargs: Optional[Dict] = None,
    ):
        """
        Args:
            model_paths:   Dict of format → path (e.g. {"tflite": "...", "onnx": "..."})
            platform:      Platform key (see PLATFORM_ALIASES)
            framework:     Framework key (see FRAMEWORK_RUNNER_MAP)
            precision:     "fp32", "fp16", "int8"
            model_name:    Human-readable model name (used in results)
            extra_kwargs:  Additional kwargs forwarded to the runner constructor
        """
        self.model_paths  = model_paths
        self.platform     = PLATFORM_ALIASES.get(platform.lower(), platform.lower())
        self.framework    = framework.lower()
        self.precision    = precision.lower()
        self.model_name   = model_name
        self.extra_kwargs = extra_kwargs or {}

        self._runner = None

    def load(self) -> None:
        """Load the appropriate framework runner."""
        key = (self.platform, self.framework)
        if key not in FRAMEWORK_RUNNER_MAP:
            raise ValueError(
                f"No runner for ({self.platform}, {self.framework}).\n"
                f"Available: {sorted(FRAMEWORK_RUNNER_MAP.keys())}"
            )

        module_path, class_name = FRAMEWORK_RUNNER_MAP[key]
        mod   = importlib.import_module(f"4_sw_framework_comparison.{module_path}")
        cls   = getattr(mod, class_name)

        # Infer the right model file format
        model_path = self._pick_model_path()
        self._runner = cls(
            model_path=model_path,
            precision=self.precision,
            **self.extra_kwargs,
        )
        self._runner.load()
        logger.info("Loaded %s/%s/%s", self.platform, self.framework, self.precision)

    def infer(self, input_data: np.ndarray) -> np.ndarray:
        if self._runner is None:
            raise RuntimeError("Call load() first")
        return self._runner.infer(input_data)

    def sweep(
        self,
        input_data: np.ndarray,
        warmup: int = 20,
        measure: int = 100,
    ) -> FrameworkRunResult:
        """
        Standard warmup + measurement sweep.

        Returns a FrameworkRunResult with latency statistics.
        """
        if self._runner is None:
            raise RuntimeError("Call load() first")

        # Warmup
        for _ in range(warmup):
            self._runner.infer(input_data)

        # Measure
        latencies = []
        for _ in range(measure):
            t0 = time.perf_counter()
            self._runner.infer(input_data)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

        result = FrameworkRunResult(
            platform=self.platform,
            framework=self.framework,
            precision=self.precision,
            model_name=self.model_name,
            n_iters=measure,
            latencies_ms=latencies,
        )
        return result

    def close(self) -> None:
        if self._runner is not None:
            self._runner.close()
            self._runner = None

    def _pick_model_path(self) -> str:
        """Select model file matching the framework's preferred format."""
        fmt_priority = {
            "tensorrt":    ["onnx", "engine"],
            "autoschedule":["onnx"],
            "openvino":    ["xml", "onnx"],
            "openvino_ncs2": ["xml", "onnx"],
            "openvino_multi": ["xml", "onnx"],
            "tflite":      ["tflite"],
            "armnn":       ["tflite", "onnx"],
            "pyarmnn":     ["tflite", "onnx"],
            "ksnn":        ["nb"],
            "rknn":        ["rknn"],
        }
        order = fmt_priority.get(self.framework, list(self.model_paths.keys()))
        for fmt in order:
            if fmt in self.model_paths:
                return self.model_paths[fmt]
        # Fallback to first available
        if self.model_paths:
            return next(iter(self.model_paths.values()))
        raise ValueError("model_paths is empty")


# ---------------------------------------------------------------------------
# Full platform × framework sweep utility (matches §5.3 evaluation)
# ---------------------------------------------------------------------------

def run_all_frameworks(
    model_name: str,
    model_paths: Dict[str, str],
    input_data: np.ndarray,
    platforms_and_frameworks: Optional[List] = None,
    warmup: int = 20,
    measure: int = 100,
) -> List[Dict]:
    """
    Run all valid (platform, framework) combinations for a model.

    Returns:
        List of result.summary() dicts — suitable for DataFrame or CSV export.
    """
    combos = platforms_and_frameworks or list(FRAMEWORK_RUNNER_MAP.keys())
    all_results = []

    for platform, framework in combos:
        for precision in ("fp32", "fp16", "int8"):
            try:
                fr = FrameworkRunner(
                    model_paths=model_paths,
                    platform=platform,
                    framework=framework,
                    precision=precision,
                    model_name=model_name,
                )
                fr.load()
                result = fr.sweep(input_data, warmup=warmup, measure=measure)
                all_results.append(result.summary())
                fr.close()
                logger.info("✓ %s/%s/%s → %.1f ms", platform, framework, precision, result.mean_ms)
            except Exception as exc:
                logger.warning("✗ %s/%s/%s — %s", platform, framework, precision, exc)

    return all_results


if __name__ == "__main__":
    print("Framework Runner — Bang for the Buck §5.3")
    print(f"\nRegistered platform/framework combinations ({len(FRAMEWORK_RUNNER_MAP)}):")
    for (plat, fw), (mod, cls) in sorted(FRAMEWORK_RUNNER_MAP.items()):
        print(f"  {plat:<6} × {fw:<18} → {cls}")
