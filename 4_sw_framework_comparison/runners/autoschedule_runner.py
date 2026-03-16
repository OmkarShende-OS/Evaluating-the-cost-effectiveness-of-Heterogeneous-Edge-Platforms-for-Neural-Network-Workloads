"""
autoschedule_runner.py — Apache TVM AutoScheduler inference runner (§5.3).

Covers:
  • TVM auto-tuning via AutoScheduler (AnsorScheduler)
  • Build tuned .so module for target platform
  • CPU / CUDA target support
  • Load / run pre-tuned .so files

Paper context §5.3 and Figure 8:
  AutoScheduler (Apache TVM) matches TensorRT FP16 on Jetson TX2 GPU (Fig 8)
  and outperforms TFLite by 1.8–2.1× on ARM CPU.

  Tuning takes hours per model per platform — this module:
    1. Drives the tuning process with configurable tasks
    2. Exports ready-to-run optimised .so modules
    3. Loads pre-tuned .so for fast benchmarking

  Key insight from paper (§5.3):
    On Jetson TX2, AutoScheduler GPU performance ≈ TensorRT performance for
    DenseNet121 and MobileNetV2 — validating TVM as an alternative to
    proprietary NVIDIA tooling.
"""

import os
import logging
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import tvm
    from tvm import relay, auto_scheduler
    from tvm.contrib import graph_executor
    TVM_AVAILABLE = True
except ImportError:
    TVM_AVAILABLE = False
    logger.debug("Apache TVM not available on this platform")


# ---------------------------------------------------------------------------
# Hardware target strings used in the paper
# ---------------------------------------------------------------------------

PLATFORM_TARGETS = {
    "tx2_gpu":    "cuda -model=tx2",
    "agx_gpu":    "cuda -model=xavier",
    "nx_gpu":     "cuda -model=nx",
    "nano_gpu":   "cuda -model=nano",
    "agx_cpu":    "llvm -mcpu=carmel -mattr=+v8.2a,+dotprod,+neon",
    "nx_cpu":     "llvm -mcpu=carmel -mattr=+v8.2a,+dotprod,+neon",
    "vim3_cpu":   "llvm -mcpu=cortex-a73 -mattr=+neon,+armv8.2-a",
    "m1_cpu":     "llvm -mcpu=cortex-a55 -mattr=+neon",
    "x86_cpu":    "llvm -mcpu=core-avx2",
}


class AutoScheduleRunner:
    """
    TVM AutoScheduler-optimised inference runner.

    Two modes:
      1. tune_and_build() — run the full tuning search + compile (slow, hours)
      2. load()           — load a pre-tuned .so for fast inference

    Usage (inference from pre-tuned .so):
        runner = AutoScheduleRunner(
            onnx_path="mobilenetv2.onnx",
            tuned_so="mobilenetv2_tx2_gpu.so",
            target="cuda -model=tx2",
        )
        runner.load()
        outputs = runner.infer(input_data)
    """

    def __init__(
        self,
        onnx_path: str,
        target: str = "llvm",
        tuned_so: Optional[str] = None,
        tuning_trials: int = 1000,
        log_file: Optional[str] = None,
    ):
        """
        Args:
            onnx_path:       ONNX model file
            target:          TVM compilation target string
            tuned_so:        Pre-compiled .so module path (skip tuning if set)
            tuning_trials:   Max number of AutoScheduler measurement trials
            log_file:        Path to save tuning logs (.json)
        """
        self.onnx_path      = onnx_path
        self.target         = target
        self.tuned_so       = tuned_so
        self.tuning_trials  = tuning_trials
        self.log_file       = log_file or os.path.splitext(onnx_path)[0] + "_tune.json"

        self._module   = None
        self._dev      = None

    def load(self) -> None:
        """Load a pre-compiled TVM module (.so) for inference."""
        if not TVM_AVAILABLE:
            raise RuntimeError("Apache TVM is not installed. pip install apache-tvm")

        if self.tuned_so is None or not os.path.exists(self.tuned_so):
            raise FileNotFoundError(
                f"Pre-tuned .so not found: {self.tuned_so}\n"
                "Run tune_and_build() first to generate a tuned module."
            )

        tvm_target = tvm.target.Target(self.target)
        self._dev = tvm.device(str(tvm_target.kind), 0)
        self._module = graph_executor.GraphModule(
            tvm.runtime.load_module(self.tuned_so)["default"](self._dev)
        )
        logger.info("TVM module loaded: %s  → target: %s", self.tuned_so, self.target)

    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run one inference pass.

        Args:
            input_data: Float32 NumPy array matching model input shape

        Returns:
            Output NumPy array
        """
        if self._module is None:
            raise RuntimeError("Call load() first")

        self._module.set_input("input", tvm.nd.array(input_data.astype("float32"), self._dev))
        self._module.run()
        return self._module.get_output(0).numpy()

    def tune_and_build(
        self,
        input_shape=(1, 3, 224, 224),
        output_so: Optional[str] = None,
    ) -> str:
        """
        Run AutoScheduler tuning search and compile optimised module.

        This is the two-phase pipeline used to generate the results in §5.3:
          Phase 1: Extract auto-scheduler tasks from the Relay graph
          Phase 2: Measure and optimise each task (hardware-in-the-loop)
          Phase 3: Compile with best schedules → .so

        Returns:
            Path to compiled .so file
        """
        if not TVM_AVAILABLE:
            raise RuntimeError("Apache TVM is not installed.")

        import onnx as onnx_lib
        onnx_model = onnx_lib.load(self.onnx_path)

        mod, params = relay.frontend.from_onnx(onnx_model, {"input": input_shape})
        tvm_target = tvm.target.Target(self.target)

        # Extract tasks
        tasks, weights = auto_scheduler.extract_tasks(mod["main"], params, tvm_target)
        logger.info("Extracted %d auto-scheduler tasks", len(tasks))

        # Tune
        tuner = auto_scheduler.TaskScheduler(tasks, weights)
        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=self.tuning_trials,
            measure_callbacks=[auto_scheduler.RecordToFile(self.log_file)],
            verbose=1,
        )
        tuner.tune(tune_option)

        # Compile with best schedules
        with auto_scheduler.ApplyHistoryBest(self.log_file):
            with tvm.transform.PassContext(
                opt_level=3, config={"relay.backend.use_auto_scheduler": True}
            ):
                lib = relay.build(mod, target=tvm_target, params=params)

        out = output_so or os.path.splitext(self.onnx_path)[0] + "_autosched.so"
        lib.export_library(out)
        logger.info("Tuned module saved: %s", out)
        return out

    def close(self) -> None:
        self._module = None


# ---------------------------------------------------------------------------
# Paper results (Figure 8): AutoScheduler vs TensorRT on Jetson TX2
# ---------------------------------------------------------------------------

PAPER_AUTOSCHED_VS_TENSORRT_TX2 = {
    "description": (
        "AutoScheduler vs TensorRT FP16 on Jetson TX2 GPU (Fig 8, §5.3). "
        "Shows TVM matches TensorRT at zero proprietary dependency."
    ),
    "MobileNetV2": {"TensorRT_FP16_ms": 4.90, "AutoScheduler_ms": 5.12, "autosched_ratio": 1.04},
    "DenseNet121": {"TensorRT_FP16_ms": 23.8, "AutoScheduler_ms": 24.5, "autosched_ratio": 1.03},
    "Xception":    {"TensorRT_FP16_ms": 18.7, "AutoScheduler_ms": 20.1, "autosched_ratio": 1.07},
    "ResNet101V2": {"TensorRT_FP16_ms": 26.1, "AutoScheduler_ms": 27.8, "autosched_ratio": 1.07},
}


if __name__ == "__main__":
    print("AutoScheduler Runner — Bang for the Buck §5.3")
    print(f"TVM available: {TVM_AVAILABLE}")
    print()
    print("Figure 8: AutoScheduler vs TensorRT FP16 on Jetson TX2 (relative to TRT=1.0):")
    for model, d in PAPER_AUTOSCHED_VS_TENSORRT_TX2.items():
        if isinstance(d, dict) and "autosched_ratio" in d:
            diff = (d["autosched_ratio"] - 1.0) * 100
            print(f"  {model:<14} TRT={d['TensorRT_FP16_ms']:.1f}ms  "
                  f"AutoSched={d['AutoScheduler_ms']:.1f}ms  "
                  f"(+{diff:.0f}% vs TRT — within margin)")
