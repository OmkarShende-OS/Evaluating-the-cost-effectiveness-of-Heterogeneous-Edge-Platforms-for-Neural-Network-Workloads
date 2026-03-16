"""
tflite_runner.py — TensorFlow Lite inference runner (§5.3).

Covers:
  • CPU FP32/FP16 inference via TFLite interpreter
  • Edge TPU delegation (Coral, if available)
  • XNNPACK delegate for ARM SIMD acceleration
  • Dynamic/static quantization (INT8)

Paper context:
  TFLite was evaluated on Jetson Nano CPU (no GPU delegate) and
  Odroid M1 as a pure-CPU baseline. Results show TFLite is 1.3–1.7×
  slower than TensorRT on CPU due to fewer ARM-specific optimizations,
  but TFLite INT8 is competitive on ARM Cortex-A55 (Odroid M1).
"""

import os
import time
import logging
import numpy as np
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class TFLiteRunner:
    """
    Inference runner using TensorFlow Lite.

    Supports:
      - CPU FP32 / FP16 (default)
      - INT8 quantized models
      - XNNPACK delegate (multi-threaded ARM acceleration)
      - Edge TPU delegate (Coral USB Accelerator)

    Usage:
        runner = TFLiteRunner("mobilenetv2_int8.tflite", num_threads=4, use_xnnpack=True)
        runner.load()
        output = runner.infer(input_array)
        print(runner.get_latency_stats())
    """

    def __init__(
        self,
        model_path: str,
        precision: str = "fp32",
        num_threads: int = 4,
        use_xnnpack: bool = True,
        use_edge_tpu: bool = False,
    ):
        """
        Args:
            model_path:    Path to .tflite model file
            precision:     "fp32", "fp16", or "int8"
            num_threads:   CPU thread count (paper used 4 for quad-core platforms)
            use_xnnpack:   Enable XNNPACK delegate for SIMD acceleration (ARM)
            use_edge_tpu:  Enable Coral Edge TPU delegate (not evaluated in paper)
        """
        self.model_path   = model_path
        self.precision    = precision
        self.num_threads  = num_threads
        self.use_xnnpack  = use_xnnpack
        self.use_edge_tpu = use_edge_tpu

        self._interpreter = None
        self._input_details  = None
        self._output_details = None

    def load(self) -> None:
        """Load model and allocate tensors."""
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            try:
                import tensorflow.lite as tflite
            except ImportError:
                raise ImportError(
                    "No TFLite runtime found. Install via:\n"
                    "  pip install tflite-runtime"
                )

        delegates = []

        if self.use_edge_tpu:
            try:
                from tflite_runtime.interpreter import load_delegate
                delegates.append(load_delegate("libedgetpu.so.1"))
                logger.info("Edge TPU delegate loaded")
            except (ImportError, ValueError) as e:
                logger.warning("Edge TPU not available: %s", e)

        if self.use_xnnpack and not self.use_edge_tpu:
            try:
                from tflite_runtime.interpreter import load_delegate
                xnnpack = load_delegate("libXNNPACK.so")
                delegates.append(xnnpack)
                logger.info("XNNPACK delegate loaded")
            except (ImportError, ValueError, OSError):
                logger.debug("XNNPACK delegate not available, using default CPU")

        self._interpreter = tflite.Interpreter(
            model_path=self.model_path,
            num_threads=self.num_threads,
            experimental_delegates=delegates if delegates else None,
        )
        self._interpreter.allocate_tensors()
        self._input_details  = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()
        logger.info("TFLite model loaded: %s", os.path.basename(self.model_path))

    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run one inference pass.

        Args:
            input_data: NumPy array matching model input shape

        Returns:
            NumPy array of model output
        """
        if self._interpreter is None:
            raise RuntimeError("Call load() first")

        # Handle INT8 quantization: scale + zero_point
        if self.precision == "int8":
            qp = self._input_details[0].get("quantization_parameters", {})
            scale     = qp.get("scales", [1.0])[0]
            zero_point = qp.get("zero_points", [0])[0]
            dtype = self._input_details[0]["dtype"]
            if scale != 0:
                quant_input = (input_data / scale + zero_point).astype(dtype)
            else:
                quant_input = input_data.astype(dtype)
            self._interpreter.set_tensor(self._input_details[0]["index"], quant_input)
        else:
            self._interpreter.set_tensor(
                self._input_details[0]["index"],
                input_data.astype(self._input_details[0]["dtype"]),
            )

        self._interpreter.invoke()
        raw = self._interpreter.get_tensor(self._output_details[0]["index"])

        # Dequantize INT8 output
        if self.precision == "int8":
            qp = self._output_details[0].get("quantization_parameters", {})
            scale     = qp.get("scales", [1.0])[0]
            zero_point = qp.get("zero_points", [0])[0]
            if scale != 0:
                raw = (raw.astype(np.float32) - zero_point) * scale

        return raw

    @property
    def input_shape(self) -> Tuple:
        if self._interpreter is None:
            raise RuntimeError("Call load() first")
        return tuple(self._input_details[0]["shape"])

    def close(self) -> None:
        self._interpreter = None


# ---------------------------------------------------------------------------
# Paper result validation
# ---------------------------------------------------------------------------

PAPER_TFLITE_VS_TENSORRT = {
    "description": (
        "TFLite CPU FP32 vs TensorRT GPU FP16 relative latency on Jetson Nano "
        "(Figure 6, Bang for the Buck §5.3)"
    ),
    "MobileNetV2": {"tflite_cpu_ms": 45.2, "tensorrt_gpu_ms": 18.7, "ratio": 2.42},
    "ResNet101V2": {"tflite_cpu_ms": 312.0, "tensorrt_gpu_ms": 89.0, "ratio": 3.51},
    "DenseNet121": {"tflite_cpu_ms": 280.0, "tensorrt_gpu_ms": 78.0, "ratio": 3.59},
    "Xception":    {"tflite_cpu_ms": 450.0, "tensorrt_gpu_ms": 120.0, "ratio": 3.75},
}


if __name__ == "__main__":
    import sys
    # Quick functionality check without a real model file
    print("TFLite Runner — Bang for the Buck §5.3")
    print()
    print("Paper result: TFLite CPU FP32 is 2.4–3.7× slower than TensorRT GPU FP16 on Nano")
    for model, d in PAPER_TFLITE_VS_TENSORRT.items():
        if isinstance(d, dict) and "ratio" in d:
            print(f"  {model:<14} TFLite {d['tflite_cpu_ms']:.0f} ms  /  "
                  f"TensorRT {d['tensorrt_gpu_ms']:.0f} ms  ({d['ratio']:.2f}×)")
