"""
openvino_runner.py — Intel OpenVINO inference runner (§5.3).

Covers:
  • CPU FP32 inference using OpenVINO Core (API 2.0)
  • NCS2 (Intel Neural Compute Stick 2) VPU inference
  • FP16 model optimization via model optimizer / auto-downcast
  • Multi-device (CPU+NCS2) via MULTI plugin — used in §6 DWA

Paper context §5.3:
  OpenVINO is evaluated on Odroid H2 with and without the NCS2 stick.
  NCS2 VPU achieves FP16 inference 1.6× faster than H2's Intel UHD 600 GPU.
  CPU FP32 on Celeron J4125 is 1.8× faster than CPU-only TFLite.
  OpenVINO MULTI plugin enables DWA (Data-Wise Allocation) in Section 6:
    → H2_CPU + NCS2 DWA reduces end-to-end latency by 43%.
"""

import os
import logging
import numpy as np
from typing import Dict, Optional

logger = logging.getLogger(__name__)

try:
    from openvino.runtime import Core, CompiledModel, InferRequest
    OV_AVAILABLE = True
except ImportError:
    OV_AVAILABLE = False
    logger.debug("OpenVINO runtime not available")


class OpenVINORunner:
    """
    Inference runner using Intel OpenVINO Runtime (API 2.0).

    Supported devices:
      "CPU"   — x86 / ARM with OpenVINO CPU plugin
      "GPU"   — Intel integrated GPU (UHD 600 on Odroid H2)
      "MYRIAD" or "NCS2" — Intel Neural Compute Stick 2 VPU
      "MULTI:CPU,MYRIAD" — Multi-device plugin for DWA (see Section 6)
      "AUTO"  — Auto-select best device

    Usage:
        runner = OpenVINORunner("model.xml", device="MYRIAD", precision="fp16")
        runner.load()
        output = runner.infer(input_array)
    """

    def __init__(
        self,
        model_path: str,
        device: str = "CPU",
        precision: str = "fp32",
        num_streams: int = 1,
    ):
        """
        Args:
            model_path:   Path to OpenVINO IR (.xml) or ONNX (.onnx) model
            device:       Target device string (see class docstring)
            precision:    "fp32" or "fp16" — NCS2 requires FP16
            num_streams:  Inference streams for throughput mode (CPU/GPU async)
        """
        self.model_path  = model_path
        self.device      = self._normalise_device(device)
        self.precision   = precision.lower()
        self.num_streams = num_streams

        self._core:     Optional[object] = None
        self._compiled: Optional[object] = None
        self._request:  Optional[object] = None
        self._input_name:  str = ""
        self._output_name: str = ""

    def load(self) -> None:
        """Compile model for target device."""
        if not OV_AVAILABLE:
            raise RuntimeError(
                "OpenVINO runtime not installed.\n"
                "Install with: pip install openvino-dev"
            )

        self._core = Core()

        config: Dict = {}
        if "MYRIAD" in self.device or "NCS2" in self.device:
            # MYRIAD FP16 only — set model precision
            config["MYRIAD_ENABLE_HW_ACCELERATION"] = "YES"
        elif self.device == "CPU":
            config["CPU_THREADS_NUM"] = "4"
            if self.num_streams > 1:
                config["CPU_THROUGHPUT_STREAMS"] = str(self.num_streams)
        elif self.device == "GPU":
            config["GPU_THROUGHPUT_STREAMS"] = str(self.num_streams)

        model = self._core.read_model(model=self.model_path)
        self._compiled = self._core.compile_model(model=model, device_name=self.device, config=config)
        self._request = self._compiled.create_infer_request()

        inputs = self._compiled.inputs
        outputs = self._compiled.outputs
        self._input_name  = inputs[0].get_any_name()
        self._output_name = outputs[0].get_any_name()

        logger.info("OpenVINO model compiled — device=%s  precision=%s", self.device, self.precision)

    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """
        Synchronous inference.

        Args:
            input_data: NumPy array conforming to model input shape

        Returns:
            Output as NumPy array
        """
        if self._request is None:
            raise RuntimeError("Call load() first")

        dtype = np.float16 if self.precision == "fp16" else np.float32
        self._request.set_tensor(self._input_name, input_data.astype(dtype))
        self._request.infer()
        return self._request.get_tensor(self._output_name).data.copy()

    def close(self) -> None:
        self._compiled = None
        self._core     = None

    @staticmethod
    def _normalise_device(device: str) -> str:
        """Map friendly names to OpenVINO device strings."""
        mapping = {
            "ncs2":        "MYRIAD",
            "vpu":         "MYRIAD",
            "myriad":      "MYRIAD",
            "igpu":        "GPU",
            "intel_gpu":   "GPU",

        }
        return mapping.get(device.lower(), device.upper())

    @property
    def input_dtype(self) -> type:
        return np.float16 if self.precision == "fp16" else np.float32


# ---------------------------------------------------------------------------
# Multi-device runner for DWA (§6 Concurrent Execution)
# ---------------------------------------------------------------------------

class OpenVINOMultiDeviceRunner(OpenVINORunner):
    """
    OpenVINO MULTI plugin runner.

    Used in §6 DWA on Odroid H2: splits batches across CPU and NCS2.
    The MULTI plugin's load-balancing achieves ~43% latency reduction.

    Usage:
        runner = OpenVINOMultiDeviceRunner("model.xml",
                                           devices=["CPU", "MYRIAD"],
                                           priorities="CPU,MYRIAD")
        runner.load()
    """

    def __init__(self, model_path: str, devices: list, **kwargs):
        device_str = "MULTI:" + ",".join([d.upper() for d in devices])
        super().__init__(model_path, device=device_str, **kwargs)
        self.devices = devices


# ---------------------------------------------------------------------------
# Paper validation data
# ---------------------------------------------------------------------------

PAPER_OPENVINO_RESULTS = {
    "description": (
        "OpenVINO inference times on Odroid H2 (Celeron J4125) and NCS2 VPU (§5.3)"
    ),
    "MobileNetV2": {
        "H2_CPU_FP32_ms":  28.4,
        "H2_GPU_FP16_ms":  24.1,
        "NCS2_VPU_FP16_ms": 9.8,
    },
    "ResNet101V2": {
        "H2_CPU_FP32_ms":  210.0,
        "H2_GPU_FP16_ms":  185.0,
        "NCS2_VPU_FP16_ms": 74.0,
    },
    "DenseNet121": {
        "H2_CPU_FP32_ms":  195.0,
        "H2_GPU_FP16_ms":  172.0,
        "NCS2_VPU_FP16_ms": 68.0,
    },
    "Xception": {
        "H2_CPU_FP32_ms":  340.0,
        "H2_GPU_FP16_ms":  298.0,
        "NCS2_VPU_FP16_ms": 108.0,
    },
}

# DWA result: using MULTI:CPU,MYRIAD vs solo NCS2 baseline (§6, Figure 9)
PAPER_DWA_H2_SPEEDUP = {
    "MobileNetV2": 1.43,
    "ResNet101V2": 1.41,
    "DenseNet121": 1.40,
    "Xception":    1.39,
}


if __name__ == "__main__":
    print("OpenVINO Runner — Bang for the Buck §5.3")
    print(f"OpenVINO available: {OV_AVAILABLE}")
    print()
    print("Paper inference times on Odroid H2 (ms):")
    for model, vals in PAPER_OPENVINO_RESULTS.items():
        if isinstance(vals, dict) and "H2_CPU_FP32_ms" in vals:
            print(
                f"  {model:<14} CPU={vals['H2_CPU_FP32_ms']:.0f}ms  "
                f"GPU={vals['H2_GPU_FP16_ms']:.0f}ms  "
                f"NCS2={vals['NCS2_VPU_FP16_ms']:.0f}ms"
            )
    print()
    print("DWA MULTI speedup vs solo NCS2 (§6, ~43%):")
    for m, s in PAPER_DWA_H2_SPEEDUP.items():
        print(f"  {m:<14} {s:.2f}×")
