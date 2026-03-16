"""
ksnn_runner.py — Khadas KSNN (VIM3 NPU) inference runner (§5.3).

KSNN (Khadas Special Neural Network) is the Amlogic NPU SDK for Khadas VIM3.
The VIM3 features an Amlogic A311D SoC with a built-in NPU capable of INT8
and INT16 inference.

KSNN setup workflow:
  1. Convert TFLite/ONNX model to KSNN .nb format using the KSNN tools
     (requires running on an x86 Linux machine with the Amlogic SDK):
         python3 ksnn_convert.py --input model.tflite --output model.nb
  2. Transfer the .nb file to the VIM3 board
  3. Use this runner to load and infer with the KSNN Python API

Paper context §5.3 and Figure 7:
  KSNN NPU achieves ~5× speedup over ARM NN GPU (GpuAcc FP16) on VIM3.
  INT8 KSNN is the fastest on VIM3 for all tested models.
  Limitation: KSNN INT8 accuracy drop is 1–3% top-1 vs FP32 baseline.
"""

import os
import logging
from typing import Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)

try:
    from ksnn.api import KSNN
    KSNN_AVAILABLE = True
except ImportError:
    KSNN_AVAILABLE = False
    logger.debug("KSNN Python API not available (requires Khadas VIM3 with KSNN SDK)")


class KSNNRunner:
    """
    Inference runner for Khadas VIM3 Amlogic NPU via KSNN Python SDK.

    Only runs on VIM3 hardware (Amlogic A311D NPU).
    Convert model to .nb format first with KSNN tools.

    Usage:
        runner = KSNNRunner("mobilenetv2_int8.nb", precision="int8")
        runner.load()
        logits = runner.infer(input_224x224)
    """

    def __init__(
        self,
        model_nb_path: str,
        precision: str = "int8",
        input_mean: float = 128.0,
        input_stdev: float = 128.0,
    ):
        """
        Args:
            model_nb_path: Path to .nb compiled NPU model
            precision:     "int8" or "uint8"
            input_mean:    Per-channel normalization mean (default 128 for INT8)
            input_stdev:   Per-channel std deviation (default 128)
        """
        self.model_nb_path = model_nb_path
        self.precision     = precision
        self.input_mean    = input_mean
        self.input_stdev   = input_stdev

        self._model = None

    def load(self) -> None:
        """Load the KSNN model onto the NPU."""
        if not KSNN_AVAILABLE:
            raise RuntimeError(
                "KSNN not available. This runner requires Khadas VIM3 hardware "
                "with KSNN SDK installed.\n"
                "SDK: https://github.com/khadas/ksnn"
            )

        if not os.path.exists(self.model_nb_path):
            raise FileNotFoundError(f"KSNN model not found: {self.model_nb_path}")

        self._model = KSNN("VIM3")
        self._model.nn_init(nbgraph=self.model_nb_path, level=0)
        logger.info("KSNN model loaded: %s", os.path.basename(self.model_nb_path))

    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run inference on the Amlogic NPU.

        Args:
            input_data: HWC uint8 image [0,255] — KSNN handles normalization internally

        Returns:
            Output float32 NumPy array
        """
        if self._model is None:
            raise RuntimeError("Call load() first")

        # KSNN expects HWC uint8 input
        if input_data.dtype != np.uint8:
            input_data = input_data.astype(np.uint8)

        results = self._model.nn_inference(
            input_data,
            platform="ONNX",
            channel_mean_value=f"{self.input_mean} {self.input_mean} "
                               f"{self.input_mean} {self.input_stdev}",
            reorder_channel="0 1 2",
            output_format=0,  # float32 output
        )
        return np.array(results, dtype=np.float32)

    def close(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None


# ---------------------------------------------------------------------------
# Model conversion helper (run on x86 host, not on VIM3)
# ---------------------------------------------------------------------------

def convert_tflite_to_nb(
    tflite_path: str,
    output_path: str,
    ksnn_tools_dir: str = "/path/to/ksnn_tools",
) -> None:
    """
    Convert TFLite model to KSNN .nb format.

    Must be run on x86 Linux with Amlogic KSNN tools installed.
    Transfer the resulting .nb file to the VIM3 for inference.

    Args:
        tflite_path:    Source .tflite model (INT8 quantized)
        output_path:    Destination .nb NPU model
        ksnn_tools_dir: Path to the ksnn_tools directory from the Khadas SDK
    """
    import subprocess
    convert_script = os.path.join(ksnn_tools_dir, "ksnn_convert.py")
    if not os.path.exists(convert_script):
        raise FileNotFoundError(
            f"KSNN convert script not found: {convert_script}\n"
            "Download the KSNN SDK from https://github.com/khadas/ksnn"
        )

    cmd = [
        "python3", convert_script,
        "--input", tflite_path,
        "--output", output_path,
        "--sdk", ksnn_tools_dir,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"KSNN conversion failed:\n{result.stderr}")
    logger.info("KSNN model saved to: %s", output_path)


# ---------------------------------------------------------------------------
# Paper results (§5.3 Figure 7): KSNN NPU vs ARM NN GPU on VIM3
# ---------------------------------------------------------------------------

PAPER_KSNN_RESULTS = {
    "description": (
        "KSNN INT8 vs ARM NN GpuAcc FP16 on Khadas VIM3 (Fig 7, §5.3). "
        "KSNN achieves ~5× speedup."
    ),
    "MobileNetV2": {"ArmNN_GpuAcc_FP16_ms": 26.8, "KSNN_INT8_ms": 5.4,  "speedup": 4.96},
    "ResNet101V2": {"ArmNN_GpuAcc_FP16_ms": 195.0, "KSNN_INT8_ms": 40.2, "speedup": 4.85},
    "DenseNet121": {"ArmNN_GpuAcc_FP16_ms": 182.0, "KSNN_INT8_ms": 37.8, "speedup": 4.81},
    "Xception":    {"ArmNN_GpuAcc_FP16_ms": 310.0, "KSNN_INT8_ms": 63.0, "speedup": 4.92},
}

PAPER_KSNN_ACCURACY = {
    "description": "INT8 top-1 accuracy drop vs TFLite FP32 baseline (ImageNet-100 subset)",
    "MobileNetV2": {"FP32_top1": 71.8, "INT8_top1": 69.9, "drop": 1.9},
    "ResNet101V2": {"FP32_top1": 77.6, "INT8_top1": 76.1, "drop": 1.5},
    "DenseNet121": {"FP32_top1": 75.0, "INT8_top1": 72.8, "drop": 2.2},
}


if __name__ == "__main__":
    print("KSNN Runner (Khadas VIM3 NPU) — Bang for the Buck §5.3")
    print(f"KSNN available: {KSNN_AVAILABLE}")
    print()
    print("Paper: KSNN INT8 NPU vs ARM NN GpuAcc FP16 on VIM3 (~5× speedup):")
    for model, d in PAPER_KSNN_RESULTS.items():
        if isinstance(d, dict) and "speedup" in d:
            print(f"  {model:<14} ArmNN={d['ArmNN_GpuAcc_FP16_ms']:.0f}ms  "
                  f"KSNN={d['KSNN_INT8_ms']:.1f}ms  ({d['speedup']:.2f}×)")
