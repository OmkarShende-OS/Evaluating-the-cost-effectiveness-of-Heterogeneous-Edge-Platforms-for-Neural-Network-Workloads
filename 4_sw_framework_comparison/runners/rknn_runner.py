"""
rknn_runner.py — Rockchip RKNN (Odroid M1 NPU) inference runner (§5.3).

RKNN (Rockchip Neural Network) is the NPU SDK for Rockchip SoCs including
the RK3568 on Odroid M1. It supports INT8 and FP16 inference on the
dedicated Rockchip NPU (0.8 TOPS).

RKNN setup workflow:
  1. Convert model to .rknn format using rknn-toolkit2 on x86:
         from rknn.api import RKNN
         rknn = RKNN()
         rknn.load_tflite("model.tflite")
         rknn.build(do_quantization=True, dataset="calib.txt")
         rknn.export_rknn("model.rknn")
  2. Transfer both model.rknn and rknn_runtime (librknnrt.so) to Odroid M1
  3. Use this runner for inference via rknnlite or rknn_matmul_api

Paper context §5.3:
  RKNN NPU (INT8) is fastest on Odroid M1 — 4.6× faster than ARM NN GPU.
  However, RKNN max INT8 accuracy drop is ~2–3% top-1 vs FP32.
  On Odroid M1: RKNN INT8 is 50% faster than KSNN INT8 on VIM3 per TOPS.
"""

import os
import logging
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from rknnlite.api import RKNNLite
    RKNN_AVAILABLE = True
except ImportError:
    try:
        from rknn.api import RKNN as RKNNLite
        RKNN_AVAILABLE = True
    except ImportError:
        RKNN_AVAILABLE = False
        RKNNLite = None
        logger.debug("RKNN/RKNNLite not available (requires Rockchip SoC with RKNN SDK)")


# Rockchip NPU core count on RK3568 (Odroid M1): 1 core at 0.8 TOPS
_DEFAULT_CORE_MASK = 1  # 0b01 = Core 0 only


class RKNNRunner:
    """
    Inference runner for Rockchip NPU via RKNN Lite Python API.

    Runs only on Rockchip SoC hardware (RK3568, RK3588, etc.)
    The Odroid M1 uses RK3568 with a single NPU core (0.8 TOPS).

    Usage:
        runner = RKNNRunner("mobilenetv2_int8.rknn")
        runner.load()
        logits = runner.infer(input_224x224_rgb)
    """

    def __init__(
        self,
        model_rknn_path: str,
        precision: str = "int8",
        npu_core_mask: int = _DEFAULT_CORE_MASK,
        async_mode: bool = False,
    ):
        """
        Args:
            model_rknn_path: Path to compiled .rknn model file
            precision:       "int8" or "fp16" (fp32 falls back to CPU)
            npu_core_mask:   Bitmask for NPU core assignment (1 = core 0, 3 = cores 0+1)
            async_mode:      Enable async inference (pipeline mode for throughput)
        """
        self.model_path    = model_rknn_path
        self.precision     = precision
        self.npu_core_mask = npu_core_mask
        self.async_mode    = async_mode

        self._rknn = None

    def load(self) -> None:
        """Load model and initialize NPU runtime."""
        if not RKNN_AVAILABLE:
            raise RuntimeError(
                "RKNN Lite not available. This runner requires Rockchip hardware "
                "(Odroid M1, Orange Pi 5, etc.) with rknn-toolkit-lite2 installed.\n"
                "SDK: https://github.com/rockchip-linux/rknn-toolkit2"
            )

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"RKNN model not found: {self.model_path}")

        self._rknn = RKNNLite()
        ret = self._rknn.load_rknn(self.model_path)
        if ret != 0:
            raise RuntimeError(f"RKNN load failed (ret={ret}): {self.model_path}")

        ret = self._rknn.init_runtime(
            core_mask=self.npu_core_mask,
            async_mode=self.async_mode,
        )
        if ret != 0:
            raise RuntimeError(f"RKNN runtime init failed (ret={ret})")

        logger.info("RKNN model loaded: %s  NPU_core=%d", 
                    os.path.basename(self.model_path), self.npu_core_mask)

    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run inference on Rockchip NPU.

        Args:
            input_data: HWC uint8 [0, 255] RGB image (RKNN handles normalization)

        Returns:
            Float32 output array
        """
        if self._rknn is None:
            raise RuntimeError("Call load() first")

        if input_data.dtype != np.uint8:
            input_data = np.clip(input_data, 0, 255).astype(np.uint8)

        outputs = self._rknn.inference(inputs=[input_data])
        return np.array(outputs[0], dtype=np.float32)

    def close(self) -> None:
        if self._rknn is not None:
            self._rknn.release()
            self._rknn = None

    def get_sdk_version(self) -> str:
        """Return RKNN SDK version string."""
        if self._rknn is None:
            raise RuntimeError("Call load() first")
        return self._rknn.get_sdk_version()


# ---------------------------------------------------------------------------
# Model conversion helper (run on x86 with rknn-toolkit2, NOT on the board)
# ---------------------------------------------------------------------------

def convert_tflite_to_rknn(
    tflite_path: str,
    output_rknn_path: str,
    calibration_dataset: Optional[str] = None,
    do_quantize: bool = True,
    target_platform: str = "rk3568",
) -> None:
    """
    Convert TFLite model to RKNN format.

    Must run on x86 Linux with rknn-toolkit2 installed (NOT on the ARM board).

    Args:
        tflite_path:          Source .tflite model
        output_rknn_path:     Destination .rknn model
        calibration_dataset:  Text file with image paths for INT8 calibration
        do_quantize:          Perform INT8 post-training quantization
        target_platform:      "rk3568" (Odroid M1), "rk3588", etc.
    """
    try:
        from rknn.api import RKNN
    except ImportError:
        raise RuntimeError(
            "rknn-toolkit2 not installed.\n"
            "Install on x86: pip install rknn-toolkit2\n"
            "https://github.com/rockchip-linux/rknn-toolkit2"
        )

    rknnobj = RKNN(verbose=False)
    rknnobj.config(
        mean_values=[[128, 128, 128]],
        std_values=[[128, 128, 128]],
        target_platform=target_platform,
    )

    ret = rknnobj.load_tflite(model=tflite_path)
    if ret != 0:
        raise RuntimeError(f"Failed to load TFLite: {tflite_path}")

    ret = rknnobj.build(
        do_quantization=do_quantize,
        dataset=calibration_dataset,
    )
    if ret != 0:
        raise RuntimeError("RKNN build failed")

    ret = rknnobj.export_rknn(output_rknn_path)
    if ret != 0:
        raise RuntimeError(f"RKNN export failed: {output_rknn_path}")

    rknnobj.release()
    logger.info("RKNN model saved: %s", output_rknn_path)


# ---------------------------------------------------------------------------
# Paper results (§5.3): RKNN NPU on Odroid M1
# ---------------------------------------------------------------------------

PAPER_RKNN_RESULTS = {
    "description": (
        "RKNN INT8 inference times on Odroid M1 (RK3568 NPU 0.8 TOPS). "
        "Compared with ARM NN GpuAcc FP16 on the same board."
    ),
    "MobileNetV2": {
        "ArmNN_GpuAcc_FP16_ms": 38.0,
        "RKNN_INT8_ms": 8.2,
        "speedup": 4.63,
    },
    "ResNet101V2": {
        "ArmNN_GpuAcc_FP16_ms": 275.0,
        "RKNN_INT8_ms": 55.0,
        "speedup": 5.00,
    },
    "DenseNet121": {
        "ArmNN_GpuAcc_FP16_ms": 255.0,
        "RKNN_INT8_ms": 52.4,
        "speedup": 4.87,
    },
    "Xception": {
        "ArmNN_GpuAcc_FP16_ms": 430.0,
        "RKNN_INT8_ms": 86.0,
        "speedup": 5.00,
    },
}

PAPER_RKNN_VS_KSNN = {
    "description": "RKNN (M1, 0.8 TOPS) vs KSNN (VIM3, 5 TOPS). M1 slower absolute but better per-TOPS.",
    "note": (
        "VIM3 KSNN has 6.25× the TOPS of M1 RKNN yet only ~1.8× faster. "
        "RKNN INT8 uses NPU more efficiently (higher NPU utilization)."
    ),
}


from typing import Optional

if __name__ == "__main__":
    print("RKNN Runner (Odroid M1 RK3568 NPU) — Bang for the Buck §5.3")
    print(f"RKNN available: {RKNN_AVAILABLE}")
    print()
    print("Paper: RKNN INT8 vs ARM NN GpuAcc FP16 on Odroid M1 (~5× speedup):")
    for model, d in PAPER_RKNN_RESULTS.items():
        if isinstance(d, dict) and "speedup" in d:
            print(f"  {model:<14} ArmNN={d['ArmNN_GpuAcc_FP16_ms']:.0f}ms  "
                  f"RKNN={d['RKNN_INT8_ms']:.1f}ms  ({d['speedup']:.2f}×)")
