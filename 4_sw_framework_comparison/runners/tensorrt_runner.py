"""
tensorrt_runner.py — NVIDIA TensorRT inference runner (§5.3).

Covers:
  • Engine build from ONNX (.onnx → .engine)
  • FP32, FP16, INT8 precision modes
  • Device-memory I/O via pycuda
  • DLA (Deep Learning Accelerator) target for Jetson AGX/NX

Paper context §5.3:
  TensorRT is the fastest GPU framework on all four Jetson boards.
  FP16 gives a 1.6–2.1× speedup over FP32 with negligible accuracy loss.
  INT8 calibration further improves by 1.3–1.5× vs FP16 on Jetson AGX.
  AutoScheduler (Apache TVM) matches TensorRT FP16 on Jetson TX2 (Figure 8).
  DLA: 3–8× slower than GPU but 40% lower power — good for thermal headroom.
"""

import os
import logging
import numpy as np
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# TensorRT import guarded — not available on non-NVIDIA platforms
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401
    TRT_AVAILABLE = True
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
except ImportError:
    TRT_AVAILABLE = False
    TRT_LOGGER = None
    logger.debug("TensorRT / pycuda not available on this platform")


class TensorRTRunner:
    """
    Inference runner for NVIDIA TensorRT optimized engines.

    Supports:
      - Build engine from ONNX (FP32 / FP16 / INT8)
      - Load pre-built .engine files
      - DLA execution (Jetson AGX Xavier / NX)

    Usage:
        runner = TensorRTRunner("model.onnx", precision="fp16")
        runner.load()
        output = runner.infer(input_array)
    """

    def __init__(
        self,
        model_path: str,
        precision: str = "fp16",
        engine_cache_path: Optional[str] = None,
        dla_core: Optional[int] = None,
        max_workspace_mb: int = 2048,
        calibrator=None,
    ):
        """
        Args:
            model_path:          .onnx or .engine file path
            precision:           "fp32", "fp16", or "int8"
            engine_cache_path:   Path to save/load compiled .engine (avoids rebuild)
            dla_core:            0 or 1 to enable DLA (Jetson only), None = GPU
            max_workspace_mb:    TRT builder workspace limit in MB
            calibrator:          INT8 calibration dataset (trt.IInt8Calibrator subclass)
        """
        self.model_path        = model_path
        self.precision         = precision.lower()
        self.engine_cache_path = engine_cache_path
        self.dla_core          = dla_core
        self.max_workspace_mb  = max_workspace_mb
        self.calibrator        = calibrator

        self._engine   = None
        self._context  = None
        self._bindings: List = []
        self._device_mems: List = []
        self._host_inputs: List  = []
        self._host_outputs: List = []
        self._stream       = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load or build engine, prepare CUDA buffers."""
        if not TRT_AVAILABLE:
            raise RuntimeError(
                "TensorRT is not installed. Available only on NVIDIA Jetson / CUDA platforms."
            )

        engine_path = self._engine_path()
        if engine_path and os.path.exists(engine_path):
            self._engine = self._load_engine(engine_path)
            logger.info("Loaded cached engine: %s", engine_path)
        elif self.model_path.endswith(".engine"):
            self._engine = self._load_engine(self.model_path)
        elif self.model_path.endswith((".onnx", ".pb")):
            self._engine = self._build_engine(self.model_path)
            if engine_path:
                self._save_engine(self._engine, engine_path)
        else:
            raise ValueError(f"Unsupported model format: {self.model_path}")

        self._context = self._engine.create_execution_context()
        self._allocate_buffers()
        self._stream = cuda.Stream()
        logger.info(
            "TensorRT engine ready — precision=%s  DLA=%s",
            self.precision,
            self.dla_core if self.dla_core is not None else "GPU",
        )

    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run one inference on the GPU.

        Args:
            input_data: Preprocessed NumPy array (NCHW or NHWC depending on model)

        Returns:
            Output tensor as NumPy array
        """
        if self._context is None:
            raise RuntimeError("Call load() before infer()")

        np.copyto(self._host_inputs[0], input_data.ravel())
        cuda.memcpy_htod_async(self._device_mems[0], self._host_inputs[0], self._stream)
        self._context.execute_async_v2(self._bindings, stream_handle=self._stream.handle)
        cuda.memcpy_dtoh_async(
            self._host_outputs[0], self._device_mems[self._n_inputs], self._stream
        )
        self._stream.synchronize()
        return self._host_outputs[0].reshape(self._output_shape)

    def close(self) -> None:
        """Release GPU memory."""
        for mem in self._device_mems:
            try:
                mem.free()
            except Exception:
                pass
        self._device_mems.clear()
        self._bindings.clear()
        self._context = None
        self._engine  = None

    # ------------------------------------------------------------------
    # Internal engine building / loading
    # ------------------------------------------------------------------

    def _engine_path(self) -> Optional[str]:
        if self.engine_cache_path:
            return self.engine_cache_path
        base = os.path.splitext(self.model_path)[0]
        return f"{base}_{self.precision}.engine"

    def _load_engine(self, path: str):
        with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def _save_engine(self, engine, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            f.write(engine.serialize())
        logger.info("Engine saved to %s", path)

    def _build_engine(self, onnx_path: str):
        builder = trt.Builder(TRT_LOGGER)
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
        config  = builder.create_builder_config()
        config.max_workspace_size = self.max_workspace_mb * (1 << 20)

        parser = trt.OnnxParser(network, TRT_LOGGER)
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
                raise RuntimeError(f"ONNX parse failed:\n{errors}")

        if self.precision == "fp16":
            if not builder.platform_has_fast_fp16:
                logger.warning("FP16 not natively supported, fallback to FP32")
            else:
                config.set_flag(trt.BuilderFlag.FP16)

        elif self.precision == "int8":
            if not builder.platform_has_fast_int8:
                logger.warning("INT8 not natively supported, fallback to FP16")
                config.set_flag(trt.BuilderFlag.FP16)
            else:
                config.set_flag(trt.BuilderFlag.INT8)
                if self.calibrator is not None:
                    config.int8_calibrator = self.calibrator

        if self.dla_core is not None:
            config.default_device_type = trt.DeviceType.DLA
            config.DLA_core = self.dla_core
            config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
            logger.info("Building for DLA core %d", self.dla_core)

        return builder.build_engine(network, config)

    def _allocate_buffers(self) -> None:
        self._n_inputs = 0
        for i in range(self._engine.num_bindings):
            shape  = tuple(self._engine.get_binding_shape(i))
            dtype  = trt.nptype(self._engine.get_binding_dtype(i))
            size   = int(np.prod(shape)) * np.dtype(dtype).itemsize
            d_mem  = cuda.mem_alloc(size)
            h_mem  = cuda.pagelocked_empty(int(np.prod(shape)), dtype)
            self._device_mems.append(d_mem)
            self._bindings.append(int(d_mem))
            if self._engine.binding_is_input(i):
                self._host_inputs.append(h_mem)
                self._n_inputs += 1
            else:
                self._host_outputs.append(h_mem)
                self._output_shape = shape


# ---------------------------------------------------------------------------
# Paper results — TensorRT FP16 inference times on Jetson boards (Figure 6)
# ---------------------------------------------------------------------------

PAPER_TENSORRT_RESULTS = {
    "description": "TensorRT FP16 median inference time (ms) per model per Jetson board (§5.3)",
    "MobileNetV2": {"AGX_GPU": 1.26, "NX_GPU": 2.11,  "TX2_GPU": 4.90,  "Nano_GPU": 14.80},
    "ResNet101V2": {"AGX_GPU": 7.32, "NX_GPU": 11.00, "TX2_GPU": 26.10, "Nano_GPU": 78.20},
    "DenseNet121": {"AGX_GPU": 6.89, "NX_GPU": 10.10, "TX2_GPU": 23.80, "Nano_GPU": 70.90},
    "Xception":    {"AGX_GPU": 5.10, "NX_GPU":  7.80,  "TX2_GPU": 18.70, "Nano_GPU": 54.40},
    "YOLOv3":      {"AGX_GPU": 12.0, "NX_GPU": 18.00, "TX2_GPU": 42.00, "Nano_GPU": 128.0},
}

PAPER_DLA_VS_GPU = {
    "description": "DLA vs GPU relative performance on Jetson AGX (FP16). DLA uses less power.",
    "MobileNetV2": {"GPU_FP16_ms": 1.26, "DLA_FP16_ms": 3.72, "DLA_power_saving_pct": 38},
    "ResNet101V2": {"GPU_FP16_ms": 7.32, "DLA_FP16_ms": 22.1, "DLA_power_saving_pct": 41},
}


if __name__ == "__main__":
    print("TensorRT Runner — Bang for the Buck §5.3")
    print(f"TensorRT available: {TRT_AVAILABLE}")
    print()
    print("Paper FP16 inference times on Jetson AGX Xavier (ms):")
    for model, vals in PAPER_TENSORRT_RESULTS.items():
        if isinstance(vals, dict) and "AGX_GPU" in vals:
            print(f"  {model:<14} AGX={vals['AGX_GPU']:.2f}  NX={vals['NX_GPU']:.2f}  "
                  f"TX2={vals['TX2_GPU']:.2f}  Nano={vals['Nano_GPU']:.2f}")
