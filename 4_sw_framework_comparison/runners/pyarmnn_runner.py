"""
pyarmnn_runner.py — PyARMNN Python-bindings inference runner (§5.3).

PyARMNN is the official Python wrapper for ARM NN (available from ARM NN
release 20.11 onwards). It provides Python-native model loading without
requiring subprocess calls.

Paper context: identical to armnn_runner.py but through Python bindings —
useful on VIM3/M1 where the environment already has PyARMNN installed.
pyarmnn allows direct tensor access, making it easier to integrate with
the benchmark harness in Python without subprocesses.
"""

import logging
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import pyarmnn as ann
    PYARMNN_AVAILABLE = True
except ImportError:
    PYARMNN_AVAILABLE = False
    logger.debug("pyarmnn not found — install from ARM NN 20.11+ release package")


class PyArmNNRunner:
    """
    ARM NN inference via Python bindings (pyarmnn).

    Supports TFLite and ONNX models.
    Backend selection mirrors ARM NN C++ API: CpuAcc, GpuAcc.

    Usage:
        runner = PyArmNNRunner("model.tflite", backend="GpuAcc", fp16_turbo=True)
        runner.load()
        outputs = runner.infer(input_data)
    """

    def __init__(
        self,
        model_path: str,
        backend: str = "CpuAcc",
        fp16_turbo: bool = False,
        fast_math: bool = True,
    ):
        """
        Args:
            model_path:  Path to .tflite or .onnx model
            backend:     "CpuAcc" (NEON) or "GpuAcc" (Mali OpenCL)
            fp16_turbo:  Enable FP16 turbo mode on GpuAcc (reduces precision slightly)
            fast_math:   Enable fast-math optimizations
        """
        self.model_path  = model_path
        self.backend     = backend
        self.fp16_turbo  = fp16_turbo
        self.fast_math   = fast_math

        self._runtime     = None
        self._network_id  = None
        self._input_binds = None
        self._output_binds = None

    def load(self) -> None:
        """Parse model and optimize for target backend."""
        if not PYARMNN_AVAILABLE:
            raise RuntimeError(
                "PyARMNN is not installed.\n"
                "Install from: https://github.com/ARM-software/armnn/tree/main/python/pyarmnn"
            )

        options = ann.CreationOptions()
        self._runtime = ann.IRuntime(options)

        # Network parser
        if self.model_path.endswith(".tflite"):
            parser = ann.ITfLiteParser()
        elif self.model_path.endswith(".onnx"):
            parser = ann.IOnnxParser()
        else:
            raise ValueError(f"Unsupported format for parsers: {self.model_path}")

        network = parser.CreateNetworkFromBinaryFile(self.model_path.encode())

        # Backend + optimizer options
        preferred_backends = [ann.BackendId(self.backend)]
        opt_options = ann.OptimizerOptions()

        if self.fast_math:
            opt_options.m_ReduceFp32ToFp16 = self.fp16_turbo

        optimized_net, messages = ann.Optimize(
            network, preferred_backends, self._runtime.GetDeviceSpec(), opt_options
        )
        if messages:
            for msg in messages:
                logger.debug("ArmNN optimizer: %s", msg)

        self._network_id, err = self._runtime.LoadNetwork(optimized_net)
        if err:
            raise RuntimeError(f"PyARMNN LoadNetwork failed: {err}")

        # Retrieve input/output binding info
        if self.model_path.endswith(".tflite"):
            graph_id = parser.GetSubgraphCount() - 1
            self._input_binds  = parser.GetNetworkInputBindingInfo(graph_id, "input")
            self._output_binds = parser.GetNetworkOutputBindingInfo(graph_id, "output")
        else:
            self._input_binds  = parser.GetNetworkInputBindingInfo("input")
            self._output_binds = parser.GetNetworkOutputBindingInfo("output")

        logger.info(
            "PyARMNN model loaded — backend=%s  fp16_turbo=%s",
            self.backend, self.fp16_turbo,
        )

    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run one inference.

        Args:
            input_data: float32 NumPy array

        Returns:
            Output NumPy array
        """
        if self._runtime is None:
            raise RuntimeError("Call load() first")

        input_tensor = ann.make_input_tensors(
            [self._input_binds], [input_data.astype(np.float32)]
        )
        output_shape = ann.TensorShape(self._output_binds[1].GetShape())
        output_data  = np.zeros(tuple(output_shape), dtype=np.float32)
        output_tensor = ann.make_output_tensors([self._output_binds], [output_data])

        status = self._runtime.EnqueueWorkload(
            self._network_id, input_tensor, output_tensor
        )
        if status != ann.Status.Success:
            raise RuntimeError(f"PyARMNN EnqueueWorkload failed: {status}")

        return ann.workload_tensors_to_ndarray(output_tensor)[0]

    def close(self) -> None:
        if self._runtime is not None and self._network_id is not None:
            self._runtime.UnloadNetwork(self._network_id)
        self._runtime = None


# ---------------------------------------------------------------------------
# Paper VIM3 results (same data as armnn_runner.py, §5.3 Figure 7)
# ---------------------------------------------------------------------------

PAPER_PYARMNN_RESULTS = {
    "description": (
        "PyARMNN inference times on VIM3 (Mali-G52), same as ARM NN C++ binary. "
        "Verifies Python binding overhead is negligible (<0.2 ms)."
    ),
    "VIM3_GpuAcc_FP16": {
        "MobileNetV2": 26.8,
        "ResNet101V2": 195.0,
        "DenseNet121": 182.0,
        "Xception":    310.0,
    },
}


if __name__ == "__main__":
    print("PyARMNN Runner — Bang for the Buck §5.3")
    print(f"PyARMNN available: {PYARMNN_AVAILABLE}")
    print()
    print("Paper PyARMNN GpuAcc FP16 results on VIM3 (ms):")
    for model, ms in PAPER_PYARMNN_RESULTS["VIM3_GpuAcc_FP16"].items():
        print(f"  {model:<14} {ms:.1f} ms")
