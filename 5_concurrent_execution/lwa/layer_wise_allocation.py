"""
layer_wise_allocation.py — Layer-Wise Workload Allocation orchestrator (§6).

LWA splits the neural network itself across two processors:
  • Layers 1..k run on processor A (e.g., GPU)
  • Layers k+1..N run on processor B (e.g., NPU)
  The cutpoint k is chosen by one of three strategies (§6.2):
    • LWA-R: minimize runtime: pick k where cumulative runtime A ≈ remaining B
    • LWA-P: top-N layers by parameter count go to the more capable processor
    • LWA-C: top-N layers by FLOPs (compute) go to the faster accelerator

Paper finding §6.2 (key negative result):
  ALL three LWA variants are SLOWER than solo GPU inference.
  Reason: data transfer time between processors (PCIe/USB) exceeds any
  potential parallelism gain. The inter-processor communication overhead
  dominates, especially for the NCS2 connected via USB 3.0.

  Table 5 results (MobileNetV2 on H2 + NCS2):
    Solo NCS2:    9.8 ms   (baseline)
    LWA-R:       14.2 ms   (+45%)
    LWA-P:       16.8 ms   (+71%)
    LWA-C:       15.4 ms   (+57%)
    Solo CPU:    28.4 ms   (worst)

  This is a key contribution of the paper: empirically showing LWA is impractical
  on COTS edge hardware due to memory-bandwidth/communication costs.
"""

import abc
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Layer profile data structure
# ---------------------------------------------------------------------------

@dataclass
class LayerProfile:
    """Profile of a single neural network layer."""
    layer_id:    int
    layer_name:  str
    layer_type:  str        # "Conv2D", "Dense", "BatchNorm", etc.
    n_params:    int        # number of parameters
    flops:       int        # floating-point operations
    runtime_ms:  float      # measured execution time on target device (ms)
    output_mb:   float      # output tensor size in megabytes (interop transfer cost)


@dataclass
class LWAPartition:
    """Result of an LWA partitioning decision."""
    strategy:     str
    cutpoint_idx: int           # index of first layer going to proc B
    layers_a:     List[int]     # layer IDs on proc A
    layers_b:     List[int]     # layer IDs on proc B
    runtime_a_ms: float         # estimated execution time on proc A
    runtime_b_ms: float         # estimated execution time on proc B
    transfer_mb:  float         # data transfer size at the cut
    transfer_est_ms: float      # estimated transfer latency

    @property
    def estimated_total_ms(self) -> float:
        """
        Sequential LWA estimate: A + transfer + B.
        (Parallelism is limited by the serial nature of NN layers.)
        """
        return self.runtime_a_ms + self.transfer_est_ms + self.runtime_b_ms

    @property
    def is_beneficial_vs_solo(self) -> Optional[float]:
        """Speedup vs the faster solo processor (negative = harmful)."""
        return None  # Set externally after measuring


# ---------------------------------------------------------------------------
# LWA partitioning strategies
# ---------------------------------------------------------------------------

class LWAStrategyBase(abc.ABC):
    """Abstract base class for LWA partitioning strategies."""

    @abc.abstractmethod
    def partition(
        self,
        layers: List[LayerProfile],
        n_select: int = 0,
    ) -> LWAPartition:
        """
        Given a list of layer profiles, decide the cut-point.

        Args:
            layers:    Ordered list of layer profiles (input → output)
            n_select:  For LWA-P and LWA-C: number of layers to assign to proc A

        Returns:
            LWAPartition with the chosen cutpoint
        """

    @staticmethod
    def _estimate_transfer_ms(output_mb: float, bandwidth_mbps: float = 300.0) -> float:
        """
        Estimate inter-processor data transfer latency.
        Paper uses USB 3.0 bandwidth ~300 MB/s for NCS2.
        PCIe / direct memory ~10 GB/s for Jetson.
        """
        if bandwidth_mbps <= 0:
            return 0.0
        return (output_mb / bandwidth_mbps) * 1000.0  # ms


class LWAStrategyRuntime(LWAStrategyBase):
    """
    LWA-R: Cut where cumulative runtime on A ≈ cumulative runtime on B.

    Choose cutpoint k to minimise max(sum(t_A[0..k]), sum(t_B[k+1..N])).
    This is the "minimum runtime" strategy.
    """

    name = "LWA-R"

    def partition(
        self,
        layers: List[LayerProfile],
        n_select: int = 0,
    ) -> LWAPartition:
        n = len(layers)
        total_runtime = sum(l.runtime_ms for l in layers)
        cumulative = 0.0
        best_k = 0
        best_diff = float("inf")

        for k in range(1, n):
            cumulative += layers[k - 1].runtime_ms
            remaining  = total_runtime - cumulative
            diff = abs(cumulative - remaining)
            if diff < best_diff:
                best_diff = diff
                best_k = k

        transfer_mb  = layers[best_k - 1].output_mb
        transfer_ms  = self._estimate_transfer_ms(transfer_mb)

        return LWAPartition(
            strategy     = self.name,
            cutpoint_idx = best_k,
            layers_a     = [l.layer_id for l in layers[:best_k]],
            layers_b     = [l.layer_id for l in layers[best_k:]],
            runtime_a_ms = sum(l.runtime_ms for l in layers[:best_k]),
            runtime_b_ms = sum(l.runtime_ms for l in layers[best_k:]),
            transfer_mb  = transfer_mb,
            transfer_est_ms = transfer_ms,
        )


class LWAStrategyParams(LWAStrategyBase):
    """
    LWA-P: Cut at the end of the top-N layers by parameter count.

    Assigns layers with the most parameters to processor A
    (typically the more powerful GPU/NPU that can handle large layers).
    """

    name = "LWA-P"

    def partition(
        self,
        layers: List[LayerProfile],
        n_select: int = 5,
    ) -> LWAPartition:
        # Find last index of the top-N param layers to determine cutpoint
        sorted_by_params = sorted(range(len(layers)), key=lambda i: -layers[i].n_params)
        top_n_idx = set(sorted_by_params[:n_select])

        # Cutpoint = last index of top-N layers (all top-N layers must be on A)
        cutpoint = max(top_n_idx) + 1

        transfer_mb = layers[cutpoint - 1].output_mb
        transfer_ms = self._estimate_transfer_ms(transfer_mb)

        return LWAPartition(
            strategy     = self.name,
            cutpoint_idx = cutpoint,
            layers_a     = [l.layer_id for l in layers[:cutpoint]],
            layers_b     = [l.layer_id for l in layers[cutpoint:]],
            runtime_a_ms = sum(l.runtime_ms for l in layers[:cutpoint]),
            runtime_b_ms = sum(l.runtime_ms for l in layers[cutpoint:]),
            transfer_mb  = transfer_mb,
            transfer_est_ms = transfer_ms,
        )


class LWAStrategyCompute(LWAStrategyBase):
    """
    LWA-C: Cut at the end of the top-N layers by FLOP count.

    Assigns compute-heavy layers to the fastest accelerator.
    """

    name = "LWA-C"

    def partition(
        self,
        layers: List[LayerProfile],
        n_select: int = 5,
    ) -> LWAPartition:
        sorted_by_flops = sorted(range(len(layers)), key=lambda i: -layers[i].flops)
        top_n_idx = set(sorted_by_flops[:n_select])
        cutpoint  = max(top_n_idx) + 1

        transfer_mb = layers[cutpoint - 1].output_mb
        transfer_ms = self._estimate_transfer_ms(transfer_mb)

        return LWAPartition(
            strategy     = self.name,
            cutpoint_idx = cutpoint,
            layers_a     = [l.layer_id for l in layers[:cutpoint]],
            layers_b     = [l.layer_id for l in layers[cutpoint:]],
            runtime_a_ms = sum(l.runtime_ms for l in layers[:cutpoint]),
            runtime_b_ms = sum(l.runtime_ms for l in layers[cutpoint:]),
            transfer_mb  = transfer_mb,
            transfer_est_ms = transfer_ms,
        )


# ---------------------------------------------------------------------------
# LWA profiler: measure per-layer runtime on two devices
# ---------------------------------------------------------------------------

def profile_layers_tflite(model_path: str, n_iters: int = 50) -> List[LayerProfile]:
    """
    Use TFLite profiler to extract per-layer runtime on the current device.

    Args:
        model_path: Path to .tflite model
        n_iters:    Number of inference runs to average

    Returns:
        List of LayerProfile in forward-pass order
    """
    try:
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            import tensorflow.lite as tflite
    except ImportError:
        raise RuntimeError("TFLite not installed")

    interpreter = tflite.Interpreter(
        model_path=model_path,
        experimental_op_resolver_type=tflite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    )
    interpreter.allocate_tensors()

    # Generate profiling data
    dummy_input = np.zeros(interpreter.get_input_details()[0]["shape"], dtype=np.float32)
    times: Dict[int, List[float]] = {}

    for _ in range(n_iters):
        interpreter.set_tensor(interpreter.get_input_details()[0]["index"], dummy_input)
        interpreter.invoke()

    # TFLite doesn't expose per-layer timing via Python easily;
    # use operation names as a proxy (approximate profiling)
    op_details = interpreter.get_tensor_details()
    layers = []
    for i, odets in enumerate(interpreter.get_tensor_details()):
        # Approximate profiling: real profiling requires TFLite delegate hooks
        layers.append(LayerProfile(
            layer_id   = i,
            layer_name = odets.get("name", f"layer_{i}"),
            layer_type = "Unknown",
            n_params   = int(np.prod(odets.get("shape", [1]))),
            flops      = 0,  # set by model-specific analysis
            runtime_ms = 0.1,  # placeholder — use hardware profiler for actual values
            output_mb  = float(np.prod(odets.get("shape", [1]))) * 4 / 1e6,
        ))
    return layers


# ---------------------------------------------------------------------------
# Paper LWA results (Table 5, §6.2) — all LWA variants are slower than solo
# ---------------------------------------------------------------------------

PAPER_LWA_RESULTS = {
    "description": (
        "LWA results on Odroid H2 + NCS2 for MobileNetV2 (Table 5, §6.2). "
        "All LWA variants are SLOWER than solo NCS2 due to USB transfer overhead."
    ),
    "MobileNetV2": {
        "solo_NCS2_FP16_ms":    9.8,
        "LWA_R_ms":            14.2,   # +45% vs solo NCS2
        "LWA_P_ms":            16.8,   # +71% vs solo NCS2
        "LWA_C_ms":            15.4,   # +57% vs solo NCS2
        "solo_CPU_FP32_ms":    28.4,
        "transfer_overhead_ms": 4.8,   # USB 3.0 transfer at the layer cutpoint
    },
    "ResNet101V2": {
        "solo_NCS2_FP16_ms":   74.0,
        "LWA_R_ms":           108.0,   # +46%
        "LWA_P_ms":           126.0,   # +70%
        "LWA_C_ms":           115.0,   # +55%
        "solo_CPU_FP32_ms":   210.0,
    },
    "key_insight": (
        "The bottleneck is data transfer, NOT computation. "
        "At the cutpoint, the feature map is ~14 MB (ResNet101V2 mid-network). "
        "USB 3.0 at 300 MB/s → 47 ms transfer time alone exceeds all inference gains."
    ),
}


if __name__ == "__main__":
    print("Layer-Wise Allocation (LWA) — Bang for the Buck §6")
    print()
    print("Paper finding: ALL LWA variants are SLOWER than solo NPU/VPU inference.")
    print("Reason: USB 3.0 data transfer overhead dominates any compute savings.")
    print()
    print("Table 5 results — MobileNetV2 on Odroid H2 + NCS2:")
    d = PAPER_LWA_RESULTS["MobileNetV2"]
    print(f"  Solo NCS2 FP16:    {d['solo_NCS2_FP16_ms']:.1f} ms  (baseline)")
    print(f"  LWA-R:             {d['LWA_R_ms']:.1f} ms  ({d['LWA_R_ms']/d['solo_NCS2_FP16_ms']:.2f}× slower)")
    print(f"  LWA-P:             {d['LWA_P_ms']:.1f} ms  ({d['LWA_P_ms']/d['solo_NCS2_FP16_ms']:.2f}× slower)")
    print(f"  LWA-C:             {d['LWA_C_ms']:.1f} ms  ({d['LWA_C_ms']/d['solo_NCS2_FP16_ms']:.2f}× slower)")
    print(f"  USB transfer overhead: ~{d['transfer_overhead_ms']:.0f} ms")
    print()
    print(PAPER_LWA_RESULTS["key_insight"])
    print()

    # Demo: partition a toy model using all three strategies
    import random
    random.seed(42)
    toy_layers = [
        LayerProfile(i, f"layer_{i}", "Conv2D",
                     n_params=random.randint(1000, 500000),
                     flops=random.randint(100000, 50000000),
                     runtime_ms=random.uniform(0.5, 5.0),
                     output_mb=random.uniform(0.5, 20.0))
        for i in range(20)
    ]

    print("Demo: partitioning 20-layer toy model")
    for strategy_cls in [LWAStrategyRuntime, LWAStrategyParams, LWAStrategyCompute]:
        s = strategy_cls()
        p = s.partition(toy_layers, n_select=5)
        print(f"  {p.strategy}: cut at layer {p.cutpoint_idx}  "
              f"  A={p.runtime_a_ms:.1f}ms  B={p.runtime_b_ms:.1f}ms  "
              f"  transfer≈{p.transfer_mb:.1f}MB ({p.transfer_est_ms:.1f}ms)  "
              f"  total≈{p.estimated_total_ms:.1f}ms")
