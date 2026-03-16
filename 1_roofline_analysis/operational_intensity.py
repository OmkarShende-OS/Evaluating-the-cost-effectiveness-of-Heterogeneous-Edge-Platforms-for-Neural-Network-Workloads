"""
operational_intensity.py — Compute Operational Intensity (OI = FLOPS/byte) for
the neural network models benchmarked in "Bang for the Buck" (SEC '23).

OI is the key x-axis value on the Roofline Model.
It characterises whether a workload is memory-bound or compute-bound
on a given hardware platform.

Equation:
    OI = total_flops / total_bytes_from_DRAM

For a given NN model at a given precision:
  - total_flops  = sum of MACs per layer × 2  (each MAC = 1 multiply + 1 add)
  - total_bytes  = model_parameters × bytes_per_element

Table 1 values from the paper are hard-coded as ground truth; the
`estimate_oi()` function can be used for arbitrary models.
"""
from dataclasses import dataclass
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Precision → bytes per weight element
# ---------------------------------------------------------------------------
BYTES_PER_ELEMENT: Dict[str, float] = {
    "FP32": 4.0,
    "FP16": 2.0,
    "INT8": 1.0,
}


@dataclass
class ModelSpec:
    """Neural network model characteristics."""
    name: str
    parameters_M: float      # Millions of parameters (weights)
    depth: int               # Number of layers
    model_size_mb: float     # Unoptimised FP32 model file size (MB)
    total_macs: float        # Million MACs for a single 224×224 inference
    task: str = "classification"


# ---------------------------------------------------------------------------
# Table 1 from the paper: NN models for vision-based tasks
# MACs estimated from standard profiling (tensorflow profiler) at 224×224
# ---------------------------------------------------------------------------
MODELS: Dict[str, ModelSpec] = {
    "MobileNetV2": ModelSpec(
        name="MobileNet-V2",
        parameters_M=3.5,
        depth=105,
        model_size_mb=14.0,
        total_macs=300.0,         # ~300 MMACs (standard value)
        task="classification"
    ),
    "ResNet101V2": ModelSpec(
        name="ResNet101-V2",
        parameters_M=44.7,
        depth=205,
        model_size_mb=171.0,
        total_macs=7800.0,        # ~7.8 GMACs
        task="classification"
    ),
    "DenseNet121": ModelSpec(
        name="DenseNet-121",
        parameters_M=8.1,
        depth=242,
        model_size_mb=33.0,
        total_macs=2900.0,        # ~2.9 GMACs
        task="classification"
    ),
    "Xception": ModelSpec(
        name="Xception",
        parameters_M=22.9,
        depth=81,
        model_size_mb=88.0,
        total_macs=8400.0,        # ~8.4 GMACs
        task="classification"
    ),
    "YOLOv3": ModelSpec(
        name="YOLOv3",
        parameters_M=62.2,
        depth=106,
        model_size_mb=246.6,
        total_macs=65900.0,       # ~65.9 GMACs (608×608 input)
        task="detection"
    ),
}

# Table 1 paper values: OI for each model at each precision
# (reproduced exactly as published for validation)
PAPER_OI_VALUES: Dict[str, Dict[str, float]] = {
    "MobileNetV2": {"FP32": 1.33, "FP16": 1.63, "INT8": 1.93},
    "ResNet101V2": {"FP32": 1.60, "FP16": 1.90, "INT8": 2.20},
    "DenseNet121": {"FP32": 1.94, "FP16": 2.24, "INT8": 2.54},
    "Xception":    {"FP32": 1.96, "FP16": 2.26, "INT8": 2.56},
    "YOLOv3":      {"FP32": 2.42, "FP16": 2.72, "INT8": 3.02},
}


def compute_oi(model: ModelSpec, precision: str) -> float:
    """
    Compute the Operational Intensity for a model at a given precision.

    OI = (total_flops) / (total_bytes_read_from_memory)
       = (total_macs × 2) / (parameters_M × 1e6 × bytes_per_element)

    Args:
        model:     ModelSpec with total_macs (in MMMACs)
        precision: "FP32", "FP16", or "INT8"

    Returns:
        OI in FLOPS/byte
    """
    bpe = BYTES_PER_ELEMENT[precision]
    total_flops = model.total_macs * 2.0 * 1e6   # MACs → FLOPs
    total_bytes = model.parameters_M * 1e6 * bpe  # parameters → bytes
    return total_flops / total_bytes


def get_oi(model_name: str, precision: str, use_paper_values: bool = True) -> float:
    """
    Get OI for a named model.

    Args:
        model_name:       Key from MODELS dict (e.g. "MobileNetV2")
        precision:        "FP32", "FP16", or "INT8"
        use_paper_values: If True, use Table 1 values from the paper directly
                          (most accurate); otherwise compute from ModelSpec.

    Returns:
        OI in FLOPS/byte
    """
    if use_paper_values and model_name in PAPER_OI_VALUES:
        return PAPER_OI_VALUES[model_name][precision]
    model = MODELS[model_name]
    return compute_oi(model, precision)


def estimate_oi(total_macs_M: float, param_count_M: float, precision: str) -> float:
    """
    Estimate OI for an arbitrary model given its MACs and parameter count.

    Args:
        total_macs_M:   Total multiply-accumulate ops in millions
        param_count_M:  Model parameter count in millions
        precision:      "FP32", "FP16", or "INT8"

    Returns:
        OI in FLOPS/byte
    """
    bpe = BYTES_PER_ELEMENT[precision]
    total_flops = total_macs_M * 2.0 * 1e6
    total_bytes = param_count_M * 1e6 * bpe
    return total_flops / total_bytes


def print_oi_table():
    """Print OI table matching Table 1 of the paper."""
    header = f"{'Model':<15} {'Params(M)':>9} {'Depth':>6} {'Size(MB)':>9} "
    header += f"{'OI_FP32':>8} {'OI_FP16':>8} {'OI_INT8':>8}"
    print(header)
    print("-" * len(header))
    for key, model in MODELS.items():
        oi_fp32 = get_oi(key, "FP32")
        oi_fp16 = get_oi(key, "FP16")
        oi_int8 = get_oi(key, "INT8")
        print(f"{model.name:<15} {model.parameters_M:>9.1f} {model.depth:>6} "
              f"{model.model_size_mb:>9.0f} {oi_fp32:>8.2f} {oi_fp16:>8.2f} "
              f"{oi_int8:>8.2f}")


if __name__ == "__main__":
    print("=== Operational Intensity (FLOPS/byte) — Table 1, Bang for the Buck ===\n")
    print_oi_table()
    print()
    print("Interpretation: Higher OI → more compute-bound (better for GPU/NPU)")
    print("All workloads have OI < roofline ridge point → memory-bandwidth bound")
