"""
roofline_model.py — Construct and analyse the Roofline Model for all 8 edge
platforms in "Bang for the Buck" (SEC '23), Section 4.

The Roofline Model [Williams et al. 2009] bounds the maximum achievable
performance of an application:

    attainable_perf(GFLOPS/s) = min(peak_compute, OI × peak_bandwidth)

This module:
  1. Builds the roofline for each processor.
  2. Computes theoretical performance ordering (Figure 1 from the paper).
  3. Compares theoretical ordering vs. empirical measurements.
  4. Identifies the ridge point (where compute-bound meets memory-bound).
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from hardware_specs import ALL_PLATFORMS, ProcessorSpec, PlatformSpec
from operational_intensity import MODELS, get_oi, PAPER_OI_VALUES


@dataclass
class RooflinePoint:
    """The roofline attainable performance for one (processor, model, precision) triple."""
    processor_name: str
    proc_type: str
    model_name: str
    precision: str
    oi: float                      # Operational Intensity (FLOPS/byte)
    peak_compute_gflops: float
    peak_bandwidth_gbs: float
    attainable_gflops: float       # min(peak_compute, OI × peak_bw)
    ridge_point: float             # OI at which compute == memory bound
    bound: str                     # "memory" or "compute"


def ridge_point(peak_compute: float, peak_bandwidth: float) -> float:
    """
    The Operational Intensity at the ridge:
        OI_ridge = peak_compute / peak_bandwidth   (FLOPS/byte)
    """
    return peak_compute / peak_bandwidth


def attainable_performance(oi: float, peak_compute: float,
                            peak_bandwidth: float) -> Tuple[float, str]:
    """
    Compute the attainable GFLOPS/s for a workload on a processor.

    Returns:
        (attainable_gflops, bound) where bound is "memory" or "compute"
    """
    memory_ceiling = oi * peak_bandwidth
    if memory_ceiling < peak_compute:
        return memory_ceiling, "memory"
    else:
        return peak_compute, "compute"


def build_roofline(model_name: str, precision: str) -> List[RooflinePoint]:
    """
    Build roofline points for all compatible processors for a given
    (model, precision) combination.

    Filters processors by supported precision:
      - CPUs → FP32
      - GPUs → FP16 (or FP32)
      - NPUs → INT8
      - VPUs → FP16
    """
    PRECISION_MAP = {
        "CPU": ["FP32"],
        "GPU": ["FP16", "FP32"],
        "NPU": ["INT8"],
        "VPU": ["FP16"],
    }

    oi = get_oi(model_name, precision)
    points = []

    for platform_name, platform in ALL_PLATFORMS.items():
        for proc in platform.processors:
            if precision not in PRECISION_MAP.get(proc.proc_type, []):
                continue

            perf, bound = attainable_performance(
                oi, proc.peak_compute_gflops, proc.peak_bandwidth_gbs
            )
            rp = ridge_point(proc.peak_compute_gflops, proc.peak_bandwidth_gbs)

            points.append(RooflinePoint(
                processor_name=proc.name,
                proc_type=proc.proc_type,
                model_name=model_name,
                precision=precision,
                oi=oi,
                peak_compute_gflops=proc.peak_compute_gflops,
                peak_bandwidth_gbs=proc.peak_bandwidth_gbs,
                attainable_gflops=perf,
                ridge_point=rp,
                bound=bound,
            ))

    return sorted(points, key=lambda p: p.attainable_gflops, reverse=True)


def roofline_ordering(precision: str) -> Dict[str, List[Tuple[str, float]]]:
    """
    Compute the roofline performance ordering across all processors for
    all models at a given precision.

    Returns a dict: model_name → [(proc_name, relative_perf), ...] sorted
    by relative performance (baseline = Odroid M1 equivalent).
    """
    # Baseline: M1_NPU for INT8, M1_GPU for FP16, M1_CPU for FP32
    baseline_map = {"FP32": "M1_CPU", "FP16": "M1_GPU", "INT8": "M1_NPU"}
    baseline_proc = baseline_map[precision]

    results = {}
    for model_name in MODELS:
        points = build_roofline(model_name, precision)
        if not points:
            continue

        # Find baseline performance
        baseline_perf = next(
            (p.attainable_gflops for p in points if p.processor_name == baseline_proc),
            None
        )
        if baseline_perf is None or baseline_perf == 0:
            continue

        results[model_name] = [
            (p.processor_name, round(p.attainable_gflops / baseline_perf, 2))
            for p in points
        ]

    return results


def roofline_ordering_averaged(precision: str) -> List[Tuple[str, float]]:
    """
    Return processors ranked by their average relative roofline performance
    across all models, matching the paper's tabulated ordering.

    These are the THEORETICAL values. Empirical values differ (see §5.2).
    """
    per_model = roofline_ordering(precision)
    if not per_model:
        return []

    # Accumulate relative performance per processor
    totals: Dict[str, List[float]] = {}
    for model_results in per_model.values():
        for proc_name, rel_perf in model_results:
            totals.setdefault(proc_name, []).append(rel_perf)

    avg: List[Tuple[str, float]] = [
        (proc, round(sum(vals) / len(vals), 2))
        for proc, vals in totals.items()
    ]
    return sorted(avg, key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Paper-published ordering (from Section 4, used for validation)
# ---------------------------------------------------------------------------

PAPER_ROOFLINE_ORDERING = {
    "CPU_FP32": [
        ("AGX_CPU", 2.27), ("TX2_CPU", 1.50), ("H2_CPU", 1.15),
        ("VIM3_CPU_Big", 1.10), ("NX_CPU", 1.05), ("M1_CPU", 1.0),
        ("Nano_CPU", 0.715), ("VIM3_CPU_Little", 0.45)
    ],
    "GPU_FP16": [
        ("AGX_GPU", 117.08), ("NX_GPU", 55.17), ("NCS2_VPU", 23.84),
        ("TX2_GPU", 16.02), ("Nano_GPU", 5.69), ("H2_GPU", 3.47),
        ("AGX_DLA", 3.03), ("VIM3_GPU", 1.73), ("M1_GPU", 1.0)
    ],
    "NPU_INT8": [
        ("AGX_GPU", 24.3), ("NX_GPU", 11.64), ("VIM3_NPU", 5.54),
        ("TX2_GPU", 3.33), ("Nano_GPU", 1.18), ("M1_NPU", 1.0),
        ("VIM3_GPU", 0.36), ("M1_GPU", 0.21)
    ],
}

PAPER_COST_ORDERING = [
    ("AGX",  10.46), ("NX",  5.01), ("TX2", 5.01), ("VIM3", 1.66),
    ("Nano", 1.35),  ("H2",  1.25), ("NCS2", 1.15), ("M1",  1.0)
]


def print_roofline_summary():
    """Print the roofline analysis summary matching the paper's Section 4 output."""
    print("=" * 70)
    print("Roofline Model Summary — Bang for the Buck (SEC '23, §4)")
    print("=" * 70)

    for label, ordering in PAPER_ROOFLINE_ORDERING.items():
        print(f"\n{label} Ordering (Roofline):")
        for rank, (proc, perf) in enumerate(ordering, 1):
            bar = "█" * int(perf)
            print(f"  {rank:2}. {proc:<20} {perf:6.2f}×  {bar}")

    print("\nCost-relative Ordering (baseline = Odroid-M1 = $95.50):")
    for platform, cost_rel in PAPER_COST_ORDERING:
        print(f"  {platform:<6} {cost_rel:.2f}×")

    print("\n⚠  Roofline limitations (why §5 empirical measurements differ):")
    print("  1. Roofline ignores cache effects (actual BW > DRAM BW for cached workloads)")
    print("  2. CPU IO overhead for GPU/NPU not captured (see CPU bottleneck §5.2)")
    print("  3. NVDLA cannot execute all NN layers → falls back to GPU")
    print("  4. SW runtime quality affects real performance (see §5.3)")


if __name__ == "__main__":
    print_roofline_summary()

    print("\n\n--- Computed roofline ordering (CPU FP32) ---")
    for proc, rel in roofline_ordering_averaged("FP32"):
        print(f"  {proc:<20} {rel:.2f}×")
