"""
lwa_c.py — LWA-C: FLOPs-based layer-wise allocation (§6.2).

Strategy: assign the top-N most compute-intensive layers (by FLOP count) to
processor A (the fast accelerator), and remaining layers to processor B.

This is the "compute greedy" approach — maximize utilization of the fast processor.

Paper result:
  LWA-C achieves 57% overhead vs solo NCS2 (better than LWA-P, worse than LWA-R).
  Its cutpoint tends to fall later in the network (near the final conv blocks
  which are most computationally intensive), resulting in smaller feature maps
  to transfer — slightly less USB overhead than LWA-P.
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

try:
    from layer_wise_allocation import LayerProfile, LWAPartition, LWAStrategyCompute
except ImportError:
    from lwa.layer_wise_allocation import LayerProfile, LWAPartition, LWAStrategyCompute


class LWA_C:
    """
    Layer-wise allocation with top-N FLOP-count partitioning.

    Assigns top-N compute-heaviest layers to proc A (accelerator).
    Generally performs better than LWA-P since compute-heavy layers
    (large conv blocks) tend to occur later in the network where
    feature maps are smaller — reducing transfer overhead.

    Still worse than solo inference due to USB data movement.
    """

    def __init__(
        self,
        model_path: str,
        n_top_layers: int = 5,
        proc_a_device: str = "GPU",
        proc_b_device: str = "CPU",
        usb_bandwidth_mbps: float = 300.0,
    ):
        self.model_path         = model_path
        self.n_top_layers       = n_top_layers
        self.proc_a_device      = proc_a_device
        self.proc_b_device      = proc_b_device
        self.usb_bandwidth_mbps = usb_bandwidth_mbps
        self._partition: Optional[LWAPartition] = None

    def profile_and_partition(
        self, layer_profiles: List[LayerProfile]
    ) -> LWAPartition:
        """Apply LWA-C strategy."""
        strategy = LWAStrategyCompute()
        self._partition = strategy.partition(layer_profiles, n_select=self.n_top_layers)
        logger.info(
            "LWA-C partition: cutpoint=%d  top-%d FLOP layers  "
            "total_flops_a=%d  transfer=%.1fMB (%.1fms)",
            self._partition.cutpoint_idx,
            self.n_top_layers,
            sum(l.flops for l in layer_profiles[:self._partition.cutpoint_idx]),
            self._partition.transfer_mb,
            self._partition.transfer_est_ms,
        )
        return self._partition

    def compare_strategies(
        self, layer_profiles: List[LayerProfile]
    ) -> dict:
        """
        Compare all three LWA strategies on the same layer profiles.
        Returns dict with estimated total latency for each strategy.
        """
        from layer_wise_allocation import (
            LWAStrategyRuntime, LWAStrategyParams, LWAStrategyCompute
        )
        results = {}
        for name, cls in [("LWA-R", LWAStrategyRuntime),
                           ("LWA-P", LWAStrategyParams),
                           ("LWA-C", LWAStrategyCompute)]:
            p = cls().partition(layer_profiles, n_select=self.n_top_layers)
            results[name] = {
                "cutpoint":      p.cutpoint_idx,
                "runtime_a_ms":  round(p.runtime_a_ms, 2),
                "runtime_b_ms":  round(p.runtime_b_ms, 2),
                "transfer_mb":   round(p.transfer_mb, 2),
                "transfer_ms":   round(p.transfer_est_ms, 2),
                "total_est_ms":  round(p.estimated_total_ms, 2),
            }
        return results


if __name__ == "__main__":
    from layer_wise_allocation import PAPER_LWA_RESULTS, LayerProfile
    import random

    print("LWA-C (Compute Balance) — Bang for the Buck §6.2")
    print()
    print("Paper result comparison (MobileNetV2 on H2 + NCS2):")
    d = PAPER_LWA_RESULTS["MobileNetV2"]
    solo = d["solo_NCS2_FP16_ms"]
    for strat, ms_key in [("LWA-R", "LWA_R_ms"), ("LWA-P", "LWA_P_ms"), ("LWA-C", "LWA_C_ms")]:
        ms = d[ms_key]
        print(f"  {strat}: {ms:.1f} ms  ({ms/solo:.2f}× vs solo NCS2={solo:.1f}ms)")
    print()

    random.seed(42)
    layers = [
        LayerProfile(i, f"conv_{i}", "Conv2D",
                     n_params=random.randint(5000, 500_000),
                     flops=random.randint(1_000_000, 100_000_000),
                     runtime_ms=random.uniform(0.3, 3.0),
                     output_mb=random.uniform(0.5, 15.0))
        for i in range(15)
    ]

    lwa_c = LWA_C("model.tflite", n_top_layers=5)
    comparison = lwa_c.compare_strategies(layers)
    print("Strategy comparison on toy model:")
    print(f"  {'Strategy':<10} {'Cut':>5} {'A_ms':>8} {'B_ms':>8} {'Xfer_ms':>8} {'Total_ms':>10}")
    for strat, vals in comparison.items():
        print(f"  {strat:<10} {vals['cutpoint']:>5} {vals['runtime_a_ms']:>8.1f} "
              f"{vals['runtime_b_ms']:>8.1f} {vals['transfer_ms']:>8.1f} {vals['total_est_ms']:>10.1f}")
    best = min(comparison, key=lambda k: comparison[k]["total_est_ms"])
    print(f"\n  Best strategy: {best} ({comparison[best]['total_est_ms']:.1f} ms)")
    print("  (But all are worse than solo inference on competent accelerator!)")
