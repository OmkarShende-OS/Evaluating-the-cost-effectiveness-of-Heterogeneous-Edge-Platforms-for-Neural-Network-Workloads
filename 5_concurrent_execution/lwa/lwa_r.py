"""
lwa_r.py — LWA-R: Runtime-balanced layer-wise allocation (§6.2).

Strategy: assign layers to proc A vs proc B such that the cumulative
runtime on each side is approximately equal.

Mathematically: find cutpoint k minimising |sum(t[0..k]) − sum(t[k+1..N])|

Paper result:
  LWA-R is 45% SLOWER than solo NCS2 VPU on Odroid H2.
  The runtime-balance optimum still requires a transfer of ~14 MB
  feature map over USB 3.0 (~47ms), which is more than the 9.8ms solo time.
"""

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from layer_wise_allocation import LayerProfile, LWAPartition, LWAStrategyRuntime
except ImportError:
    from lwa.layer_wise_allocation import LayerProfile, LWAPartition, LWAStrategyRuntime


class LWA_R:
    """
    Layer-wise allocation with runtime-balance partitioning.

    Profiles the model, finds the runtime-balanced cutpoint, then splits
    the TFLite subgraphs and runs each on a separate processor.

    Note: On COTS hardware the USB transfer cost renders this slower than solo.
    """

    def __init__(
        self,
        model_path: str,
        proc_a_device: str = "CPU",
        proc_b_device: str = "MYRIAD",
        usb_bandwidth_mbps: float = 300.0,
    ):
        self.model_path         = model_path
        self.proc_a_device      = proc_a_device
        self.proc_b_device      = proc_b_device
        self.usb_bandwidth_mbps = usb_bandwidth_mbps
        self._partition: Optional[LWAPartition] = None

    def profile_and_partition(
        self, layer_profiles: List[LayerProfile]
    ) -> LWAPartition:
        """
        Apply LWA-R strategy to supplied layer profiles.

        Args:
            layer_profiles: Per-layer profile from profiling run

        Returns:
            LWAPartition with runtime-balanced cutpoint
        """
        strategy = LWAStrategyRuntime()
        self._partition = strategy.partition(layer_profiles)
        logger.info(
            "LWA-R partition: cutpoint=%d  A=%.1fms  B=%.1fms  "
            "transfer=%.1fMB (%.1fms)  total_est=%.1fms",
            self._partition.cutpoint_idx,
            self._partition.runtime_a_ms,
            self._partition.runtime_b_ms,
            self._partition.transfer_mb,
            self._partition.transfer_est_ms,
            self._partition.estimated_total_ms,
        )
        return self._partition

    def analyse_vs_paper(self) -> dict:
        """Compare LWA-R partition estimate against paper Table 5 result."""
        from layer_wise_allocation import PAPER_LWA_RESULTS
        paper = PAPER_LWA_RESULTS.get("MobileNetV2", {})
        paper_lwa_r = paper.get("LWA_R_ms", None)
        solo_ncs2   = paper.get("solo_NCS2_FP16_ms", None)

        result = {
            "strategy":          "LWA-R",
            "paper_lwa_r_ms":    paper_lwa_r,
            "solo_ncs2_ms":      solo_ncs2,
            "overhead_vs_solo":  None,
        }
        if paper_lwa_r and solo_ncs2:
            result["overhead_vs_solo"] = f"{(paper_lwa_r/solo_ncs2 - 1)*100:.0f}% slower"
        return result


if __name__ == "__main__":
    from layer_wise_allocation import PAPER_LWA_RESULTS, LWAStrategyRuntime, LayerProfile
    import random

    print("LWA-R (Runtime Balance) — Bang for the Buck §6.2")
    print()
    print("Paper result (MobileNetV2 on H2 + NCS2):")
    d = PAPER_LWA_RESULTS["MobileNetV2"]
    solo = d["solo_NCS2_FP16_ms"]
    lvar = d["LWA_R_ms"]
    print(f"  Solo NCS2: {solo} ms  →  LWA-R: {lvar} ms  ({lvar/solo:.2f}× SLOWER)")
    print()

    # Demo: apply LWA-R to a synthetic model
    random.seed(0)
    layers = [
        LayerProfile(i, f"conv_{i}", "Conv2D",
                     n_params=random.randint(5000, 200000),
                     flops=random.randint(1_000_000, 50_000_000),
                     runtime_ms=random.uniform(0.3, 3.0),
                     output_mb=random.uniform(1.0, 15.0))
        for i in range(15)
    ]

    lwa_r = LWA_R("model.tflite")
    partition = lwa_r.profile_and_partition(layers)
    print(f"Found cutpoint at layer {partition.cutpoint_idx}")
    print(f"  Proc A: {len(partition.layers_a)} layers, {partition.runtime_a_ms:.1f} ms")
    print(f"  Proc B: {len(partition.layers_b)} layers, {partition.runtime_b_ms:.1f} ms")
    print(f"  Transfer: {partition.transfer_mb:.1f} MB  ({partition.transfer_est_ms:.1f} ms at 300 MB/s)")
    print(f"  Total estimated: {partition.estimated_total_ms:.1f} ms")
