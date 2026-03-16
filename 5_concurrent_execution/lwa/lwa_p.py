"""
lwa_p.py — LWA-P: Parameter-count layer-wise allocation (§6.2).

Strategy: assign the top-N layers by parameter count to processor A
(the more capable device — GPU/NPU), and the rest to processor B.

Rationale: layers with more parameters tend to be higher FLOP layers
and benefit more from the accelerator.

Paper result:
  LWA-P is the WORST performer among the three LWA strategies.
  71% slower than solo NCS2 on Odroid H2 (MobileNetV2).
  The top-N param layers are concentrated in the middle of the network →
  the cutpoint is in a region with large feature maps → large USB transfer.
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

try:
    from layer_wise_allocation import LayerProfile, LWAPartition, LWAStrategyParams
except ImportError:
    from lwa.layer_wise_allocation import LayerProfile, LWAPartition, LWAStrategyParams


class LWA_P:
    """
    Layer-wise allocation with top-N parameter-count partitioning.

    The top-N layers with most parameters are placed on proc A.
    The cutpoint is set at the last of the top-N layers.
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
        """Apply LWA-P to supplied layer profiles."""
        strategy = LWAStrategyParams()
        self._partition = strategy.partition(layer_profiles, n_select=self.n_top_layers)
        logger.info(
            "LWA-P partition: cutpoint=%d  top-%d param layers  "
            "transfer=%.1fMB (%.1fms)",
            self._partition.cutpoint_idx,
            self.n_top_layers,
            self._partition.transfer_mb,
            self._partition.transfer_est_ms,
        )
        return self._partition

    def why_lwa_p_underperforms(self) -> str:
        """Return paper explanation for why LWA-P is the worst strategy."""
        return (
            "LWA-P places high-parameter layers (often mid-network conv blocks) on proc A. "
            "This forces the cutpoint to be in the middle of the feature extraction pipeline, "
            "where output tensors are large (e.g., 56×56×256 = 3.2MB for ResNet). "
            "The 14-22 MB transfer over USB 3.0 at 300 MB/s takes 47-77 ms — "
            "far exceeding the 9.8ms solo inference time of the NCS2."
        )


if __name__ == "__main__":
    from layer_wise_allocation import PAPER_LWA_RESULTS, LayerProfile
    import random

    print("LWA-P (Parameter Balance) — Bang for the Buck §6.2")
    print()
    print("Paper result (MobileNetV2 on H2 + NCS2):")
    d = PAPER_LWA_RESULTS["MobileNetV2"]
    solo = d["solo_NCS2_FP16_ms"]
    lwap = d["LWA_P_ms"]
    print(f"  Solo NCS2: {solo} ms  →  LWA-P: {lwap} ms  ({lwap/solo:.2f}× SLOWER)")
    print()
    lwa = LWA_P("model.tflite", n_top_layers=5)
    print("Why LWA-P underperforms:")
    print(f"  {lwa.why_lwa_p_underperforms()}")
    print()

    random.seed(1)
    layers = [
        LayerProfile(i, f"conv_{i}", "Conv2D",
                     n_params=random.randint(5000, 1_000_000),
                     flops=random.randint(1_000_000, 50_000_000),
                     runtime_ms=random.uniform(0.3, 3.0),
                     output_mb=random.uniform(1.0, 20.0))
        for i in range(15)
    ]

    partition = lwa.profile_and_partition(layers)
    top5 = sorted(range(len(layers)), key=lambda i: -layers[i].n_params)[:5]
    print(f"Top-5 param layers: {sorted(top5)}")
    print(f"LWA-P cutpoint: layer {partition.cutpoint_idx}")
    print(f"  Transfer at cutpoint: {partition.transfer_mb:.1f} MB = {partition.transfer_est_ms:.1f} ms")
    print(f"  Total estimated: {partition.estimated_total_ms:.1f} ms")
