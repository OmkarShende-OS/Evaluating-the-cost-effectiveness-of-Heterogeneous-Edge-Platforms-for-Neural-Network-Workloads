"""
plot_roofline.py — Reproduce Figure 1 from "Bang for the Buck" (SEC '23).

Generates three roofline plots:
  (a) CPU roofline for FP32
  (b) GPU and NN accelerator roofline for FP16
  (c) NPU/GPU roofline for INT8

Each plot shows:
  - Horizontal "compute roof" lines per processor
  - Sloped "memory bandwidth roof" lines
  - Vertical dashed markers for each NN model's OI
  - Ridge points where compute roof meets memory roof

Usage:
    python plot_roofline.py                  # saves figure to ../results/
    python plot_roofline.py --show           # opens interactive window
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from hardware_specs import ALL_PLATFORMS, ProcessorSpec
from operational_intensity import MODELS, get_oi, PAPER_OI_VALUES
from roofline_model import PAPER_ROOFLINE_ORDERING, ridge_point

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Processor groups for the three subplots
# ---------------------------------------------------------------------------
CPU_PROCS = [
    ("AGX_CPU",          "AGX",    ALL_PLATFORMS["AGX"].processors[0]),
    ("NX_CPU",           "NX",     ALL_PLATFORMS["NX"].processors[0]),
    ("TX2_CPU",          "TX2",    ALL_PLATFORMS["TX2"].processors[0]),
    ("Nano_CPU",         "Nano",   ALL_PLATFORMS["Nano"].processors[0]),
    ("VIM3_CPU_Big",     "VIM3-B", ALL_PLATFORMS["VIM3"].processors[0]),
    ("VIM3_CPU_Little",  "VIM3-L", ALL_PLATFORMS["VIM3"].processors[1]),
    ("H2_CPU",           "H2",     ALL_PLATFORMS["H2"].processors[0]),
    ("M1_CPU",           "M1",     ALL_PLATFORMS["M1"].processors[0]),
]

GPU_PROCS = [
    ("AGX_GPU",   "AGX",  ALL_PLATFORMS["AGX"].processors[1]),
    ("NX_GPU",    "NX",   ALL_PLATFORMS["NX"].processors[1]),
    ("TX2_GPU",   "TX2",  ALL_PLATFORMS["TX2"].processors[1]),
    ("Nano_GPU",  "Nano", ALL_PLATFORMS["Nano"].processors[1]),
    ("VIM3_GPU",  "VIM3", ALL_PLATFORMS["VIM3"].processors[2]),
    ("H2_GPU",    "H2",   ALL_PLATFORMS["H2"].processors[1]),
    ("NCS2_VPU",  "NCS2", ALL_PLATFORMS["NCS2"].processors[0]),
    ("M1_GPU",    "M1",   ALL_PLATFORMS["M1"].processors[1]),
]

NPU_PROCS = [
    ("AGX_GPU",   "AGX-GPU",   ALL_PLATFORMS["AGX"].processors[1]),   # INT8 on Volta
    ("NX_GPU",    "NX-GPU",    ALL_PLATFORMS["NX"].processors[1]),
    ("TX2_GPU",   "TX2-GPU",   ALL_PLATFORMS["TX2"].processors[1]),
    ("Nano_GPU",  "Nano-GPU",  ALL_PLATFORMS["Nano"].processors[1]),
    ("VIM3_NPU",  "VIM3-NPU",  ALL_PLATFORMS["VIM3"].processors[3]),
    ("VIM3_GPU",  "VIM3-GPU",  ALL_PLATFORMS["VIM3"].processors[2]),
    ("M1_NPU",    "M1-NPU",    ALL_PLATFORMS["M1"].processors[2]),
    ("M1_GPU",    "M1-GPU",    ALL_PLATFORMS["M1"].processors[1]),
]

# OI values for all models (average across FP32/FP16/INT8)
MODEL_OI_FP32  = {m: get_oi(m, "FP32") for m in MODELS}
MODEL_OI_FP16  = {m: get_oi(m, "FP16") for m in MODELS}
MODEL_OI_INT8  = {m: get_oi(m, "INT8") for m in MODELS}

COLORS = plt.cm.tab10.colors if MATPLOTLIB_AVAILABLE else []


def _plot_single_roofline(ax, proc_list, model_oi_dict: dict, precision: str,
                           title: str):
    """
    Draw one roofline subplot.

    Args:
        ax:              matplotlib Axes
        proc_list:       list of (label, short_name, ProcessorSpec)
        model_oi_dict:   {model_name: OI_value}
        precision:       "FP32" / "FP16" / "INT8"
        title:           subplot title
    """
    oi_range = np.logspace(-2, 3, 400)  # x-axis OI range

    for i, (label, short, proc) in enumerate(proc_list):
        color = COLORS[i % len(COLORS)]
        pc = proc.peak_compute_gflops
        bw = proc.peak_bandwidth_gbs
        rp = ridge_point(pc, bw)

        # Roofline = min(memory_roof, compute_roof)
        y = np.minimum(bw * oi_range, pc)
        ax.loglog(oi_range, y, color=color, linewidth=1.8, label=label)

        # Mark ridge point
        ax.axvline(x=rp, color=color, linestyle=":", alpha=0.4, linewidth=0.8)

    # Mark model OI positions
    legend_patches = []
    for j, (model_name, oi_val) in enumerate(model_oi_dict.items()):
        ax.axvline(x=oi_val, color="black", linestyle="--",
                   alpha=0.5 + j * 0.1, linewidth=1.2)
        ax.text(oi_val * 1.05, ax.get_ylim()[0] * 1.5,
                MODELS[model_name].name.split("-")[0],
                rotation=90, fontsize=7, color="black", alpha=0.75)

    ax.set_xlabel("Operational Intensity (FLOPS/byte)", fontsize=9)
    ax.set_ylabel("Attainable Performance (GFLOPS/s)", fontsize=9)
    ax.set_title(f"({title}) {precision} Roofline", fontsize=10, fontweight="bold")
    ax.legend(fontsize=6, loc="upper left", ncol=2)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)


def plot_roofline(show: bool = False, save_path: str = None):
    """Generate and optionally save the three-panel roofline figure."""
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not installed — run:  pip install matplotlib")
        _print_ascii_roofline()
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "Roofline Model for Edge Platforms (Bang for the Buck, SEC '23 — Figure 1)",
        fontsize=13, fontweight="bold"
    )

    _plot_single_roofline(axes[0], CPU_PROCS,  MODEL_OI_FP32, "FP32", "a")
    _plot_single_roofline(axes[1], GPU_PROCS,  MODEL_OI_FP16, "FP16", "b")
    _plot_single_roofline(axes[2], NPU_PROCS,  MODEL_OI_INT8, "INT8", "c")

    plt.tight_layout()

    if save_path is None:
        results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
        os.makedirs(results_dir, exist_ok=True)
        save_path = os.path.join(results_dir, "fig1_roofline.png")

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Roofline figure saved → {save_path}")

    if show:
        plt.show()


def _print_ascii_roofline():
    """Fallback ASCII representation when matplotlib is unavailable."""
    print("\n=== Roofline Performance Ordering (paper §4) ===")
    for label, ordering in PAPER_ROOFLINE_ORDERING.items():
        print(f"\n  {label}:")
        max_val = ordering[0][1]
        for proc, rel in ordering:
            bar_len = int(50 * rel / max_val)
            bar = "▓" * bar_len
            print(f"    {proc:<22} {rel:6.2f}×  {bar}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot roofline model (Figure 1)")
    parser.add_argument("--show", action="store_true",
                        help="Open interactive matplotlib window")
    parser.add_argument("--out", type=str, default=None,
                        help="Output path for the figure")
    args = parser.parse_args()

    plot_roofline(show=args.show, save_path=args.out)
