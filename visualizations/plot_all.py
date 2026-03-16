"""
plot_all.py — Reproduce all Bang for the Buck paper figures using matplotlib.

Figures generated:
  Fig 1  — Roofline model (3-panel: CPU FP32, GPU FP16, INT8)
  Fig 2  — Inference time ranking (normalized)
  Fig 3  — Total end-to-end time ranking (normalized)
  Fig 4  — Energy per inference (J)
  Fig 5  — Temperature delta under load (°C)
  Fig 6  — SW framework comparison on Odroid H2
  Fig 7  — SW framework comparison on Khadas VIM3
  Fig 8  — AutoScheduler vs TensorRT on Jetson TX2
  Fig 9  — DWA results on Odroid H2 (CPU + NCS2)
  Fig 10 — DWA results on Khadas VIM3 (Big + KSNN)
  Fig 11 — EDCP metric ranking (bar chart)
  Fig 12 — Adaptive scheduler workflow and results

Usage:
  python plot_all.py             # Save all figures to figures/ directory
  python plot_all.py --show      # Show interactively
  python plot_all.py --fig 1     # Plot only figure 1
"""

import argparse
import os
import sys

# Add project root for paper data imports
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend (safe for headless)
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogLocator, LogFormatter
    MPLOK = True
except ImportError:
    MPLOK = False
    print("matplotlib not available — install with: pip install matplotlib")

import numpy as np


OUTPUT_DIR = os.path.join(_HERE, "..", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

STYLE = {
    "figure.dpi":   120,
    "font.size":    10,
    "axes.grid":    True,
    "grid.alpha":   0.3,
    "axes.spines.top":   False,
    "axes.spines.right": False,
}

PLATFORM_COLORS = {
    "AGX":   "#e63946",  # Jetson AGX
    "NX":    "#f4a261",  # Jetson NX
    "TX2":   "#2a9d8f",  # Jetson TX2
    "Nano":  "#264653",  # Jetson Nano
    "VIM3":  "#457b9d",  # Khadas VIM3
    "M1":    "#1d3557",  # Odroid M1
    "H2":    "#a8dadc",  # Odroid H2
    "H2_NCS2": "#e9c46a",  # Odroid H2 + NCS2
}


# ════════════════════════════════════════════════════════════════════════════
# Figure 1 — Roofline Model
# ════════════════════════════════════════════════════════════════════════════

def fig1_roofline(show: bool = False, save: bool = True) -> None:
    """Reproduce Figure 1: three-panel roofline model."""
    from bang_for_the_buck.one_roofline_analysis.roofline_model import build_roofline
    from bang_for_the_buck.one_roofline_analysis.hardware_specs import ALL_PLATFORMS
    from bang_for_the_buck.one_roofline_analysis.operational_intensity import PAPER_OI_VALUES

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
        panels = [
            ("CPU_FP32",  "CPU FP32",  ["agx_cpu", "nx_cpu", "tx2_cpu", "nano_cpu", "vim3_big", "m1_cpu", "h2_cpu"]),
            ("GPU_FP16",  "GPU FP16",  ["agx_gpu", "nx_gpu", "tx2_gpu", "nano_gpu", "vim3_gpu", "m1_gpu"]),
            ("INT8",      "INT8",      ["agx_gpu_int8", "nx_gpu_int8", "vim3_npu", "m1_npu", "h2_ncs2"]),
        ]

        model_ois = {
            "MNV2": 1.33, "DN121": 1.80, "Xcep": 1.96,
            "RN101": 2.41, "YOLOv3": 2.42,
        }

        for ax, (mode, title, proc_keys) in zip(axes, panels):
            oi_range = np.logspace(-1, 2, 300)

            for pk in proc_keys:
                plat = ALL_PLATFORMS.get(pk)
                if plat is None:
                    continue
                label = f"{plat.name}"
                roof = build_roofline(plat, mode.split("_")[1].lower() if "_" in mode else "fp32")
                perf = [roof.attainable(oi) for oi in oi_range]
                ax.loglog(oi_range, perf, label=label, linewidth=1.5)

            # Mark model OI positions
            for mname, oi in model_ois.items():
                ax.axvline(x=oi, linestyle="--", color="grey", alpha=0.5, linewidth=0.8)
                ax.text(oi * 1.05, ax.get_ylim()[0] if ax.get_yscale() == "log" else 0.1,
                        mname, fontsize=7, rotation=90, color="grey")

            ax.set_title(title)
            ax.set_xlabel("Operational Intensity (FLOP/byte)")
            ax.set_ylabel("Attainable Performance (GFLOPS)")
            ax.legend(fontsize=7, loc="upper left")

        fig.suptitle("Figure 1: Roofline Models — Bang for the Buck (SEC '23)", y=1.02)
        plt.tight_layout()
        _save_or_show(fig, "fig1_roofline.pdf", show, save)


# ════════════════════════════════════════════════════════════════════════════
# Figure 2 — Inference Time Ranking
# ════════════════════════════════════════════════════════════════════════════

def fig2_inference_time(show: bool = False, save: bool = True) -> None:
    """Reproduce Figure 2: normalized inference time ranking (lower = faster)."""
    from bang_for_the_buck.three_metrics.inference_time import PAPER_INFERENCE_ORDERING

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        modes = [("CPU_FP32", "CPU FP32"), ("GPU_FP16", "GPU FP16"), ("INT8", "INT8")]

        for ax, (mode, title) in zip(axes, modes):
            data = PAPER_INFERENCE_ORDERING.get(mode, {})
            ordering = data.get("ordering", [])
            if not ordering:
                continue
            labels = [o[0] for o in ordering]
            values = [o[1] for o in ordering]
            colors = [PLATFORM_COLORS.get(l.split("_")[0], "#888") for l in labels]

            bars = ax.barh(labels, values, color=colors, edgecolor="white", linewidth=0.5)
            ax.set_xlabel("Normalized inference time (lower = faster)")
            ax.set_title(f"Fig 2: Inference Time\n{title}")
            ax.axvline(x=1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
            for bar, val in zip(bars, values):
                ax.text(val + 0.1, bar.get_y() + bar.get_height() / 2,
                        f"{val:.2f}×", va="center", fontsize=8)

        plt.tight_layout()
        _save_or_show(fig, "fig2_inference_time.pdf", show, save)


# ════════════════════════════════════════════════════════════════════════════
# Figure 3 — Total Time Ranking (inference vs end-to-end)
# ════════════════════════════════════════════════════════════════════════════

def fig3_total_time(show: bool = False, save: bool = True) -> None:
    """Reproduce Figure 3: inference vs total time speedup per platform."""
    from bang_for_the_buck.three_metrics.total_time import PAPER_SPEEDUPS

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 6))

        platforms = list(PAPER_SPEEDUPS.keys())
        infer_speedups = [PAPER_SPEEDUPS[p]["inference_speedup"] for p in platforms]
        total_speedups = [PAPER_SPEEDUPS[p]["total_speedup"] for p in platforms]
        pct_lost      = [PAPER_SPEEDUPS[p]["speedup_lost_pct"] for p in platforms]

        x = np.arange(len(platforms))
        w = 0.35
        b1 = ax.bar(x - w/2, infer_speedups, w, label="Inference-only speedup",
                    color="#e63946", alpha=0.85)
        b2 = ax.bar(x + w/2, total_speedups, w, label="Total end-to-end speedup",
                    color="#457b9d", alpha=0.85)

        for bar, pct in zip(b2, pct_lost):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f"−{pct:.0f}%", ha="center", va="bottom", fontsize=8, color="#e63946")

        ax.set_xticks(x)
        ax.set_xticklabels(platforms, rotation=20)
        ax.set_ylabel("Speedup vs CPU FP32 baseline")
        ax.set_title("Figure 3: Inference vs Total Time Speedup\n(CPU bottleneck reduces gains)")
        ax.legend()
        ax.axhline(y=1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.4)

        plt.tight_layout()
        _save_or_show(fig, "fig3_total_time.pdf", show, save)


# ════════════════════════════════════════════════════════════════════════════
# Figure 4 — Energy Per Inference
# ════════════════════════════════════════════════════════════════════════════

def fig4_energy(show: bool = False, save: bool = True) -> None:
    """Reproduce Figure 4: energy per inference (J) per platform per precision."""
    # Paper energy data (J per inference, MobileNetV2, normalised to M1_NPU=1.0)
    energy_data = {
        "M1_NPU":    {"INT8": 1.00, "FP32": 3.20},
        "VIM3_NPU":  {"INT8": 1.45, "FP32": 4.10},
        "H2_NCS2":   {"FP16": 1.62, "FP32": 6.30},
        "Nano_GPU":  {"FP16": 2.10, "FP32": 5.80},
        "NX_GPU":    {"FP16": 4.30, "FP32": 9.20},
        "TX2_GPU":   {"FP16": 5.90, "FP32": 12.4},
        "AGX_GPU":   {"FP16": 15.2, "FP32": 28.9},
        "AGX_DLA":   {"INT8": 3.80},
    }

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(11, 6))
        xticks = list(energy_data.keys())
        y_fp32 = [energy_data[p].get("FP32", 0) for p in xticks]
        y_quant = [energy_data[p].get("FP16", energy_data[p].get("INT8", 0)) for p in xticks]

        x = np.arange(len(xticks))
        w = 0.35
        ax.bar(x - w/2, y_fp32,  w, label="FP32", color="#e63946", alpha=0.8)
        ax.bar(x + w/2, y_quant, w, label="FP16/INT8", color="#2a9d8f", alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(xticks, rotation=20)
        ax.set_ylabel("Normalized energy per inference (M1 NPU = 1.0)")
        ax.set_title("Figure 4: Energy Per Inference\n(Lower is better; M1 NPU is most energy-efficient)")
        ax.legend()
        plt.tight_layout()
        _save_or_show(fig, "fig4_energy.pdf", show, save)


# ════════════════════════════════════════════════════════════════════════════
# Figure 5 — Temperature Delta
# ════════════════════════════════════════════════════════════════════════════

def fig5_temperature(show: bool = False, save: bool = True) -> None:
    """Reproduce Figure 5: temperature rise under inference load."""
    from bang_for_the_buck.three_metrics.temperature_monitor import PAPER_TEMPERATURE_DELTAS_C

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(12, 6))

        platforms = list(PAPER_TEMPERATURE_DELTAS_C.keys())
        procs = ["CPU_FP32", "GPU_FP16", "NPU_INT8"]
        labels_readable = {"CPU_FP32": "CPU FP32", "GPU_FP16": "GPU/DLA FP16", "NPU_INT8": "NPU INT8"}
        colors = {"CPU_FP32": "#e63946", "GPU_FP16": "#f4a261", "NPU_INT8": "#2a9d8f"}
        x = np.arange(len(platforms))
        w = 0.25

        for i, proc in enumerate(procs):
            vals = [PAPER_TEMPERATURE_DELTAS_C[p].get(proc, 0) for p in platforms]
            ax.bar(x + (i - 1) * w, vals, w, label=labels_readable[proc],
                   color=colors[proc], alpha=0.85)

        ax.axhline(y=40, color="red", linestyle="--", linewidth=1, alpha=0.7,
                   label="Thermal concern threshold (40°C)")
        ax.set_xticks(x)
        ax.set_xticklabels(platforms, rotation=20)
        ax.set_ylabel("Temperature rise under load (°C)")
        ax.set_title("Figure 5: Temperature Delta Under Inference Load\n"
                     "(H2+NCS2 reaches 85°C — thermal risk!)")
        ax.legend(fontsize=8)
        plt.tight_layout()
        _save_or_show(fig, "fig5_temperature.pdf", show, save)


# ════════════════════════════════════════════════════════════════════════════
# Figure 6 — SW Framework Comparison Odroid H2
# ════════════════════════════════════════════════════════════════════════════

def fig6_sw_odroid(show: bool = False, save: bool = True) -> None:
    """Figure 6: OpenVINO framework comparison on Odroid H2."""
    from bang_for_the_buck.four_sw_framework_comparison.runners.openvino_runner import PAPER_OPENVINO_RESULTS

    models = ["MobileNetV2", "ResNet101V2", "DenseNet121", "Xception"]
    cpu_vals  = [PAPER_OPENVINO_RESULTS[m]["H2_CPU_FP32_ms"] for m in models]
    gpu_vals  = [PAPER_OPENVINO_RESULTS[m]["H2_GPU_FP16_ms"] for m in models]
    ncs2_vals = [PAPER_OPENVINO_RESULTS[m]["NCS2_VPU_FP16_ms"] for m in models]

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(models))
        w = 0.25
        ax.bar(x - w, cpu_vals,  w, label="CPU FP32",    color="#e63946", alpha=0.85)
        ax.bar(x,     gpu_vals,  w, label="iGPU FP16",   color="#f4a261", alpha=0.85)
        ax.bar(x + w, ncs2_vals, w, label="NCS2 VPU FP16", color="#2a9d8f", alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(models, rotation=15)
        ax.set_ylabel("Inference time (ms)")
        ax.set_title("Figure 6: OpenVINO Framework Comparison — Odroid H2\n"
                     "(NCS2 is fastest, CPU is 2–3× slower than NCS2)")
        ax.legend()
        plt.tight_layout()
        _save_or_show(fig, "fig6_sw_odroid.pdf", show, save)


# ════════════════════════════════════════════════════════════════════════════
# Figure 7 — SW Framework Comparison VIM3
# ════════════════════════════════════════════════════════════════════════════

def fig7_sw_vim3(show: bool = False, save: bool = True) -> None:
    """Figure 7: ARM NN vs KSNN framework comparison on Khadas VIM3."""
    from bang_for_the_buck.four_sw_framework_comparison.runners.ksnn_runner import PAPER_KSNN_RESULTS
    from bang_for_the_buck.four_sw_framework_comparison.runners.armnn_runner import PAPER_ARMNN_RESULTS

    models = ["MobileNetV2", "ResNet101V2", "DenseNet121", "Xception"]
    vim3_data = PAPER_ARMNN_RESULTS.get("VIM3", {})

    cpu_vals  = [vim3_data.get(m, {}).get("CpuAcc_FP32_ms", 0) for m in models]
    gpu_vals  = [vim3_data.get(m, {}).get("GpuAcc_FP16_ms", 0) for m in models]
    npu_vals  = [PAPER_KSNN_RESULTS.get(m, {}).get("KSNN_INT8_ms", 0) for m in models]

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(models))
        w = 0.25
        ax.bar(x - w, cpu_vals, w, label="ARM NN CPU FP32",  color="#e63946", alpha=0.85)
        ax.bar(x,     gpu_vals, w, label="ARM NN GPU FP16",  color="#f4a261", alpha=0.85)
        ax.bar(x + w, npu_vals, w, label="KSNN NPU INT8",    color="#2a9d8f", alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(models, rotation=15)
        ax.set_ylabel("Inference time (ms)")
        ax.set_title("Figure 7: ARM NN vs KSNN — Khadas VIM3\n"
                     "(NPU ~5× faster than Mali GPU — best on VIM3)")
        ax.legend()
        plt.tight_layout()
        _save_or_show(fig, "fig7_sw_vim3.pdf", show, save)


# ════════════════════════════════════════════════════════════════════════════
# Figure 8 — AutoScheduler vs TensorRT
# ════════════════════════════════════════════════════════════════════════════

def fig8_autosched(show: bool = False, save: bool = True) -> None:
    """Figure 8: AutoScheduler vs TensorRT on Jetson TX2."""
    from bang_for_the_buck.four_sw_framework_comparison.runners.autoschedule_runner import (
        PAPER_AUTOSCHED_VS_TENSORRT_TX2,
    )

    models = ["MobileNetV2", "DenseNet121", "Xception", "ResNet101V2"]
    trt_vals  = [PAPER_AUTOSCHED_VS_TENSORRT_TX2[m]["TensorRT_FP16_ms"] for m in models]
    auto_vals = [PAPER_AUTOSCHED_VS_TENSORRT_TX2[m]["AutoScheduler_ms"] for m in models]

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(models))
        w = 0.35
        ax.bar(x - w/2, trt_vals,  w, label="TensorRT FP16",    color="#e63946", alpha=0.85)
        ax.bar(x + w/2, auto_vals, w, label="AutoScheduler FP16", color="#457b9d", alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(models, rotation=15)
        ax.set_ylabel("Inference time (ms)")
        ax.set_title("Figure 8: AutoScheduler vs TensorRT — Jetson TX2\n"
                     "(AutoScheduler matches TensorRT within ~7% — no proprietary tools needed)")
        ax.legend()
        plt.tight_layout()
        _save_or_show(fig, "fig8_autosched_vs_trt.pdf", show, save)


# ════════════════════════════════════════════════════════════════════════════
# Figure 9 — DWA Odroid H2
# ════════════════════════════════════════════════════════════════════════════

def fig9_dwa_h2(show: bool = False, save: bool = True) -> None:
    """Figure 9: DWA results on Odroid H2 (CPU + NCS2 ~43% improvement)."""
    from bang_for_the_buck.five_concurrent_execution.dwa.data_wise_allocation import (
        PAPER_DWA_RESULTS,
    )

    platform_data = PAPER_DWA_RESULTS.get("Odroid_H2_CPU_NCS2", {})
    models = [k for k, v in platform_data.items() if isinstance(v, dict)]

    solo_cpu  = [platform_data[m]["solo_CPU_ms"] for m in models]
    solo_ncs2 = [float(platform_data[m].get("solo_NCS2_ms", 0)) for m in models]
    dwa_vals  = [float(platform_data[m].get("DWA_ms", 0)) for m in models]

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(models))
        w = 0.25
        ax.bar(x - w, solo_cpu,  w, label="Solo CPU FP32",    color="#e63946", alpha=0.85)
        ax.bar(x,     solo_ncs2, w, label="Solo NCS2 VPU FP16", color="#2a9d8f", alpha=0.85)
        ax.bar(x + w, dwa_vals,  w, label="DWA CPU+NCS2",     color="#264653", alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(models, rotation=15)
        ax.set_ylabel("End-to-end latency (ms)")
        ax.set_title("Figure 9: DWA — Odroid H2 CPU + NCS2\n"
                     "(43% latency reduction vs solo NCS2)")
        ax.legend()
        plt.tight_layout()
        _save_or_show(fig, "fig9_dwa_h2.pdf", show, save)


# ════════════════════════════════════════════════════════════════════════════
# Figure 10 — DWA VIM3
# ════════════════════════════════════════════════════════════════════════════

def fig10_dwa_vim3(show: bool = False, save: bool = True) -> None:
    """Figure 10: DWA on VIM3 — modest improvement (speed asymmetry too high)."""
    from bang_for_the_buck.four_sw_framework_comparison.runners.ksnn_runner import PAPER_KSNN_RESULTS
    from bang_for_the_buck.four_sw_framework_comparison.runners.armnn_runner import PAPER_ARMNN_RESULTS

    models  = ["MobileNetV2", "ResNet101V2", "DenseNet121", "Xception"]
    vim3    = PAPER_ARMNN_RESULTS.get("VIM3", {})
    dwa_est = {
        "MobileNetV2": 34.8, "ResNet101V2": 252.0, "DenseNet121": 234.0, "Xception": 390.0
    }

    cpu   = [vim3.get(m, {}).get("CpuAcc_FP32_ms", 0) for m in models]
    npu   = [PAPER_KSNN_RESULTS.get(m, {}).get("KSNN_INT8_ms", 0) for m in models]
    dwa   = [dwa_est.get(m, 0) for m in models]

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(models))
        w = 0.25
        ax.bar(x - w, cpu, w, label="Solo CPU (ARM NN FP32)",  color="#e63946", alpha=0.85)
        ax.bar(x,     npu, w, label="Solo NPU (KSNN INT8)",    color="#2a9d8f", alpha=0.85)
        ax.bar(x + w, dwa, w, label="DWA CPU+NPU (manual)",   color="#264653", alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(models, rotation=15)
        ax.set_ylabel("End-to-end latency (ms)")
        ax.set_title("Figure 10: DWA — Khadas VIM3 CPU + KSNN NPU\n"
                     "(Only ~11% improvement — high speed asymmetry limits DWA benefit)")
        ax.legend()
        plt.tight_layout()
        _save_or_show(fig, "fig10_dwa_vim3.pdf", show, save)


# ════════════════════════════════════════════════════════════════════════════
# Figure 11 — EDCP Ranking
# ════════════════════════════════════════════════════════════════════════════

def fig11_edcp(show: bool = False, save: bool = True) -> None:
    """Figure 11: EDCP ranking (novel metric from paper)."""
    from bang_for_the_buck.three_metrics.edcp import PAPER_EDCP_ORDERING

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        platforms = [r["platform"] for r in PAPER_EDCP_ORDERING]
        edcp_vals = [r["normalized_edcp"] for r in PAPER_EDCP_ORDERING]
        colors    = [PLATFORM_COLORS.get(p.split("_")[0], "#888") for p in platforms]

        bars = ax.bar(platforms, edcp_vals, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_ylabel("Normalized EDCP (M1 = 1.0, lower is better)")
        ax.set_title("Figure 11: EDCP – Energy × Delay × Cost Product\n"
                     "(Novel metric — M1 is most cost-efficient; AGX is worst despite high throughput)")

        for bar, val in zip(bars, edcp_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                    f"{val:.1f}×", ha="center", va="bottom", fontsize=9)

        ax.axhline(y=1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.4)
        plt.tight_layout()
        _save_or_show(fig, "fig11_edcp.pdf", show, save)


# ════════════════════════════════════════════════════════════════════════════
# Figure 12 — Adaptive Scheduler Workflow
# ════════════════════════════════════════════════════════════════════════════

def fig12_adaptive_scheduler(show: bool = False, save: bool = True) -> None:
    """Figure 12: Adaptive scheduler results and workflow diagram."""
    from bang_for_the_buck.six_application_case_study.vehicle_detector import PAPER_YOLOV3_ACCURACY
    from bang_for_the_buck.six_application_case_study.heterogeneous_scheduler import (
        PAPER_SCHEDULING_POLICY,
    )

    with plt.rc_context(STYLE):
        fig, (ax_bar, ax_text) = plt.subplots(1, 2, figsize=(14, 6))

        # Left: bar chart comparing GPU-only vs NPU-only vs Adaptive
        modes = ["GPU_FP16", "NPU_INT8", "Adaptive"]
        fps   = [PAPER_YOLOV3_ACCURACY[m]["FPS"] for m in modes]
        ap    = [PAPER_YOLOV3_ACCURACY[m]["AP_pct"] for m in modes]
        colors_bar = ["#e63946", "#2a9d8f", "#457b9d"]

        x = np.arange(len(modes))
        w = 0.35
        ax2 = ax_bar.twinx()
        ax_bar.bar(x - w/2, fps, w, color=colors_bar, alpha=0.7, label="FPS")
        ax2.bar(   x + w/2, ap,  w, color=colors_bar, alpha=0.4, hatch="//", label="AP %")

        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(["GPU FP16\n(accurate)", "NPU INT8\n(fast)", "Adaptive\n(best tradeoff)"])
        ax_bar.set_ylabel("FPS", color="#e63946")
        ax2.set_ylabel("AP %", color="#2a9d8f")
        ax_bar.set_title("Fig 12a: Results — YOLOv3 Detection")

        lines_fps  = mpatches.Patch(color="#e63946", alpha=0.7, label="FPS")
        lines_ap   = mpatches.Patch(color="#2a9d8f", alpha=0.4, hatch="//", label="AP %")
        ax_bar.legend(handles=[lines_fps, lines_ap], loc="upper left")

        # Right: workflow text diagram
        ax_text.axis("off")
        workflow = (
            "Adaptive Scheduler Workflow (§7)\n\n"
            "Video Frame\n"
            "     │\n"
            "     ▼\n"
            "Traffic Density Estimator (CPU)\n"
            "  Background subtraction\n"
            "  Vehicle blob counting\n"
            "     │\n"
            "  ┌──┴──────────────┐\n"
            "  │ HEAVY (≥9 vehicles) │  LIGHT/MODERATE\n"
            "  ▼                   ▼\n"
            "GPU FP16           NPU INT8\n"
            "TensorRT            RKNN\n"
            "AP=96.3%           AP=90.1%\n"
            "14 FPS             38 FPS\n"
            "     └──────────────┘\n"
            "              │\n"
            "     Adaptive: AP=93.9%  FPS=28.4\n"
            f"\nGPU activated: {PAPER_SCHEDULING_POLICY['gpu_activation_pct']}% of frames"
        )
        ax_text.text(0.05, 0.95, workflow, transform=ax_text.transAxes,
                     verticalalignment="top", fontsize=9, family="monospace",
                     bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8))
        ax_text.set_title("Fig 12b: Scheduling Workflow")

        plt.tight_layout()
        _save_or_show(fig, "fig12_adaptive_scheduler.pdf", show, save)


# ════════════════════════════════════════════════════════════════════════════
# Utilities
# ════════════════════════════════════════════════════════════════════════════

def _save_or_show(fig, filename: str, show: bool, save: bool) -> None:
    """Save figure to disk and/or display."""
    if save:
        path = os.path.join(OUTPUT_DIR, filename)
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"  Saved: {path}")
    if show:
        plt.show()
    plt.close(fig)


ALL_FIGURES = {
    1:  fig1_roofline,
    2:  fig2_inference_time,
    3:  fig3_total_time,
    4:  fig4_energy,
    5:  fig5_temperature,
    6:  fig6_sw_odroid,
    7:  fig7_sw_vim3,
    8:  fig8_autosched,
    9:  fig9_dwa_h2,
    10: fig10_dwa_vim3,
    11: fig11_edcp,
    12: fig12_adaptive_scheduler,
}


def plot_all(show: bool = False) -> None:
    """Generate all figures."""
    if not MPLOK:
        print("matplotlib not available")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Generating {len(ALL_FIGURES)} figures → {OUTPUT_DIR}/")
    for fig_num, fn in ALL_FIGURES.items():
        print(f"  Figure {fig_num:>2}: {fn.__name__}...")
        try:
            fn(show=show, save=True)
        except Exception as exc:
            print(f"    Warning: could not generate figure {fig_num}: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bang for the Buck — paper figure reproduction")
    parser.add_argument("--show",    action="store_true", help="Show figures interactively")
    parser.add_argument("--fig",     type=int, default=0, help="Generate specific figure (1-12)")
    parser.add_argument("--no-save", action="store_true", help="Don't save to disk")
    args = parser.parse_args()

    save = not args.no_save

    if args.fig > 0:
        fn = ALL_FIGURES.get(args.fig)
        if fn is None:
            print(f"Figure {args.fig} not found. Available: {sorted(ALL_FIGURES.keys())}")
            sys.exit(1)
        print(f"Generating Figure {args.fig}...")
        fn(show=args.show, save=save)
    else:
        plot_all(show=args.show)
