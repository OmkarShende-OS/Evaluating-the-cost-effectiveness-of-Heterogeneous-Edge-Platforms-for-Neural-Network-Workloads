"""
platform_configs.py — Runtime platform detection and configuration for the
benchmarks in "Bang for the Buck" (SEC '23).

Provides:
  • Auto-detection of the current hardware platform
  • Mapping from platform → available software frameworks
  • Default benchmark parameters per platform
  • Table 2 platform specifications (cost, memory, processors)
"""

import os
import sys
import platform
import subprocess
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class PlatformBenchConfig:
    """Runtime benchmark configuration for a specific platform."""
    name: str
    short_name: str
    cost_usd: float
    warmup_iterations: int = 20
    measure_iterations: int = 100
    batch_size: int = 1
    # Available framework-processor combinations on this platform
    # format: [(framework_key, processor_name, precision), ...]
    supported_combos: List[tuple] = field(default_factory=list)
    power_monitor: str = "none"   # "tegrastats", "ina219", "nvml", "none"
    notes: str = ""


# ---------------------------------------------------------------------------
# Platform configurations used in the paper
# ---------------------------------------------------------------------------
PLATFORM_CONFIGS: Dict[str, PlatformBenchConfig] = {
    "jetson_agx": PlatformBenchConfig(
        name="Jetson Xavier AGX",
        short_name="AGX",
        cost_usd=999.0,
        warmup_iterations=30,
        measure_iterations=100,
        supported_combos=[
            ("tensorrt",      "AGX_GPU",  "FP16"),
            ("tensorrt",      "AGX_GPU",  "INT8"),
            ("autoschedule",  "AGX_GPU",  "FP16"),
            ("autoschedule",  "AGX_GPU",  "INT8"),
            ("tflite",        "AGX_CPU",  "FP32"),
            ("tensorrt",      "AGX_DLA",  "INT8"),
        ],
        power_monitor="tegrastats",
        notes="Primary high-performance platform; 2× NVDLA v1"
    ),
    "jetson_nx": PlatformBenchConfig(
        name="Jetson Xavier NX",
        short_name="NX",
        cost_usd=479.0,
        warmup_iterations=30,
        measure_iterations=100,
        supported_combos=[
            ("tensorrt",      "NX_GPU",   "FP16"),
            ("tensorrt",      "NX_GPU",   "INT8"),
            ("tflite",        "NX_CPU",   "FP32"),
            ("tensorrt",      "NX_DLA",   "INT8"),
        ],
        power_monitor="tegrastats",
    ),
    "jetson_tx2": PlatformBenchConfig(
        name="Jetson TX2",
        short_name="TX2",
        cost_usd=479.0,
        supported_combos=[
            ("tensorrt",  "TX2_GPU",  "FP16"),
            ("tensorrt",  "TX2_GPU",  "INT8"),
            ("tflite",    "TX2_CPU",  "FP32"),
        ],
        power_monitor="tegrastats",
    ),
    "jetson_nano": PlatformBenchConfig(
        name="Jetson Nano",
        short_name="Nano",
        cost_usd=119.0,
        supported_combos=[
            ("tensorrt",  "Nano_GPU",  "FP16"),
            ("tflite",    "Nano_CPU",  "FP32"),
        ],
        power_monitor="tegrastats",
    ),
    "vim3": PlatformBenchConfig(
        name="Khadas VIM3",
        short_name="VIM3",
        cost_usd=159.90,
        supported_combos=[
            ("armnn",     "VIM3_CPU_Big",    "FP32"),
            ("armnn",     "VIM3_CPU_Big",    "INT8"),
            ("pyarmnn",   "VIM3_CPU_Big",    "FP32"),
            ("armnn",     "VIM3_GPU",        "FP16"),
            ("tflite",    "VIM3_CPU_Big",    "FP32"),
            ("tflite",    "VIM3_CPU_Big",    "INT8"),
            ("ksnn",      "VIM3_NPU",        "INT8"),
        ],
        power_monitor="ina219",
        notes="ARM big.LITTLE + Mali GPU + VeriSilicon NPU; needs ARMNN + KSNN"
    ),
    "odroid_h2": PlatformBenchConfig(
        name="Odroid H2",
        short_name="H2",
        cost_usd=110.0,
        supported_combos=[
            ("openvino",  "H2_CPU",    "FP32"),
            ("openvino",  "H2_CPU",    "FP16"),
            ("openvino",  "H2_CPU",    "INT8"),
            ("openvino",  "H2_GPU",    "FP16"),
            ("openvino",  "NCS2_VPU",  "FP16"),  # NCS2 plugged via USB3
            ("tflite",    "H2_CPU",    "FP32"),
        ],
        power_monitor="ina219",
        notes="Intel platform — OpenVINO MULTI plugin for DWA concurrency"
    ),
    "odroid_m1": PlatformBenchConfig(
        name="Odroid M1",
        short_name="M1",
        cost_usd=95.50,
        supported_combos=[
            ("armnn",   "M1_CPU",   "FP32"),
            ("armnn",   "M1_GPU",   "FP16"),
            ("armnn",   "M1_GPU",   "INT8"),
            ("tflite",  "M1_CPU",   "FP32"),
            ("rknn",    "M1_NPU",   "INT8"),
        ],
        power_monitor="ina219",
        notes="Baseline platform ($95.50). RKNN for NPU inference."
    ),
}

# The paper's baseline platform for normalisation
BASELINE_PLATFORM = "odroid_m1"
BASELINE_PROCESSOR = {
    "FP32": "M1_CPU",
    "FP16": "M1_GPU",
    "INT8": "M1_NPU",
}


# ---------------------------------------------------------------------------
# Auto-detection helpers
# ---------------------------------------------------------------------------

def _run(cmd: str) -> str:
    """Run a shell command, return stdout, empty string on error."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True,
                                text=True, timeout=5)
        return result.stdout.strip()
    except Exception:
        return ""


def detect_platform() -> Optional[str]:
    """
    Auto-detect the current hardware platform.
    Returns a key from PLATFORM_CONFIGS or None if unrecognised.
    """
    # Check Jetson platforms via /proc/device-tree/model
    model_file = "/proc/device-tree/model"
    if os.path.exists(model_file):
        with open(model_file, "rb") as f:
            model_str = f.read().decode("utf-8", errors="ignore").lower()
        if "agx xavier" in model_str:
            return "jetson_agx"
        if "xavier nx" in model_str:
            return "jetson_nx"
        if "tx2" in model_str:
            return "jetson_tx2"
        if "nano" in model_str:
            return "jetson_nano"

    # Check via /etc/nv_tegra_release
    if os.path.exists("/etc/nv_tegra_release"):
        return _detect_jetson_from_nvinfo()

    # Check Khadas VIM3 via board info
    board = _run("cat /proc/device-tree/amlogic-dt-id 2>/dev/null")
    if "kvim3" in board.lower():
        return "vim3"

    # Check Odroid M1 via cpuinfo
    cpuinfo = _run("cat /proc/cpuinfo")
    if "rk3568" in cpuinfo.lower():
        return "odroid_m1"

    # Check Odroid H2 via DMI
    dmi = _run("cat /sys/class/dmi/id/board_name 2>/dev/null")
    if "odroid-h2" in dmi.lower():
        return "odroid_h2"

    return None


def _detect_jetson_from_nvinfo() -> Optional[str]:
    """Parse /etc/nv_tegra_release to disambiguate Jetson variants."""
    try:
        content = open("/etc/nv_tegra_release").read().lower()
        if "xavier" in content:
            # Distinguish AGX vs NX by memory
            mem = _run("cat /proc/meminfo | grep MemTotal")
            if mem:
                kb = int(mem.split()[1])
                gb = kb / (1024 * 1024)
                return "jetson_agx" if gb > 20 else "jetson_nx"
        if "tx2" in content:
            return "jetson_tx2"
    except Exception:
        pass
    return None


def get_platform_config(platform_key: Optional[str] = None) -> PlatformBenchConfig:
    """
    Get benchmark configuration for a specific platform.
    If platform_key is None, auto-detect the current platform.
    Falls back to odroid_m1 default if detection fails.
    """
    if platform_key is None:
        platform_key = detect_platform()
    if platform_key is None or platform_key not in PLATFORM_CONFIGS:
        print(f"[WARNING] Platform unknown or not in config. "
              f"Detected: {platform_key!r}. Using odroid_m1 defaults.")
        platform_key = "odroid_m1"
    return PLATFORM_CONFIGS[platform_key]


def print_platform_table():
    """Print Table 2 from the paper."""
    print(f"{'Platform':<25} {'Cost':>6} {'Combos':>7}")
    print("-" * 44)
    for key, cfg in PLATFORM_CONFIGS.items():
        print(f"{cfg.name:<25} ${cfg.cost_usd:>6.2f} {len(cfg.supported_combos):>7}")


if __name__ == "__main__":
    print("=== Platform Configurations (Table 2, Bang for the Buck) ===\n")
    print_platform_table()
    print()
    detected = detect_platform()
    print(f"Auto-detected platform: {detected!r}")
    cfg = get_platform_config(detected)
    print(f"Using config: {cfg.name}")
    print(f"  Supported combos: {cfg.supported_combos}")
    print(f"  Power monitor:    {cfg.power_monitor}")
