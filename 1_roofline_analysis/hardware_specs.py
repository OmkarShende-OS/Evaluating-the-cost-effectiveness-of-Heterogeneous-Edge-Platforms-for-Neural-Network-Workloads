"""
hardware_specs.py — Peak compute and memory bandwidth specifications for all
8 edge platforms evaluated in "Bang for the Buck" (SEC '23).

Data sourced from vendor datasheets (Table 2 of the paper).
Used by roofline_model.py to construct theoretical performance bounds.

Platforms covered:
  - Jetson Xavier AGX, NX, TX2, Nano  (NVIDIA)
  - Khadas VIM3                         (ARM CPU/GPU + VeriSilicon NPU)
  - Odroid H2 + Intel NCS2             (Intel CPU/GPU + Myriad X VPU)
  - Odroid M1                           (ARM CPU/GPU + Rockchip NPU)
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ProcessorSpec:
    """Specifications for a single processing element."""
    name: str
    proc_type: str          # "CPU", "GPU", "NPU", "VPU"
    precision: str          # "FP32", "FP16", "INT8"
    peak_compute_gflops: float   # GFLOPS (or GOPS for INT8)
    peak_bandwidth_gbs: float    # GB/s memory bandwidth
    num_cores: int
    frequency_ghz: float
    notes: str = ""


@dataclass
class PlatformSpec:
    """Complete specification of an edge platform."""
    name: str
    short_name: str
    cost_usd: float
    memory_gb: float
    memory_bandwidth_gbs: float  # Shared DRAM bandwidth
    processors: List[ProcessorSpec] = field(default_factory=list)
    notes: str = ""


# ---------------------------------------------------------------------------
# Per-platform specifications from vendor datasheets
# Peak compute = cores × ops_per_cycle × SIMD_width × frequency
# ---------------------------------------------------------------------------

JETSON_AGX = PlatformSpec(
    name="Jetson Xavier AGX",
    short_name="AGX",
    cost_usd=999.0,
    memory_gb=32.0,
    memory_bandwidth_gbs=137.0,  # 256-bit LPDDR4x @ 2133 MHz
    processors=[
        ProcessorSpec(
            name="AGX_CPU",
            proc_type="CPU",
            precision="FP32",
            # 8-core Carmel ARM v8.2 @ 2.265 GHz, 128-bit NEON SIMD
            peak_compute_gflops=8 * 2 * 4 * 2.265,   # cores×FMA×NEON_width×freq
            peak_bandwidth_gbs=137.0,
            num_cores=8,
            frequency_ghz=2.265,
            notes="ARM Carmel v8.2 with FP64/FP32/FP16/INT8 NEON"
        ),
        ProcessorSpec(
            name="AGX_GPU",
            proc_type="GPU",
            precision="FP16",
            # 512-core Volta @ 1.377 GHz, 2 FMA per clock per core
            peak_compute_gflops=512 * 2 * 2 * 1.377,
            peak_bandwidth_gbs=137.0,
            num_cores=512,
            frequency_ghz=1.377,
            notes="Volta GPU — 8 Tensor Cores per SM, FP16x2 throughput"
        ),
        ProcessorSpec(
            name="AGX_DLA",
            proc_type="NPU",
            precision="INT8",
            peak_compute_gflops=11.0,   # 11 TOPS per NVDLA v1 (2 present)
            peak_bandwidth_gbs=137.0,
            num_cores=1,
            frequency_ghz=1.0,
            notes="NVDLA v1 — does not support all NN layer types"
        ),
    ]
)

JETSON_NX = PlatformSpec(
    name="Jetson Xavier NX",
    short_name="NX",
    cost_usd=479.0,
    memory_gb=16.0,
    memory_bandwidth_gbs=51.2,   # 128-bit LPDDR4x @ 1600 MHz
    processors=[
        ProcessorSpec(
            name="NX_CPU",
            proc_type="CPU",
            precision="FP32",
            peak_compute_gflops=6 * 2 * 4 * 1.4,
            peak_bandwidth_gbs=51.2,
            num_cores=6,
            frequency_ghz=1.4,
            notes="ARM Carmel v8.2 6-core"
        ),
        ProcessorSpec(
            name="NX_GPU",
            proc_type="GPU",
            precision="FP16",
            peak_compute_gflops=384 * 2 * 2 * 1.1,
            peak_bandwidth_gbs=51.2,
            num_cores=384,
            frequency_ghz=1.1,
            notes="Volta 384-core"
        ),
        ProcessorSpec(
            name="NX_DLA",
            proc_type="NPU",
            precision="INT8",
            peak_compute_gflops=5.5,    # 5.5 TOPS per NVDLA
            peak_bandwidth_gbs=51.2,
            num_cores=1,
            frequency_ghz=1.0,
            notes="NVDLA v1 — same limitations as AGX DLA"
        ),
    ]
)

JETSON_TX2 = PlatformSpec(
    name="Jetson TX2",
    short_name="TX2",
    cost_usd=479.0,
    memory_gb=8.0,
    memory_bandwidth_gbs=59.7,   # 128-bit LPDDR4 @ 1866 MHz
    processors=[
        ProcessorSpec(
            name="TX2_CPU",
            proc_type="CPU",
            precision="FP32",
            # 2× Denver2 + 4× Cortex-A57 @ 2 GHz
            # Denver2 is fused 128-bit NEON; A57 has 64-bit NEON
            peak_compute_gflops=(2 * 2 * 8 + 4 * 2 * 4) * 2.0,
            peak_bandwidth_gbs=59.7,
            num_cores=6,
            frequency_ghz=2.0,
            notes="Heterogeneous big.LITTLE-style: 2× Denver2 + 4× Cortex-A57"
        ),
        ProcessorSpec(
            name="TX2_GPU",
            proc_type="GPU",
            precision="FP16",
            peak_compute_gflops=256 * 2 * 2 * 1.3,
            peak_bandwidth_gbs=59.7,
            num_cores=256,
            frequency_ghz=1.3,
            notes="Pascal 256-core"
        ),
    ]
)

JETSON_NANO = PlatformSpec(
    name="Jetson Nano",
    short_name="Nano",
    cost_usd=119.0,
    memory_gb=4.0,
    memory_bandwidth_gbs=25.6,   # 64-bit LPDDR4 @ 1600 MHz
    processors=[
        ProcessorSpec(
            name="Nano_CPU",
            proc_type="CPU",
            precision="FP32",
            peak_compute_gflops=4 * 2 * 4 * 1.43,
            peak_bandwidth_gbs=25.6,
            num_cores=4,
            frequency_ghz=1.43,
            notes="ARM Cortex-A57 4-core"
        ),
        ProcessorSpec(
            name="Nano_GPU",
            proc_type="GPU",
            precision="FP16",
            peak_compute_gflops=128 * 2 * 2 * 0.921,
            peak_bandwidth_gbs=25.6,
            num_cores=128,
            frequency_ghz=0.921,
            notes="Maxwell 128-core"
        ),
    ]
)

KHADAS_VIM3 = PlatformSpec(
    name="Khadas VIM3",
    short_name="VIM3",
    cost_usd=159.90,
    memory_gb=4.0,
    memory_bandwidth_gbs=12.8,   # 64-bit LPDDR4 @ 800 MHz
    processors=[
        ProcessorSpec(
            name="VIM3_CPU_Big",
            proc_type="CPU",
            precision="FP32",
            # 4× Cortex-A73 @ 2.2 GHz, 128-bit NEON
            peak_compute_gflops=4 * 2 * 4 * 2.2,
            peak_bandwidth_gbs=12.8,
            num_cores=4,
            frequency_ghz=2.2,
            notes="big cluster: ARM Cortex-A73"
        ),
        ProcessorSpec(
            name="VIM3_CPU_Little",
            proc_type="CPU",
            precision="FP32",
            # 2× Cortex-A53 @ 1.8 GHz, 64-bit NEON
            peak_compute_gflops=2 * 2 * 2 * 1.8,
            peak_bandwidth_gbs=12.8,
            num_cores=2,
            frequency_ghz=1.8,
            notes="little cluster: ARM Cortex-A53"
        ),
        ProcessorSpec(
            name="VIM3_GPU",
            proc_type="GPU",
            precision="FP16",
            # Mali-G52 MP4 @ 800 MHz
            peak_compute_gflops=4 * 2 * 0.8,   # 4 cores × 2 FP16 FMA × freq
            peak_bandwidth_gbs=12.8,
            num_cores=4,
            frequency_ghz=0.8,
            notes="Mali-G52 MP4 — Bifrost architecture"
        ),
        ProcessorSpec(
            name="VIM3_NPU",
            proc_type="NPU",
            precision="INT8",
            peak_compute_gflops=5.0,   # 5 TOPS Amlogic A311D NPU
            peak_bandwidth_gbs=12.8,
            num_cores=1,
            frequency_ghz=0.8,
            notes="Amlogic / VeriSilicon 5 TOPS NPU — INT8 only"
        ),
    ]
)

ODROID_H2 = PlatformSpec(
    name="Odroid H2",
    short_name="H2",
    cost_usd=110.0,
    memory_gb=16.0,
    memory_bandwidth_gbs=34.1,   # DDR4-2133 dual-channel
    processors=[
        ProcessorSpec(
            name="H2_CPU",
            proc_type="CPU",
            precision="FP32",
            # 4× Intel Celeron J4105 @ 2.3 GHz, SSE4.2 (no AVX)
            peak_compute_gflops=4 * 2 * 4 * 2.3,   # 4 cores × FMA × 128-bit SSE
            peak_bandwidth_gbs=34.1,
            num_cores=4,
            frequency_ghz=2.3,
            notes="Intel Gemini Lake — no AVX, SSE4.2 only"
        ),
        ProcessorSpec(
            name="H2_GPU",
            proc_type="GPU",
            precision="FP16",
            peak_compute_gflops=0.192,  # Intel UHD 600 — 12 EU @ 750 MHz
            peak_bandwidth_gbs=34.1,
            num_cores=12,
            frequency_ghz=0.75,
            notes="Intel UHD 600 (Gemini Lake integrated)"
        ),
    ]
)

INTEL_NCS2 = PlatformSpec(
    name="Intel NCS2 (Myriad X VPU)",
    short_name="NCS2",
    cost_usd=95.50,     # standalone stick price
    memory_gb=0.5,      # 512 MB LPDDR4 on-stick
    memory_bandwidth_gbs=4.0,
    processors=[
        ProcessorSpec(
            name="NCS2_VPU",
            proc_type="VPU",
            precision="FP16",
            peak_compute_gflops=4.0,    # Myriad X: 4 TOPS FP16
            peak_bandwidth_gbs=4.0,     # on-chip — very low external BW
            num_cores=16,               # 16 SHAVE processors
            frequency_ghz=0.7,
            notes="Myriad X VPU — 16 SHAVE cores, plugged via USB 3.0 to H2"
        ),
    ]
)

ODROID_M1 = PlatformSpec(
    name="Odroid M1",
    short_name="M1",
    cost_usd=95.50,
    memory_gb=8.0,
    memory_bandwidth_gbs=13.3,   # 64-bit LPDDR4 @ 1600 MHz
    processors=[
        ProcessorSpec(
            name="M1_CPU",
            proc_type="CPU",
            precision="FP32",
            # 4× Cortex-A55 @ 2.0 GHz, 128-bit NEON
            peak_compute_gflops=4 * 2 * 4 * 2.0,
            peak_bandwidth_gbs=13.3,
            num_cores=4,
            frequency_ghz=2.0,
            notes="Rockchip RK3568 — ARM Cortex-A55"
        ),
        ProcessorSpec(
            name="M1_GPU",
            proc_type="GPU",
            precision="FP16",
            # Mali-G52 MP2 @ 650 MHz
            peak_compute_gflops=2 * 2 * 0.65,
            peak_bandwidth_gbs=13.3,
            num_cores=2,
            frequency_ghz=0.65,
            notes="Mali-G52 MP2 — Bifrost architecture"
        ),
        ProcessorSpec(
            name="M1_NPU",
            proc_type="NPU",
            precision="INT8",
            peak_compute_gflops=0.8,    # 0.8 TOPS Rockchip NPU
            peak_bandwidth_gbs=13.3,
            num_cores=1,
            frequency_ghz=0.5,
            notes="Rockchip RK3568 NPU — 0.8 TOPS INT8"
        ),
    ]
)

# ---------------------------------------------------------------------------
# Registry: all platforms indexed by short name for easy lookup
# ---------------------------------------------------------------------------

ALL_PLATFORMS: Dict[str, PlatformSpec] = {
    "AGX":  JETSON_AGX,
    "NX":   JETSON_NX,
    "TX2":  JETSON_TX2,
    "Nano": JETSON_NANO,
    "VIM3": KHADAS_VIM3,
    "H2":   ODROID_H2,
    "NCS2": INTEL_NCS2,
    "M1":   ODROID_M1,
}

# Platform cost ordering (from cheapest to most expensive)
COST_ORDERING = sorted(ALL_PLATFORMS.keys(), key=lambda k: ALL_PLATFORMS[k].cost_usd)


def get_processor(platform_name: str, proc_name: str) -> Optional[ProcessorSpec]:
    """Look up a specific processor by platform and processor name."""
    platform = ALL_PLATFORMS.get(platform_name)
    if platform is None:
        return None
    for proc in platform.processors:
        if proc.name == proc_name:
            return proc
    return None


def list_processors_by_type(proc_type: str) -> List[ProcessorSpec]:
    """Return all processors of a given type (CPU/GPU/NPU/VPU) across all platforms."""
    result = []
    for platform in ALL_PLATFORMS.values():
        for proc in platform.processors:
            if proc.proc_type == proc_type:
                result.append(proc)
    return result


if __name__ == "__main__":
    print("=== Edge Platform Specifications (Bang for the Buck, SEC '23) ===\n")
    for name, plat in ALL_PLATFORMS.items():
        print(f"{plat.name} (${plat.cost_usd:.2f})")
        for proc in plat.processors:
            print(f"  [{proc.proc_type}] {proc.name}: "
                  f"{proc.peak_compute_gflops:.1f} GFLOPS @ {proc.precision}, "
                  f"BW={proc.peak_bandwidth_gbs:.1f} GB/s")
        print()
