"""
temperature_monitor.py — Thermal monitoring for edge platforms.

"Bang for the Buck" (SEC '23) §5.2 measures average temperature during
NN inference to identify thermal throttling risk.

Key paper finding:
  - Odroid H2 + NCS2 reaches up to 85°C (thermal risk!)
  - AGX, NX, Odroid-M1 stay at ~35–40°C cooler than H2-NCS2
  - Using NN accelerators (vs CPU) drops temperature 1–2°C on average

Platform temperature readings come from:
  - Jetson:  /sys/devices/virtual/thermal/thermal_zone*/temp
  - ARM SBCs: /sys/class/thermal/thermal_zoneN/temp
  - Intel:   coretemp via lm-sensors
"""

import os
import re
import subprocess
import threading
import time
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Thermal zone reader
# ---------------------------------------------------------------------------

def read_thermal_zones() -> Dict[str, float]:
    """
    Read all available thermal zones from sysfs.

    Returns:
        dict: {zone_name: temperature_celsius}
    """
    zones: Dict[str, float] = {}
    thermal_base = "/sys/class/thermal"
    if not os.path.isdir(thermal_base):
        return zones

    for entry in sorted(os.listdir(thermal_base)):
        if not entry.startswith("thermal_zone"):
            continue
        zone_path = os.path.join(thermal_base, entry)
        temp_file = os.path.join(zone_path, "temp")
        type_file = os.path.join(zone_path, "type")

        try:
            with open(temp_file) as f:
                temp_mc = int(f.read().strip())   # millicelsius
            zone_type = "unknown"
            if os.path.exists(type_file):
                with open(type_file) as f:
                    zone_type = f.read().strip()
            zones[f"{entry}_{zone_type}"] = temp_mc / 1000.0  # → Celsius
        except (IOError, ValueError):
            pass

    return zones


def read_cpu_temperature() -> Optional[float]:
    """
    Return the average CPU temperature across all thermal zones.
    Returns None if temperature cannot be read.
    """
    zones = read_thermal_zones()
    cpu_temps = [
        v for k, v in zones.items()
        if any(kw in k.lower() for kw in ["cpu", "core", "soc", "thermal"])
    ]
    if not cpu_temps:
        # Fall back to all zones average
        if zones:
            return sum(zones.values()) / len(zones)
        return None
    return sum(cpu_temps) / len(cpu_temps)


def read_nvidia_gpu_temperature() -> Optional[float]:
    """Return NVIDIA GPU temperature via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=3
        )
        if result.returncode == 0:
            return float(result.stdout.strip())
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Continuous temperature monitor
# ---------------------------------------------------------------------------

class TemperatureMonitor:
    """
    Continuously samples platform temperature during inference.
    Used to compute the average temperature metric in §5.2.

    Usage:
        monitor = TemperatureMonitor(interval_s=0.5)
        monitor.start()
        runner.infer(...)   # run your inference here
        readings = monitor.stop()
        avg_temp = sum(readings) / len(readings)
    """

    def __init__(self, interval_s: float = 0.5, zone_name: Optional[str] = None):
        """
        Args:
            interval_s:  Sampling interval in seconds
            zone_name:   Specific thermal zone to monitor (None = auto-detect best)
        """
        self.interval_s = interval_s
        self.zone_name  = zone_name
        self._readings: List[float] = []
        self._running   = False
        self._thread: Optional[threading.Thread] = None

    def _read_temperature(self) -> Optional[float]:
        """Read a single temperature sample."""
        # Try GPU first for GPU inference workloads
        gpu_temp = read_nvidia_gpu_temperature()
        if gpu_temp is not None:
            return gpu_temp
        return read_cpu_temperature()

    def start(self):
        """Begin temperature sampling in a background thread."""
        self._readings = []
        self._running  = True
        self._thread   = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            temp = self._read_temperature()
            if temp is not None:
                self._readings.append(temp)
            time.sleep(self.interval_s)

    def stop(self) -> List[float]:
        """Stop sampling and return all readings."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        return list(self._readings)

    @property
    def avg_temperature(self) -> Optional[float]:
        if not self._readings:
            return None
        return sum(self._readings) / len(self._readings)

    @property
    def max_temperature(self) -> Optional[float]:
        return max(self._readings) if self._readings else None


# ---------------------------------------------------------------------------
# Paper-published thermal results (for comparison/reproduction)
# Figure 5: Relative average temperature vs baseline (Odroid-M1)
# ---------------------------------------------------------------------------

# Approximate actual temperatures (°C) inferred from Figure 5 data
# Baseline (M1) is the 0 reference; values are DELTA °C from M1
PAPER_TEMPERATURE_DELTAS_C = {
    # CPU FP32 (Figure 5a)
    "CPU_FP32": {
        "AGX":  -10.0,    # ~10°C cooler than M1
        "VIM3":   2.0,
        "TX2":    5.0,
        "NX":     0.0,
        "H2":    21.0,    # H2 runs hot (Celeron SoC with poor cooling)
        "Nano":   3.0,
    },
    # GPU/NPU FP16 (Figure 5b)
    "GPU_FP16": {
        "AGX":     -5.0,
        "NX":      -2.0,
        "AGX_DLA":  0.0,
        "NX_DLA":   1.0,
        "TX2":      3.0,
        "H2_NCS2": 30.0,   # H2+NCS2 runs dangerously hot
        "NCS2":    30.0,
        "Nano":     6.0,
    },
    # INT8 (Figure 5c) — M1_NPU = 0 baseline
    "INT8": {
        "AGX":      -4.0,
        "NX":       -2.5,
        "AGX_DLA":  -1.0,
        "NX_DLA":    1.0,
        "VIM3_NPU":  4.0,
        "VIM3_GPU":  3.5,
        "M1_GPU":    2.0,
    }
}

# Critical finding: H2-NCS2 can reach 85°C during sustained inference
H2_NCS2_MAX_TEMP_C = 85.0
M1_BASELINE_APPROX_C = 42.0   # Approximate absolute M1 temperature during inference


def print_temperature_analysis():
    """Print the temperature analysis matching §5.2 of the paper."""
    print("=" * 65)
    print("Temperature Analysis — Bang for the Buck (SEC '23, §5.2)")
    print("=" * 65)
    print(f"\n  Baseline: Odroid-M1 ≈ {M1_BASELINE_APPROX_C}°C (set as ΔT = 0)\n")

    for mode, deltas in PAPER_TEMPERATURE_DELTAS_C.items():
        print(f"\n  Mode: {mode}")
        for platform, delta in sorted(deltas.items(), key=lambda x: x[1]):
            sign   = "+" if delta >= 0 else ""
            bar    = "▓" * abs(int(delta / 2))
            warn   = " ⚠ THERMAL RISK!" if M1_BASELINE_APPROX_C + delta > 80 else ""
            est_c  = M1_BASELINE_APPROX_C + delta
            print(f"    {platform:<12} {sign}{delta:+.1f}°C  (~{est_c:.0f}°C){warn}")

    print()
    print(f"  H2+NCS2 peak temperature: {H2_NCS2_MAX_TEMP_C}°C  ← near thermal limit!")
    print("  Key: Using NN accelerators reduces temperature 1–2°C vs CPU inference")
    print("  because accelerators complete inference faster (less cumulative heat).")


if __name__ == "__main__":
    print("=== Current platform temperatures ===")
    zones = read_thermal_zones()
    if zones:
        for zone, temp in zones.items():
            print(f"  {zone}: {temp:.1f}°C")
    else:
        print("  (no sysfs thermal zones available on this host)")

    cpu_t = read_cpu_temperature()
    print(f"\n  CPU temperature: {cpu_t}°C")

    gpu_t = read_nvidia_gpu_temperature()
    print(f"  GPU temperature: {gpu_t}°C")

    print()
    print_temperature_analysis()
