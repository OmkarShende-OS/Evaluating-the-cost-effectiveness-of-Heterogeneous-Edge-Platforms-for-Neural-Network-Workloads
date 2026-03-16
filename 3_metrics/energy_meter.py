"""
energy_meter.py  — Per-inference energy measurement for edge platforms.

"Bang for the Buck" (SEC '23) §5.2 measures energy consumption per inference.
Different platforms require different power monitoring methods:

  • Jetson (AGX/NX/TX2/Nano)  → tegrastats (Nvidia tool) or jtop
  • Khadas VIM3 / Odroid       → INA219 power sensor via I2C
  • Intel NCS2 (via Odroid H2) → no direct measurement; use platform total

Energy per inference (Joules) = Average Power (Watts) × Total Time (seconds)

This module provides:
  • TegrastatsMonitor   — for Jetson platforms via subprocess
  • INA219Monitor       — for ARM SBCs via smbus2 / i2c (or stub)
  • StubMonitor         — software fallback using nvml / psutil
  • EnergyMeter         — unified context-manager interface
"""

import os
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from typing import List, Optional


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class PowerMonitorBase(ABC):
    """Read platform power consumption in Watts."""

    @abstractmethod
    def start(self):
        """Begin continuous power sampling."""

    @abstractmethod
    def stop(self) -> List[float]:
        """Stop sampling. Return list of watt readings taken while running."""

    def average_power_watts(self) -> Optional[float]:
        """Convenience: start → run callback → stop → return mean power."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Tegrastats monitor (NVIDIA Jetson platforms)
# ---------------------------------------------------------------------------

class TegrastatsMonitor(PowerMonitorBase):
    """
    Launch `tegrastats --interval <ms>` as a subprocess and parse power lines.

    Tegrastats output example:
        ... POM_5V_IN 4203/4000 POM_5V_CPU 1234/1234 POM_5V_GPU 2500/2500 ...

    The module sums 'POM_5V_IN' (total board power) for each sample.
    """

    def __init__(self, interval_ms: int = 100):
        self.interval_ms = interval_ms
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._readings: List[float] = []
        self._running = False

    def start(self):
        self._readings = []
        self._running = True
        self._proc = subprocess.Popen(
            ["tegrastats", "--interval", str(self.interval_ms)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True
        )
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        for line in self._proc.stdout:
            if not self._running:
                break
            watts = self._parse_line(line)
            if watts is not None:
                self._readings.append(watts)

    @staticmethod
    def _parse_line(line: str) -> Optional[float]:
        """
        Parse a tegrastats output line and extract total board power in Watts.
        Looks for the pattern: POM_5V_IN <current>/<average>
        """
        try:
            if "POM_5V_IN" in line:
                idx = line.index("POM_5V_IN") + len("POM_5V_IN ")
                part = line[idx:].split()[0]
                current_mw = int(part.split("/")[0])
                return current_mw / 1000.0  # mW → W
        except (ValueError, IndexError):
            pass
        return None

    def stop(self) -> List[float]:
        self._running = False
        if self._proc:
            self._proc.terminate()
            self._proc.wait(timeout=3)
        if self._thread:
            self._thread.join(timeout=3)
        return list(self._readings)


# ---------------------------------------------------------------------------
# INA219 monitor (ARM SBCs: VIM3, Odroid)
# ---------------------------------------------------------------------------

class INA219Monitor(PowerMonitorBase):
    """
    Read power from an INA219 sensor over I2C on ARM SBCs.
    Requires: pip install smbus2 adafruit-circuitpython-ina219

    The INA219 is a current/power monitor IC typically on the board's
    PMIC rail, measuring total board power consumption.
    """

    INA219_ADDRESS = 0x40    # Default I2C address
    INA219_REG_POWER = 0x03  # Power register (read × 20 μW → mW)

    def __init__(self, i2c_bus: int = 1, address: int = 0x40,
                 interval_s: float = 0.1):
        self.i2c_bus = i2c_bus
        self.address = address
        self.interval_s = interval_s
        self._readings: List[float] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._ina = None

    def _init_ina(self):
        """Initialise the INA219 sensor using adafruit library."""
        try:
            import board
            import busio
            from adafruit_ina219 import INA219
            i2c = busio.I2C(board.SCL, board.SDA)
            self._ina = INA219(i2c, addr=self.address)
        except ImportError:
            # Fallback: raw smbus2 register reads
            try:
                import smbus2
                self._bus = smbus2.SMBus(self.i2c_bus)
                self._ina = "raw"
            except ImportError:
                raise ImportError(
                    "INA219 monitoring requires: pip install adafruit-circuitpython-ina219 smbus2"
                )

    def start(self):
        self._readings = []
        self._running = True
        self._init_ina()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_power_mw(self) -> float:
        if self._ina == "raw":
            raw = self._bus.read_word_data(self.address, self.INA219_REG_POWER)
            # Swap bytes (INA219 is big-endian)
            raw = ((raw & 0xFF) << 8) | ((raw >> 8) & 0xFF)
            return raw * 20.0e-3   # LSB = 20μW → mW
        else:
            return self._ina.power  # adafruit returns mW directly

    def _read_loop(self):
        while self._running:
            try:
                mw = self._read_power_mw()
                self._readings.append(mw / 1000.0)  # mW → W
            except Exception:
                pass
            time.sleep(self.interval_s)

    def stop(self) -> List[float]:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        return list(self._readings)


# ---------------------------------------------------------------------------
# Stub monitor (software-only; uses psutil / pynvml)
# ---------------------------------------------------------------------------

class StubMonitor(PowerMonitorBase):
    """
    Fallback power monitor using software tools.
    Accuracy is lower than hardware monitors but sufficient for testing.

    Tries (in order):
      1. pynvml NVML (NVIDIA GPU power)
      2. psutil CPU power estimate from CPU freq × TDP scaling
      3. Returns 0.0 (if no measurement possible)
    """

    def __init__(self, interval_s: float = 0.1):
        self.interval_s = interval_s
        self._readings: List[float] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._nvml_handle = None

    def _try_init_nvml(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._pynvml = pynvml
        except Exception:
            self._nvml_handle = None

    def start(self):
        self._readings = []
        self._running = True
        self._try_init_nvml()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_current_power(self) -> float:
        if self._nvml_handle:
            try:
                mw = self._pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle)
                return mw / 1000.0  # mW → W
            except Exception:
                pass
        # Rough CPU estimate: use /sys/class/powercap if available (Intel RAPL)
        rapl_path = "/sys/class/powercap/intel-rapl:0/energy_uj"
        if hasattr(self, "_last_rapl") and os.path.exists(rapl_path):
            try:
                with open(rapl_path) as f:
                    curr_uj = int(f.read())
                if hasattr(self, "_last_rapl_uj"):
                    delta_uj = curr_uj - self._last_rapl_uj
                    watts = delta_uj * 1e-6 / self.interval_s
                    self._last_rapl_uj = curr_uj
                    return watts
                self._last_rapl_uj = curr_uj
            except Exception:
                pass
        return 0.0  # Unknown

    def _read_loop(self):
        self._last_rapl = True  # enable RAPL on first iteration
        while self._running:
            self._readings.append(self._read_current_power())
            time.sleep(self.interval_s)

    def stop(self) -> List[float]:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        return list(self._readings)


# ---------------------------------------------------------------------------
# Unified EnergyMeter context manager
# ---------------------------------------------------------------------------

class EnergyMeter:
    """
    Context manager that measures energy consumed during a block of code.

    Usage:
        meter = EnergyMeter(monitor_type="tegrastats")
        with meter:
            runner.infer(batch)
        print(f"Energy: {meter.energy_joules:.4f} J  |  Avg power: {meter.avg_watts:.2f} W")
    """

    MONITOR_CLASSES = {
        "tegrastats": TegrastatsMonitor,
        "ina219":     INA219Monitor,
        "stub":       StubMonitor,
        "nvml":       StubMonitor,   # StubMonitor uses NVML internally
        "none":       StubMonitor,
    }

    def __init__(self, monitor_type: str = "none", **kwargs):
        MonitorClass = self.MONITOR_CLASSES.get(monitor_type, StubMonitor)
        self._monitor = MonitorClass(**kwargs)
        self._start_time: Optional[float] = None
        self._elapsed_s: float = 0.0
        self.power_readings: List[float] = []

    def __enter__(self):
        self._start_time = time.perf_counter()
        self._monitor.start()
        return self

    def __exit__(self, *args):
        self.power_readings = self._monitor.stop()
        self._elapsed_s = time.perf_counter() - self._start_time

    @property
    def avg_watts(self) -> float:
        if not self.power_readings:
            return 0.0
        return sum(self.power_readings) / len(self.power_readings)

    @property
    def energy_joules(self) -> float:
        """Energy = average power × elapsed time."""
        return self.avg_watts * self._elapsed_s

    @property
    def elapsed_ms(self) -> float:
        return self._elapsed_s * 1000.0


def make_energy_meter(platform_key: str) -> EnergyMeter:
    """
    Factory: return the appropriate EnergyMeter for a platform.
    Uses platform_configs.PLATFORM_CONFIGS to look up the monitor type.
    """
    try:
        from platform_configs import PLATFORM_CONFIGS
        monitor_type = PLATFORM_CONFIGS[platform_key].power_monitor
    except (ImportError, KeyError):
        monitor_type = "none"
    return EnergyMeter(monitor_type=monitor_type)


if __name__ == "__main__":
    print("=== EnergyMeter stub demo ===")
    meter = EnergyMeter("none")
    with meter:
        # Simulate a 50ms inference
        time.sleep(0.05)
    print(f"  Elapsed: {meter.elapsed_ms:.1f} ms")
    print(f"  Avg power: {meter.avg_watts:.2f} W  (0.0 when no sensor)")
    print(f"  Energy:    {meter.energy_joules:.6f} J")
    print()
    print("  On a Jetson: use EnergyMeter('tegrastats')")
    print("  On VIM3/Odroid: use EnergyMeter('ina219')")
