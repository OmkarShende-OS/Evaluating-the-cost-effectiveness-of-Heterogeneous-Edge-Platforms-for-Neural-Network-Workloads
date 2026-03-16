"""Power and thermal profile utilities."""

from __future__ import annotations

from statistics import mean
from typing import Iterable


def summarize_power_temperature(power_w: Iterable[float], temperature_c: Iterable[float]) -> dict:
    power_values = list(power_w)
    temp_values = list(temperature_c)
    return {
        "avg_power_w": round(mean(power_values), 4) if power_values else None,
        "avg_temperature_c": round(mean(temp_values), 4) if temp_values else None,
        "max_temperature_c": max(temp_values) if temp_values else None,
    }
