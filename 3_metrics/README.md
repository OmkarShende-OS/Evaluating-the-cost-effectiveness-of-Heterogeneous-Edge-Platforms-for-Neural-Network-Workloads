# 3_metrics — §5.2 Hardware Performance Metrics

This module implements all hardware performance metrics defined in **Section 5.2** of the
*Evaluating the Cost-Effectiveness of Heterogeneous Edge Platforms for Neural Network Workloads* paper.

## Metrics

| File | Metric | Description |
|------|--------|-------------|
| `inference_time.py` | Inference latency (ms) | Pure accelerator execution time |
| `total_time.py` | End-to-end latency (ms) | Load + Preprocess + Infer + Postprocess |
| `energy_meter.py` | Energy per inference (J) | Power × time using Tegrastats/INA219 |
| `temperature_monitor.py` | Thermal delta (°C) | Temperature rise under inference workload |
| `edcp.py` | EDCP (normalised) | Energy × Delay × Cost Product — **novel metric** |

---

## Quick usage

```python
import numpy as np
from inference_time import warmup_and_measure
from energy_meter import EnergyMeter, make_energy_meter
from edcp import compute_edcp, rank_platforms_by_edcp

# 1 — time inference
stats = warmup_and_measure(lambda: model.predict(x), warmup_iters=20, measure_iters=100)
print(f"median = {stats['median_ms']:.1f} ms,  FPS = {stats['fps']:.1f}")

# 2 — measure energy
meter = make_energy_meter(platform="agx")
with meter:
    for _ in range(100):
        model.predict(x)
print(f"energy/inference = {meter.energy_joules / 100:.4f} J")

# 3 — compute EDCP
edcp = compute_edcp(
    energy_J=meter.energy_joules / 100,
    latency_s=stats['mean_ms'] / 1000,
    cost_usd=399.0,    # AGX Xavier developer kit price
)
print(f"EDCP = {edcp:.4f}  J·s·$")
```

---

## Paper Findings (§5.2, §6, §7)

### Inference time (Figure 2)
- **GPU FP16** fastest absolute, but 24× slower on AGX CPU vs M1 NPU baseline
- **INT8** closes the gap — M1 NPU approaches GPU speed at a fraction of the power
- **DLA** (Jetson Deep Learning Accelerator) is 3–8× slower than GPU but 40% lower power

### End-to-end (Figure 3)
- CPU bottleneck consumes up to **53%** of AGX GPU speedup
- AGX GPU: 24.6× inference speedup → only 11.6× total speedup
- TX2, NX show similar patterns (Table 5)

### Energy (Figure 4)
- Odroid M1 NPU achieves best energy efficiency at INT8
- Jetson AGX GPU (FP16) uses 15× more energy per inference than M1 NPU

### Temperature (Figure 5)
- Odroid H2 + NCS2 reaches **85 °C** — highest thermal risk
- NVIDIA DLA is 1–2 °C cooler than GPU at same throughput
- VIM3 NPU runs 3 °C cooler than VIM3 Big CPU

### EDCP (novel metric — Section 5.2)
Normalised EDCP ordering (M1 = 1.0 baseline):

```
M1    1.00  ██
VIM3  1.92  ████
NX    2.32  █████
Nano  4.93  ██████████
NCS2  6.01  ████████████
AGX   7.86  ████████████████
TX2  11.13  ██████████████████████
H2   15.38  ███████████████████████████████
```

**Key insight**: AGX Xavier is the *worst* cost-efficiency despite the highest raw throughput.
The Odroid M1 provides the best energy-delay-cost trade-off for INT8 workloads.
