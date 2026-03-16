# Section 4: Roofline Analysis

Implements the **theoretical Roofline Model** for all 8 edge platforms.

## What the Roofline Model Tells Us

The Roofline bounds the **maximum achievable performance** of an NN workload:

```
attainable_perf (GFLOPS/s) = min(peak_compute, OI × peak_bandwidth)
```

Where **OI = FLOPS / byte** (Operational Intensity) characterises whether a workload is:
- **Memory-bound** (OI < ridge point): performance = OI × bandwidth
- **Compute-bound** (OI > ridge point): performance = peak_compute

All NN models in this study have OI ≈ 1.3–3.0 FLOPS/byte, which places them in the **memory-bound region** on all tested platforms.

## Files

| File | Purpose |
|---|---|
| `hardware_specs.py` | Peak compute & memory bandwidth from vendor datasheets for all 8 platforms |
| `operational_intensity.py` | OI computation for MobileNetV2, ResNet101, DenseNet121, Xception, YOLOv3 |
| `roofline_model.py` | Roofline construction, ridge point calculation, performance ordering |
| `plot_roofline.py` | Reproduces Figure 1 (a/b/c) from the paper |

## Theoretical Performance Ordering (from paper §4)

**CPU (FP32):**
```
AGX (2.27×) > TX2 (1.50×) > H2 (1.15×) > VIM3-Big (1.10×) > NX (1.05×) > M1 (1.0×) > Nano (0.715×) > VIM3-Little (0.45×)
```

**GPU/VPU (FP16):**
```
AGX (117×) > NX (55×) > NCS2 (23.8×) > TX2 (16×) > Nano (5.7×) > H2 (3.5×) > NVDLA (3.0×) > VIM3 (1.7×) > M1 (1.0×)
```

**NPU/GPU (INT8):**
```
AGX (24.3×) > NX (11.6×) > VIM3-NPU (5.5×) > TX2 (3.3×) > Nano (1.2×) > M1-NPU (1.0×)
```

## Key Roofline Findings

1. **All NN workloads are memory-bound** on all platforms → memory bandwidth is the bottleneck, not raw compute.
2. **Roofline ordering differs from empirical ordering** (§5) — caches, SW runtime quality, and CPU I/O overhead are not captured theoretically.
3. **NVDLA underperforms its theoretical bound** — it cannot execute all layer types and falls back to GPU.

## Run

```bash
cd 1_roofline_analysis/
python hardware_specs.py           # Print all platform specs
python operational_intensity.py    # Print Table 1 (OI values)
python roofline_model.py           # Print roofline ordering summary
python plot_roofline.py            # Generate Figure 1 → ../results/fig1_roofline.png
python plot_roofline.py --show     # Open interactive window
```
