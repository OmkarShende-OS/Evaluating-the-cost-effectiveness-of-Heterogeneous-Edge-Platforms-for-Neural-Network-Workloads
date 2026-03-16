# Bang for the Buck: Evaluating Cost-Effectiveness of Heterogeneous Edge Platforms for Neural Network Workloads

> **Published at ACM/IEEE SEC '23** — Eighth Symposium on Edge Computing, December 6–9, 2023, Wilmington, DE, USA  
> DOI: [10.1145/3583740.3628437](https://doi.org/10.1145/3583740.3628437)  
> Authors: Amarjeet Saini\*, **Omkar B Shende**\*, Mohammad Khalid Pandit, Rijurekha Sen, Gayathri Ananthanarayanan  
> (\* Equal contribution)

---

## What This Work Is About

Deploying neural networks at the edge means navigating a complex stack of choices:

- **Which NN model?** (MobileNet vs. ResNet vs. Xception vs. YOLOv3)
- **Which edge platform?** (Jetson AGX/NX/TX2/Nano, Khadas VIM3, Odroid H2/M1 + Intel NCS2)
- **Which software runtime?** (TensorRT, OpenVINO, ARM NN, TFLite, AutoScheduler, KSNN, RKNN)
- **Which processor?** (CPU, GPU, NPU, VPU — or all of them concurrently?)
- **What metric matters?** (Throughput, energy, cost, accuracy, temperature — or all of them?)

This paper is the **largest empirical benchmarking study of edge ML** at the time of publication, covering **8 hardware platforms**, **8+ software runtimes**, **20 processing elements**, and **12+ NN models**. We go beyond measuring performance numbers — we find actionable insights and introduce a novel metric.

---

## Key Contributions

| Contribution | Where |
|---|---|
| 📐 Roofline model for 8 edge platforms (CPU/GPU/NPU/VPU) | [`1_roofline_analysis/`](1_roofline_analysis/) |
| 🔬 Comprehensive latency/energy/thermal benchmarking framework | [`2_benchmark_harness/`](2_benchmark_harness/) + [`3_metrics/`](3_metrics/) |
| 💡 **Novel EDCP metric** (Energy × Delay × Cost Product) | [`3_metrics/edcp.py`](3_metrics/edcp.py) |
| 🖥️ Finding: vendor SW ≠ best SW (AutoScheduler beats TensorRT by 4×) | [`4_sw_framework_comparison/`](4_sw_framework_comparison/) |
| ⚡ Finding: CPU bottleneck limits GPU accelerators | [`3_metrics/total_time.py`](3_metrics/total_time.py) |
| 🔀 Data-Wise Workload Allocation (DWA) across co-processors | [`5_concurrent_execution/dwa/`](5_concurrent_execution/dwa/) |
| 🧩 Layer-Wise Workload Allocation (LWA) with subgraph splitting | [`5_concurrent_execution/lwa/`](5_concurrent_execution/lwa/) |
| 🚗 Real-world traffic detection with adaptive GPU↔NPU scheduling | [`6_application_case_study/`](6_application_case_study/) |
| 📊 All paper figures reproduced | [`visualizations/`](visualizations/) |

---

## Repository Structure

```
bang-for-the-buck/
├── 1_roofline_analysis/          # §4  Theoretical roofline for all 8 platforms
│   ├── hardware_specs.py         #     Peak compute & memory bandwidth from datasheets
│   ├── operational_intensity.py  #     OI (FLOPS/byte) for NN models at FP32/FP16/INT8
│   ├── roofline_model.py         #     Roofline construction + performance ordering
│   └── plot_roofline.py          #     Reproduces Figure 1 from the paper
│
├── 2_benchmark_harness/          # §5.1 Experimental setup
│   ├── benchmark_runner.py       #     Main benchmark loop: load → warmup → measure
│   ├── model_configs.py          #     Table 1: NN model metadata
│   ├── platform_configs.py       #     Table 2: hardware platform specs
│   └── imagenet_loader.py        #     ImageNet 224×224 data loader
│
├── 3_metrics/                    # §5.2 Hardware performance analysis
│   ├── inference_time.py         #     Inference-only latency measurement
│   ├── total_time.py             #     End-to-end latency (shows CPU bottleneck)
│   ├── energy_meter.py           #     Energy per inference via tegrastats/ina219
│   ├── temperature_monitor.py    #     Thermal tracking across platforms
│   └── edcp.py                   #     ★ EDCP metric: Energy × Delay × Cost
│
├── 4_sw_framework_comparison/    # §5.3 Software stack performance analysis
│   ├── framework_runner.py       #     Unified experiment runner
│   ├── quantization_analysis.py  #     FP32 → FP16 → INT8 accuracy vs speed
│   ├── runners/
│   │   ├── tensorrt_runner.py    #     NVIDIA TensorRT
│   │   ├── autoschedule_runner.py#     Apache AutoScheduler (beats TensorRT!)
│   │   ├── armnn_runner.py       #     ARM NN (C++ subprocess)
│   │   ├── pyarmnn_runner.py     #     PyARMNN Python wrapper
│   │   ├── tflite_runner.py      #     TensorFlow Lite
│   │   ├── openvino_runner.py    #     Intel OpenVINO
│   │   ├── ksnn_runner.py        #     Khadas KSNN (VIM3 NPU)
│   │   └── rknn_runner.py        #     Rockchip RKNN (Odroid M1 NPU)
│
├── 5_concurrent_execution/       # §6  Concurrent execution on heterogeneous co-processors
│   ├── dwa/
│   │   ├── data_wise_allocation.py   # DWA strategy: proportional data split
│   │   ├── odroid_h2_ncs2_dwa.py    # H2+NCS2 via OpenVINO MULTI plugin
│   │   └── vim3_dwa.py              # VIM3 manual DWA (ARMNN + KSNN)
│   └── lwa/
│       ├── layer_wise_allocation.py  # LWA orchestrator + layer profiler
│       ├── lwa_r.py                  # LWA-R: minimum runtime per layer
│       ├── lwa_p.py                  # LWA-P: subgraph with highest parameters
│       └── lwa_c.py                  # LWA-C: subgraph with highest compute
│
├── 6_application_case_study/     # §7  Vehicle detection on Indian traffic dataset
│   ├── traffic_density.py        #     CPU-based background subtraction classifier
│   ├── vehicle_detector.py       #     YOLOv3 wrapper for GPU (FP32) and NPU (INT8)
│   ├── heterogeneous_scheduler.py#     Adaptive CPU→GPU/NPU switching logic
│   └── demo.py                   #     End-to-end demo reproducing Table 7 + Figure 12
│
├── visualizations/               # Paper figures reproduced
│   ├── fig1_roofline.py
│   ├── fig2_inference_time.py
│   ├── fig3_total_time.py
│   ├── fig4_energy.py
│   ├── fig5_temperature.py
│   ├── fig6_sw_odroid.py
│   ├── fig7_sw_vim3.py
│   ├── fig8_tensorrt_vs_autosched.py
│   ├── fig9_dwa_odroid.py
│   ├── fig10_dwa_vim3.py
│   └── fig12_app_workflow.py
│
├── results/                      # Pre-computed results (JSON/CSV)
│   ├── roofline_data.json
│   ├── inference_time_results.json
│   └── edcp_results.json
│
└── requirements.txt
```

---

## Skills Demonstrated

| Skill | Evidence |
|---|---|
| **Embedded systems / SoC architecture** | Characterizing 8 SoCs: Jetson, VIM3, Odroid, NCS2 across CPU/GPU/NPU/VPU |
| **ML inference optimization** | TensorRT, OpenVINO, ARM NN, AutoScheduler, TFLite, KSNN, RKNN |
| **Quantization (FP32/FP16/INT8)** | Section 5.3, `quantization_analysis.py` — INT8 drops 2–4% accuracy on most models |
| **Roofline performance modeling** | `1_roofline_analysis/` — theoretical vs. empirical bounds |
| **Energy/power measurement** | `tegrastats`, INA219, `jtop` for real Joules-per-inference |
| **Parallel / concurrent programming** | DWA (data parallel), LWA (task/layer parallel) across co-processors |
| **Computer vision (object detection)** | YOLOv3 on a novel Delhi traffic intersection dataset |
| **Novel metrics design** | EDCP (Energy × Delay × Cost Product) — outperforms single-metric comparisons |
| **Statistical benchmarking** | >100-iteration runs, p50/p90/p99, outlier detection, coefficient of variation |
| **Python scientific stack** | NumPy, Matplotlib, OpenCV, TensorFlow / PyTorch, subprocess, threading |

---

## Key Findings

1. **CPU is the bottleneck**: A 24× GPU inference speedup collapses to 11× for end-to-end due to weak driving CPUs.
2. **Vendor SW ≠ Best SW**: AutoScheduler gives ~4× more throughput than NVIDIA TensorRT on AGX, with *less* energy per frame.
3. **EDCP reveals Odroid-M1 wins**: Despite low specs, M1 is the most *cost-effective* platform (EDCP = 1.0 baseline).
4. **DWA works for off-chip, not on-chip**: H2+NCS2 (USB VPU) gets 43% latency reduction; VIM3 (on-chip) gets none.
5. **LWA is always slower**: Inter-processor communication overhead (2.5–25% of E2E time) negates compute savings.
6. **Heterogeneity is still useful**: Use CPU to sense traffic density, route to GPU for accuracy or NPU for speed.

---

## Platforms Tested

| Platform | CPU | GPU/Accelerator | TOPS | Cost |
|---|---|---|---|---|
| Jetson Xavier AGX | 8-core Carmel ARM v8.2 | 512-core Volta + 2× NVDLA | — | $999 |
| Jetson Xavier NX | 6-core Carmel ARM v8.2 | 384-core Volta + 2× NVDLA | — | $479 |
| Jetson TX2 | 2× Denver + 4× Cortex-A57 | 256-core Pascal | — | $479 |
| Jetson Nano | 4× Cortex-A57 | 128-core Maxwell | — | $119 |
| Khadas VIM3 | 4× Cortex-A73 + 2× A53 | Mali-G52 MP4 + Amlogic NPU | 5 TOPS | $159 |
| Odroid H2 + NCS2 | 4× Intel Celeron J4105 | Intel UHD 600 + Myriad X VPU | — | $130 |
| Odroid M1 | 4× Cortex-A55 | Mali-G52 MP2 + Rockchip NPU | 0.8 TOPS | $95 |

---

## Citation

```bibtex
@inproceedings{saini2023bang,
  title={Bang for the Buck: Evaluating the cost-effectiveness of Heterogeneous Edge Platforms for Neural Network Workloads},
  author={Saini, Amarjeet and Shende, Omkar B and Pandit, Mohammad Khalid and Sen, Rijurekha and Ananthanarayanan, Gayathri},
  booktitle={Proceedings of the Eighth ACM/IEEE Symposium on Edge Computing (SEC '23)},
  pages={94--107},
  year={2023},
  doi={10.1145/3583740.3628437}
}
```
