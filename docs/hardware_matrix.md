# Hardware Matrix

| Device Class | Example Platforms | Accelerator Type | Measurement Support | Notes |
|---|---|---|---|---|
| Jetson-class | AGX/NX/TX2/Nano style | CPU/GPU/DLA | latency, throughput, power, thermal | Strong vendor runtime ecosystem |
| ARM SBC class | Odroid/Khadas/Rockchip style | CPU/NPU/GPU | latency, throughput, thermal, optional power | Cost-efficient deployments |
| x86 + VPU class | Odroid H2 + NCS2 style | CPU/VPU | latency, throughput, optional power | Useful for CPU+accelerator concurrency studies |

## Precision support (typical)

- FP32: broadest compatibility
- FP16: acceleration on capable GPUs/NPUs/VPUs
- INT8: highest efficiency where quantized model/runtime support exists
