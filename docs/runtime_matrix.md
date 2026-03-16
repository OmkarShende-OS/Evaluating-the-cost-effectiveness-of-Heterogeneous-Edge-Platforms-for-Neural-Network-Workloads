# Runtime Matrix

| Runtime / Backend | Typical Targets | Precision Modes | Notes |
|---|---|---|---|
| ONNX Runtime | CPU/GPU (EP-dependent) | FP32/FP16/INT8 (backend-dependent) | Good portability baseline |
| TensorRT | NVIDIA GPUs | FP32/FP16/INT8 | High performance for NVIDIA deployments |
| OpenVINO | Intel CPU/GPU/VPU | FP32/FP16/INT8 | Common for NCS2/VPU-style studies |
| TensorFlow Lite | CPU/NPU delegates | FP32/FP16/INT8 | Broad embedded/mobile usage |
| ARM NN / PyARMNN | ARM devices | FP32/FP16/INT8 (delegate-dependent) | Useful for ARM-centric deployment studies |
| KSNN / RKNN class | Vendor NPUs | INT8-centric | Device-specific SDK constraints |

## Runtime-selection dimension

The repository includes `examples/runtime_selection_demo.py` to show policy-style runtime choice based on latency, energy, and deployment goals.
