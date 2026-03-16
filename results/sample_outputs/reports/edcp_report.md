# EDCP Comparison Report

| Device | Runtime | EDCP | Normalized EDCP |
|---|---|---:|---:|
| Odroid-M1 | tflite-cpu | 0.102137 | 1.0000 |
| Odroid-M1 | rknn-npu | 0.023226 | 0.2274 |
| Jetson-AGX | tensorrt-gpu | 0.266333 | 2.6076 |
| Jetson-AGX | tensorrt-int8 | 0.225974 | 2.2125 |
| Odroid-H2 | openvino-vpu | 0.057772 | 0.5656 |
