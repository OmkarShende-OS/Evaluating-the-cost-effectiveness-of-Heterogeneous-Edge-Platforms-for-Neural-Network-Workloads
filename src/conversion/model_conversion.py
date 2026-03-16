"""Model conversion hook stubs.

Public repo includes a lightweight abstraction to avoid embedding
vendor-proprietary conversion logic directly.
"""


def list_supported_conversions() -> list[str]:
    return [
        "onnx -> tensorrt",
        "onnx -> openvino",
        "onnx -> tflite",
        "onnx -> rknn",
    ]
