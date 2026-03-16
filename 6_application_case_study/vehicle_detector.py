"""
vehicle_detector.py — YOLOv3 vehicle detection wrapper (§7).

Wraps two YOLOv3 variants used in the adaptive scheduler:
  1. GPU FP16   — TensorRT optimized, highest accuracy (AP=96.3%)
  2. NPU INT8   — RKNN/KSNN optimized, highest throughput (AP=90.1%)

Provides unified inference API so the heterogeneous_scheduler can
switch between paths transparently.

Detection pipeline:
  raw_image → preprocess (letterbox 608×608) → infer → decode (NMS) → detections

Paper §7 evaluation metric: mean Average Precision (mAP) on a curated
traffic video dataset with 12 vehicle classes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


# ---------------------------------------------------------------------------
# Detection output data structures
# ---------------------------------------------------------------------------

@dataclass
class BoundingBox:
    x1: float; y1: float; x2: float; y2: float
    confidence: float
    class_id: int
    class_name: str = ""

    @property
    def area(self) -> float:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    def iou(self, other: "BoundingBox") -> float:
        inter_x1 = max(self.x1, other.x1)
        inter_y1 = max(self.y1, other.y1)
        inter_x2 = min(self.x2, other.x2)
        inter_y2 = min(self.y2, other.y2)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        union_area = self.area + other.area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0


@dataclass
class DetectionResult:
    boxes: List[BoundingBox] = field(default_factory=list)
    inference_ms: float = 0.0
    backend: str = "unknown"
    input_shape: Tuple = (608, 608)


# ---------------------------------------------------------------------------
# YOLOv3 preprocessing
# ---------------------------------------------------------------------------

def letterbox_resize(
    image: np.ndarray,
    target_size: Tuple[int, int] = (608, 608),
    fill_value: int = 114,
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Letterbox resize: maintain aspect ratio, pad to square.

    Args:
        image:       BGR uint8 input
        target_size: (H, W) target
        fill_value:  Padding pixel value (YOLOv3 uses 114)

    Returns:
        (resized_padded_image, scale, (pad_top, pad_left))
    """
    if not CV2_AVAILABLE:
        h, w = image.shape[:2]
        scale = min(target_size[0] / h, target_size[1] / w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = image.astype(np.float32)[:new_h, :new_w]  # naive crop fallback
        canvas = np.full((*target_size, 3), fill_value, dtype=np.uint8)
        canvas[:resized.shape[0], :resized.shape[1]] = resized
        return canvas, scale, (0, 0)

    h, w = image.shape[:2]
    th, tw = target_size
    scale  = min(th / h, tw / w)
    new_h  = int(h * scale)
    new_w  = int(w * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas  = np.full((th, tw, 3), fill_value, dtype=np.uint8)
    pad_top  = (th - new_h) // 2
    pad_left = (tw - new_w) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    return canvas, scale, (pad_top, pad_left)


def preprocess_yolov3(
    image: np.ndarray,
    input_size: Tuple[int, int] = (608, 608),
    normalize: bool = True,
) -> Tuple[np.ndarray, dict]:
    """
    YOLOv3 standard preprocessing: letterbox + BGR→RGB + normalize.

    Returns:
        (input_tensor NCHW float32, meta dict for postprocessing)
    """
    lb_image, scale, (pad_top, pad_left) = letterbox_resize(image, input_size)

    if CV2_AVAILABLE:
        rgb = cv2.cvtColor(lb_image, cv2.COLOR_BGR2RGB)
    else:
        rgb = lb_image[:, :, ::-1]

    if normalize:
        tensor = rgb.astype(np.float32) / 255.0
    else:
        tensor = rgb.astype(np.float32)

    # NHWC → NCHW
    tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis]

    meta = {
        "scale":    scale,
        "pad_top":  pad_top,
        "pad_left": pad_left,
        "orig_h":   image.shape[0],
        "orig_w":   image.shape[1],
    }
    return tensor, meta


# ---------------------------------------------------------------------------
# NMS utility
# ---------------------------------------------------------------------------

def nms(boxes: List[BoundingBox], iou_threshold: float = 0.45) -> List[BoundingBox]:
    """Standard NMS: remove overlapping boxes, keep highest confidence."""
    if not boxes:
        return []
    boxes_sorted = sorted(boxes, key=lambda b: -b.confidence)
    kept: List[BoundingBox] = []
    for box in boxes_sorted:
        if all(box.iou(k) < iou_threshold for k in kept if k.class_id == box.class_id):
            kept.append(box)
    return kept


# ---------------------------------------------------------------------------
# Main detector class
# ---------------------------------------------------------------------------

class YOLOv3VehicleDetector:
    """
    Unified YOLOv3 vehicle detector supporting GPU FP16 and NPU INT8 backends.

    Internally wraps the appropriate FrameworkRunner based on backend selection.

    Backends (as in paper §7):
      "gpu_fp16" → TensorRT FP16 on Jetson GPU (or OpenVINO GPU FP16 on H2)
      "npu_int8" → RKNN INT8 on Odroid M1 / KSNN INT8 on VIM3

    Usage:
        detector = YOLOv3VehicleDetector("gpu_fp16", onnx_path="yolov3.onnx")
        detector.load()
        result = detector.detect(bgr_frame)
        for box in result.boxes:
            print(box.class_name, box.confidence)
    """

    COCO_VEHICLE_CLASSES = {
        2: "car", 3: "motorcycle", 5: "bus", 7: "truck",
        0: "person",  # also relevant in traffic
    }

    # YOLOv3 anchor boxes (standard COCO anchors, 3 scales)
    ANCHORS = [
        [(10, 13), (16, 30), (33, 23)],       # small
        [(30, 61), (62, 45), (59, 119)],      # medium
        [(116, 90), (156, 198), (373, 326)],  # large
    ]

    def __init__(
        self,
        backend: str = "gpu_fp16",
        onnx_path: Optional[str] = None,
        rknn_path: Optional[str] = None,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.45,
        input_size: Tuple[int, int] = (608, 608),
    ):
        self.backend        = backend
        self.onnx_path      = onnx_path
        self.rknn_path      = rknn_path
        self.conf_threshold = conf_threshold
        self.nms_threshold  = nms_threshold
        self.input_size     = input_size

        self._runner = None

    def load(self) -> None:
        """Load the appropriate runner based on backend."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

        if self.backend == "gpu_fp16":
            try:
                from bang_for_the_buck.four_sw_framework_comparison.runners.tensorrt_runner import TensorRTRunner
                self._runner = TensorRTRunner(self.onnx_path or "yolov3.onnx", precision="fp16")
                self._runner.load()
            except (ImportError, RuntimeError):
                logger.warning("TensorRT not available, using synthetic stub")
                self._runner = _StubRunner("gpu_fp16")

        elif self.backend == "npu_int8":
            try:
                from bang_for_the_buck.four_sw_framework_comparison.runners.rknn_runner import RKNNRunner
                self._runner = RKNNRunner(self.rknn_path or "yolov3_int8.rknn")
                self._runner.load()
            except (ImportError, RuntimeError):
                logger.warning("RKNN not available, using synthetic stub")
                self._runner = _StubRunner("npu_int8")
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        logger.info("YOLOv3 detector loaded — backend=%s", self.backend)

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """
        Run vehicle detection on a single frame.

        Args:
            frame: BGR uint8 image

        Returns:
            DetectionResult with bounding boxes
        """
        import time
        if self._runner is None:
            raise RuntimeError("Call load() first")

        tensor, meta = preprocess_yolov3(frame, self.input_size)

        t0 = time.perf_counter()
        raw_output = self._runner.infer(tensor)
        infer_ms = (time.perf_counter() - t0) * 1000

        boxes = self._decode_output(raw_output, meta)
        boxes = nms(boxes, self.nms_threshold)

        return DetectionResult(
            boxes        = boxes,
            inference_ms = infer_ms,
            backend      = self.backend,
            input_shape  = self.input_size,
        )

    def _decode_output(self, raw: np.ndarray, meta: dict) -> List[BoundingBox]:
        """
        Decode YOLOv3 raw output to bounding boxes.

        Simple flat-output decoder (post-NMS single output head format).
        Full multi-scale YOLOv3 decoding would require per-scale sigmoid + anchor.
        """
        boxes = []
        if raw is None or raw.size == 0:
            return boxes

        raw = raw.reshape(-1, 85)  # [N, 85] for COCO: 80 classes + 4 box + 1 obj
        for det in raw:
            obj_conf = float(det[4])
            if obj_conf < self.conf_threshold:
                continue
            class_scores = det[5:]
            class_id     = int(np.argmax(class_scores))
            class_conf   = float(class_scores[class_id]) * obj_conf
            if class_conf < self.conf_threshold:
                continue
            if class_id not in self.COCO_VEHICLE_CLASSES:
                continue

            # Decode box (cx, cy, w, h) → (x1, y1, x2, y2) in original image coords
            cx, cy, bw, bh = det[:4]
            scale    = meta["scale"]
            pad_top  = meta["pad_top"]
            pad_left = meta["pad_left"]
            ih, iw   = self.input_size

            x1 = (cx * iw - bw * iw / 2 - pad_left) / scale
            y1 = (cy * ih - bh * ih / 2 - pad_top)  / scale
            x2 = (cx * iw + bw * iw / 2 - pad_left) / scale
            y2 = (cy * ih + bh * ih / 2 - pad_top)  / scale

            boxes.append(BoundingBox(
                x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                confidence=class_conf,
                class_id=class_id,
                class_name=self.COCO_VEHICLE_CLASSES[class_id],
            ))
        return boxes

    def close(self) -> None:
        if self._runner:
            self._runner.close()


class _StubRunner:
    """Stub runner for testing without hardware."""
    def __init__(self, name: str):
        self.name = name
    def load(self): pass
    def infer(self, x): return np.zeros((1, 100, 85), dtype=np.float32)
    def close(self): pass


# ---------------------------------------------------------------------------
# Paper §7 accuracy results
# ---------------------------------------------------------------------------

PAPER_YOLOV3_ACCURACY = {
    "description": "YOLOv3 vehicle detection accuracy and throughput (Table 7, §7)",
    "GPU_FP16": {
        "AP_pct": 96.3,
        "FPS": 14.2,
        "avg_latency_ms": 70.4,
        "n_detections_per_frame": 4.8,
        "note": "Used for HEAVY traffic (high density → high AP needed)",
    },
    "NPU_INT8": {
        "AP_pct": 90.1,
        "FPS": 38.7,
        "avg_latency_ms": 25.8,
        "note": "Used for LIGHT/MODERATE traffic (throughput prioritized)",
    },
    "Adaptive": {
        "AP_pct": 93.9,
        "FPS": 28.4,
        "avg_latency_ms": 35.2,
        "note": "Adaptive scheduler: GPU for 30% heavy frames, NPU for 70% others",
    },
}


if __name__ == "__main__":
    print("YOLOv3 Vehicle Detector — Bang for the Buck §7")
    print()
    print("Paper Table 7 — Adaptive detection results:")
    for mode, d in PAPER_YOLOV3_ACCURACY.items():
        if "AP_pct" in d:
            print(f"  {mode:<10} AP={d['AP_pct']:.1f}%  FPS={d['FPS']:.1f}  "
                  f"lat={d['avg_latency_ms']:.1f}ms  | {d['note']}")
    print()
    print("Key finding: Adaptive scheduler achieves 93.9% AP (3.4pt drop vs GPU)")
    print("             but 2× the FPS (28.4 vs 14.2) — best accuracy-efficiency trade-off")
