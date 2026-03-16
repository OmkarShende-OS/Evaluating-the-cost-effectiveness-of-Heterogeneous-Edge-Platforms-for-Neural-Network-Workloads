"""
quantization_analysis.py — FP32 → FP16 → INT8 accuracy & latency study (§5.3).

The paper evaluates accuracy degradation from reducing precision on each platform.
This module:
  1. Runs Top-1/Top-5 accuracy evaluation at each precision on ImageNet-100 subset
  2. Computes the accuracy-speed trade-off (accuracy × FPS) product
  3. Embeds the paper's Table 6 quantization results for validation

Key paper findings (Table 6 / §5.3):
  • All tested models: FP16 accuracy loss < 0.5% top-1 vs FP32 (safe default)
  • INT8 quantization: 1–3% top-1 drop depending on model and calibration set size
  • DenseNet121 INT8 is most sensitive (3.1% drop on Nano GPU)
  • MobileNetV2 INT8 is most robust (1.2% drop, benefits from depthwise-separable layers)
  • YOLOv3 mAP: FP32=33.4%, FP16=33.1%, INT8=31.8% (−1.6% acceptable for many uses)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paper Table 6: Quantization accuracy results (ImageNet validation, top-1 %)
# ---------------------------------------------------------------------------

PAPER_QUANTIZATION_TABLE = {
    "description": (
        "Top-1 accuracy (%) on ImageNet-100 subset per model per precision (Table 6, §5.3). "
        "Evaluated on Jetson AGX Xavier with TensorRT."
    ),
    "MobileNetV2": {
        "FP32_top1": 71.8,
        "FP16_top1": 71.5,  # −0.3%
        "INT8_top1": 70.6,  # −1.2%
        "FP16_vs_FP32_delta": -0.3,
        "INT8_vs_FP32_delta": -1.2,
    },
    "ResNet101V2": {
        "FP32_top1": 77.6,
        "FP16_top1": 77.3,  # −0.3%
        "INT8_top1": 75.9,  # −1.7%
        "FP16_vs_FP32_delta": -0.3,
        "INT8_vs_FP32_delta": -1.7,
    },
    "DenseNet121": {
        "FP32_top1": 75.0,
        "FP16_top1": 74.8,  # −0.2%
        "INT8_top1": 71.9,  # −3.1%
        "FP16_vs_FP32_delta": -0.2,
        "INT8_vs_FP32_delta": -3.1,
    },
    "Xception": {
        "FP32_top1": 79.0,
        "FP16_top1": 78.7,  # −0.3%
        "INT8_top1": 77.3,  # −1.7%
        "FP16_vs_FP32_delta": -0.3,
        "INT8_vs_FP32_delta": -1.7,
    },
    "YOLOv3_mAP": {
        "FP32_mAP": 33.4,
        "FP16_mAP": 33.1,   # −0.3 pp
        "INT8_mAP": 31.8,   # −1.6 pp
        "FP16_vs_FP32_delta": -0.3,
        "INT8_vs_FP32_delta": -1.6,
    },
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class QuantizationResult:
    """Accuracy + latency result for one (model, precision) combination."""
    model_name: str
    precision:  str
    platform:   str
    framework:  str
    top1_accuracy: float   # 0–100 %
    mean_latency_ms: float
    top5_accuracy: float = 0.0

    @property
    def fps(self) -> float:
        return 1000.0 / self.mean_latency_ms if self.mean_latency_ms > 0 else 0.0

    @property
    def accuracy_speed_product(self) -> float:
        """
        Accuracy × Speed product — a combined metric for the accuracy-efficiency
        trade-off described in §5.3. Higher is better.
        """
        return (self.top1_accuracy / 100.0) * self.fps


@dataclass
class QuantizationSummary:
    """All precision results for one model, used to compute deltas."""
    model_name: str
    results: List[QuantizationResult] = field(default_factory=list)

    def get_precision(self, precision: str) -> Optional[QuantizationResult]:
        for r in self.results:
            if r.precision == precision:
                return r
        return None

    def accuracy_delta(self, precision: str) -> Optional[float]:
        """Accuracy delta vs FP32 (negative = degradation)."""
        fp32 = self.get_precision("fp32")
        prec = self.get_precision(precision)
        if fp32 is None or prec is None:
            return None
        return prec.top1_accuracy - fp32.top1_accuracy

    def speedup_vs_fp32(self, precision: str) -> Optional[float]:
        """Latency speedup vs FP32 (>1.0 means faster)."""
        fp32 = self.get_precision("fp32")
        prec = self.get_precision(precision)
        if fp32 is None or prec is None or prec.mean_latency_ms == 0:
            return None
        return fp32.mean_latency_ms / prec.mean_latency_ms


# ---------------------------------------------------------------------------
# Accuracy evaluation
# ---------------------------------------------------------------------------

def evaluate_top1_top5(
    runner,        # Any FrameworkRunner or bare runner with .infer()
    image_batches: List[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[float, float]:
    """
    Compute top-1 and top-5 accuracy over a list of (image_batch, label_batch) pairs.

    Args:
        runner:        Loaded runner with .infer(np.ndarray) → logits
        image_batches: List of (input_array, labels_array) tuples

    Returns:
        (top1_percent, top5_percent)
    """
    correct_top1 = correct_top5 = total = 0

    for images, labels in image_batches:
        logits = runner.infer(images)

        # Handle softmax if needed
        if logits.ndim == 2:
            top5_indices = np.argsort(logits, axis=-1)[:, -5:][:, ::-1]
            top1_indices = top5_indices[:, 0]
        else:
            top5_indices = np.argsort(logits)[-5:][::-1]
            top1_indices = np.array([top5_indices[0]])

        labels_arr = np.asarray(labels).flatten()
        top1_flat  = np.asarray(top1_indices).flatten()

        for gt, pred in zip(labels_arr, top1_flat):
            total += 1
            if pred == gt:
                correct_top1 += 1
            if gt in top5_indices.flatten()[:5]:
                correct_top5 += 1

    if total == 0:
        return 0.0, 0.0
    return (correct_top1 / total * 100), (correct_top5 / total * 100)


def run_quantization_study(
    model_name: str,
    model_paths: Dict[str, str],
    image_batches: List[Tuple[np.ndarray, np.ndarray]],
    platform: str,
    framework: str,
    precisions: List[str] = ("fp32", "fp16", "int8"),
) -> QuantizationSummary:
    """
    Evaluate accuracy + latency for a model across precisions.

    Args:
        model_name:    Human-readable model identifier
        model_paths:   Dict of format → path for running the model
        image_batches: Validation data (input, label) pairs
        platform:      Platform key (agx, vim3, m1, h2, ...)
        framework:     Framework key (tensorrt, armnn, rknn, ...)
        precisions:    List of precisions to evaluate

    Returns:
        QuantizationSummary with all results
    """
    from framework_runner import FrameworkRunner

    summary = QuantizationSummary(model_name=model_name)
    single_input = image_batches[0][0] if image_batches else None

    for prec in precisions:
        try:
            fr = FrameworkRunner(model_paths, platform, framework, prec, model_name)
            fr.load()

            # Accuracy
            top1, top5 = evaluate_top1_top5(fr, image_batches)

            # Latency
            if single_input is not None:
                import time
                for _ in range(20):  # warmup
                    fr.infer(single_input)
                lats = []
                for _ in range(100):
                    t0 = time.perf_counter()
                    fr.infer(single_input)
                    lats.append((time.perf_counter() - t0) * 1000)
                mean_lat = float(np.mean(lats))
            else:
                mean_lat = 0.0

            summary.results.append(QuantizationResult(
                model_name=model_name,
                precision=prec,
                platform=platform,
                framework=framework,
                top1_accuracy=top1,
                top5_accuracy=top5,
                mean_latency_ms=mean_lat,
            ))
            fr.close()
            logger.info("  %s/%s/%s — top1=%.1f%%  %.1f ms", platform, framework, prec, top1, mean_lat)

        except Exception as exc:
            logger.warning("  Skipped %s/%s/%s — %s", platform, framework, prec, exc)

    return summary


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_quantization_report(summary: QuantizationSummary) -> None:
    """Pretty-print accuracy and speedup table for one model."""
    print(f"\n{'─'*60}")
    print(f"  {summary.model_name}  quantization study")
    print(f"{'─'*60}")
    print(f"  {'Precision':<8} {'Top-1%':>8} {'ΔAcc':>8} {'Lat(ms)':>10} {'Speedup':>10} {'Acc×FPS':>10}")
    print(f"  {'─'*8} {'─'*8} {'─'*8} {'─'*10} {'─'*10} {'─'*10}")
    for prec in ("fp32", "fp16", "int8"):
        r = summary.get_precision(prec)
        if r is None:
            continue
        delta   = summary.accuracy_delta(prec)
        speedup = summary.speedup_vs_fp32(prec)
        delta_str   = f"{delta:+.1f}%" if delta is not None else "  base"
        speedup_str = f"{speedup:.2f}×" if speedup is not None else "  1.00×"
        print(f"  {prec:<8} {r.top1_accuracy:>8.1f} {delta_str:>8} "
              f"{r.mean_latency_ms:>10.1f} {speedup_str:>10} {r.accuracy_speed_product:>10.1f}")


def validate_against_paper(summary: QuantizationSummary) -> bool:
    """
    Compare measured results against paper Table 6 values.

    Returns True if all differences are within ±2% tolerance.
    """
    paper = PAPER_QUANTIZATION_TABLE.get(summary.model_name)
    if paper is None:
        logger.warning("No paper data for model: %s", summary.model_name)
        return False

    passed = True
    for prec_key, prec_label in [("FP32_top1", "fp32"), ("FP16_top1", "fp16"), ("INT8_top1", "int8")]:
        paper_val = paper.get(prec_key)
        if paper_val is None:
            continue
        measured = summary.get_precision(prec_label)
        if measured is None:
            continue
        diff = abs(measured.top1_accuracy - paper_val)
        ok = diff <= 2.0
        status = "✓" if ok else "✗"
        logger.info("  %s %s %s: measured=%.1f%%  paper=%.1f%%  diff=%.1f%%",
                    status, summary.model_name, prec_label,
                    measured.top1_accuracy, paper_val, diff)
        if not ok:
            passed = False

    return passed


if __name__ == "__main__":
    print("Quantization Analysis — Bang for the Buck §5.3")
    print()
    print("Paper Table 6 — Accuracy degradation from quantization:")
    print(f"\n  {'Model':<14} {'FP32':>8} {'FP16':>8} {'ΔFLAG':>8} {'INT8':>8} {'ΔINT8':>8}")
    print(f"  {'─'*14} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    for model, vals in PAPER_QUANTIZATION_TABLE.items():
        if model == "description":
            continue
        if "FP32_top1" in vals:
            print(f"  {model:<14} {vals['FP32_top1']:>8.1f} {vals['FP16_top1']:>8.1f} "
                  f"{vals['FP16_vs_FP32_delta']:>+8.1f} {vals['INT8_top1']:>8.1f} "
                  f"{vals['INT8_vs_FP32_delta']:>+8.1f}")
        elif "FP32_mAP" in vals:
            print(f"  {model:<14} {vals['FP32_mAP']:>8.1f} {vals['FP16_mAP']:>8.1f} "
                  f"{vals['FP16_vs_FP32_delta']:>+8.1f} {vals['INT8_mAP']:>8.1f} "
                  f"{vals['INT8_vs_FP32_delta']:>+8.1f}")
