"""
benchmark_runner.py — Main benchmark orchestrator for "Bang for the Buck" (SEC '23).

Executes the full measurement pipeline (§5.1 Experimental Setup):
  1. Platform detection
  2. Model loading + framework initialisation
  3. Warmup (20 iterations, discarded)
  4. Measure inference_time, total_time, energy, temperature
  5. Save results to JSON

Usage:
    # Single run: MobileNetV2 on TFLite CPU FP32
    python benchmark_runner.py --model mobilenetv2 --framework tflite \
           --processor CPU --precision FP32

    # Sweep all combos for the current platform
    python benchmark_runner.py --sweep

    # Use ImageNet val set
    python benchmark_runner.py --model xception --imagenet-dir /data/imagenet/val

    # Save results
    python benchmark_runner.py --sweep --output results/my_platform.json
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

# Local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from model_configs import MODEL_CONFIGS, get_model_config
from platform_configs import get_platform_config, detect_platform, PLATFORM_CONFIGS
from imagenet_loader import make_loader


# ---------------------------------------------------------------------------
# Framework runner registry
# Runners are loaded lazily to avoid hard dependencies on unavailable libs
# ---------------------------------------------------------------------------

FRAMEWORK_RUNNER_MAP = {
    "tflite":       "4_sw_framework_comparison.runners.tflite_runner.TFLiteRunner",
    "tensorrt":     "4_sw_framework_comparison.runners.tensorrt_runner.TensorRTRunner",
    "openvino":     "4_sw_framework_comparison.runners.openvino_runner.OpenVINORunner",
    "armnn":        "4_sw_framework_comparison.runners.armnn_runner.ARMNNRunner",
    "pyarmnn":      "4_sw_framework_comparison.runners.pyarmnn_runner.PyARMNNRunner",
    "autoschedule": "4_sw_framework_comparison.runners.autoschedule_runner.AutoScheduleRunner",
    "ksnn":         "4_sw_framework_comparison.runners.ksnn_runner.KSNNRunner",
    "rknn":         "4_sw_framework_comparison.runners.rknn_runner.RKNNRunner",
}


def _load_runner_class(framework_key: str):
    """Dynamically import a runner class by its dotted path."""
    module_path, class_name = FRAMEWORK_RUNNER_MAP[framework_key].rsplit(".", 1)
    # Adjust sys.path to find the module relative to this file
    repo_root = os.path.join(os.path.dirname(__file__), "..")
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


# ---------------------------------------------------------------------------
# BenchmarkResult dataclass
# ---------------------------------------------------------------------------

class BenchmarkResult:
    """Holds all measurements for one (model, framework, processor, precision) run."""

    def __init__(self, model: str, framework: str, processor: str, precision: str,
                 platform: str):
        self.model = model
        self.framework = framework
        self.processor = processor
        self.precision = precision
        self.platform = platform
        self.num_images = 0
        self.warmup_done = False
        # §5.2 metrics
        self.inference_times_ms: List[float] = []   # per-image inference only
        self.total_times_ms: List[float] = []        # full load+pre+infer+post
        self.energy_joules: List[float] = []         # per-image energy
        self.temperature_c: List[float] = []         # platform avg temp
        self.top1_correct = 0
        self.errors: List[str] = []

    def record(self, infer_ms: float, total_ms: float,
               energy_j: Optional[float] = None,
               temp_c: Optional[float] = None,
               correct: bool = False):
        self.inference_times_ms.append(infer_ms)
        self.total_times_ms.append(total_ms)
        if energy_j is not None:
            self.energy_joules.append(energy_j)
        if temp_c is not None:
            self.temperature_c.append(temp_c)
        if correct:
            self.top1_correct += 1
        self.num_images += 1

    def summary(self) -> Dict[str, Any]:
        """Return statistics matching the paper's reported metrics."""
        def stats(arr):
            if not arr:
                return {}
            a = np.array(arr)
            return {
                "mean": float(np.mean(a)),
                "median": float(np.median(a)),
                "p90": float(np.percentile(a, 90)),
                "p99": float(np.percentile(a, 99)),
                "std": float(np.std(a)),
                "min": float(np.min(a)),
                "max": float(np.max(a)),
            }

        # Throughput (FPS) = 1000 / mean_inference_time_ms
        fps_infer  = 1000.0 / np.mean(self.inference_times_ms) if self.inference_times_ms else 0
        fps_total  = 1000.0 / np.mean(self.total_times_ms) if self.total_times_ms else 0
        top1_acc   = self.top1_correct / max(self.num_images, 1) * 100.0
        avg_energy = float(np.mean(self.energy_joules)) if self.energy_joules else None
        avg_temp   = float(np.mean(self.temperature_c)) if self.temperature_c else None

        return {
            "model":       self.model,
            "framework":   self.framework,
            "processor":   self.processor,
            "precision":   self.precision,
            "platform":    self.platform,
            "num_images":  self.num_images,
            "inference_time": stats(self.inference_times_ms),
            "total_time":     stats(self.total_times_ms),
            "throughput_fps_inference":  round(fps_infer, 2),
            "throughput_fps_total":      round(fps_total, 2),
            "energy_joules_per_frame":   avg_energy,
            "avg_temperature_c":         avg_temp,
            "top1_accuracy_pct":         round(top1_acc, 2) if self.top1_correct > 0 else None,
        }


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """
    Orchestrates model loading, warmup, and measurement across frameworks.

    Implements §5.1 Experimental Setup of the paper.
    """

    def __init__(
        self,
        platform_key: Optional[str] = None,
        imagenet_dir: Optional[str] = None,
        warmup_iters: int = 20,
        measure_iters: int = 100,
        output_path: Optional[str] = None,
        verbose: bool = True,
    ):
        self.platform_key = platform_key or detect_platform() or "odroid_m1"
        self.platform_cfg = get_platform_config(self.platform_key)
        self.imagenet_dir = imagenet_dir
        self.warmup_iters = warmup_iters
        self.measure_iters = measure_iters
        self.output_path = output_path
        self.verbose = verbose
        self.results: List[Dict] = []

    def _log(self, msg: str):
        if self.verbose:
            print(f"[BenchmarkRunner] {msg}")

    def run_single(
        self,
        model_key: str,
        framework_key: str,
        processor_name: str,
        precision: str,
        model_path: Optional[str] = None,
    ) -> BenchmarkResult:
        """
        Run a single benchmark combination.

        Args:
            model_key:      Key from MODEL_CONFIGS (e.g. "mobilenetv2")
            framework_key:  Key from FRAMEWORK_RUNNER_MAP (e.g. "tflite")
            processor_name: Processor name (e.g. "M1_CPU", "NX_GPU")
            precision:      "FP32", "FP16", or "INT8"
            model_path:     Path to model file (optional; auto-resolve if None)
        """
        model_cfg = get_model_config(model_key)
        result = BenchmarkResult(
            model=model_cfg.name,
            framework=framework_key,
            processor=processor_name,
            precision=precision,
            platform=self.platform_cfg.short_name,
        )

        self._log(f"Setting up: {model_cfg.name} | {framework_key} | "
                  f"{processor_name} | {precision}")

        # Load runner
        try:
            RunnerClass = _load_runner_class(framework_key)
        except Exception as e:
            result.errors.append(f"Runner load failed: {e}")
            self._log(f"  ERROR: {e}")
            return result

        # Instantiate runner with model path
        if model_path is None:
            model_path = self._resolve_model_path(model_cfg, precision, framework_key)

        runner = RunnerClass(
            model_path=model_path,
            precision=precision,
            processor=processor_name,
        )

        try:
            runner.load()
        except Exception as e:
            result.errors.append(f"Model load failed: {e}")
            self._log(f"  ERROR loading model: {e}")
            return result

        # Data loader
        loader = make_loader(
            self.imagenet_dir,
            model_name=model_cfg.keras_name or model_cfg.name,
            input_shape=model_cfg.input_shape,
            max_images=self.warmup_iters + self.measure_iters,
            synthetic_fallback=True,
        )

        images = list(loader)
        warmup_imgs  = images[:self.warmup_iters]
        measure_imgs = images[self.warmup_iters:self.warmup_iters + self.measure_iters]

        # Warmup
        self._log(f"  Warmup ({self.warmup_iters} iters)…")
        for batch, _ in warmup_imgs:
            _ = runner.infer(batch)
        result.warmup_done = True

        # Measurement
        self._log(f"  Measuring ({len(measure_imgs)} images)…")
        for batch, img_path in measure_imgs:
            t_total_start = time.perf_counter()

            # Inference only
            t_infer_start = time.perf_counter()
            output = runner.infer(batch)
            t_infer_end = time.perf_counter()

            t_total_end = time.perf_counter()

            infer_ms = (t_infer_end - t_infer_start) * 1000.0
            total_ms = (t_total_end - t_total_start) * 1000.0

            # Energy metric (if monitor available)
            metrics = runner.get_metrics() if hasattr(runner, "get_metrics") else {}
            energy_j = metrics.get("energy_joules")
            temp_c   = metrics.get("temperature_c")

            result.record(infer_ms, total_ms, energy_j, temp_c)

        runner.close()

        summary = result.summary()
        self._log(f"  ✓ {summary['throughput_fps_inference']:.1f} FPS (inference), "
                  f"{summary['throughput_fps_total']:.1f} FPS (total)")
        return result

    def run_sweep(self, model_keys: Optional[List[str]] = None) -> List[Dict]:
        """
        Run all supported (model, framework, processor, precision) combinations
        for the current platform.
        """
        platform_combos = self.platform_cfg.supported_combos
        if model_keys is None:
            model_keys = list(MODEL_CONFIGS.keys())

        all_results = []
        total = len(model_keys) * len(platform_combos)
        done = 0

        for model_key in model_keys:
            for framework, processor, precision in platform_combos:
                done += 1
                self._log(f"[{done}/{total}] {model_key} | {framework} | "
                          f"{processor} | {precision}")
                result = self.run_single(model_key, framework, processor, precision)
                data = result.summary()
                if result.errors:
                    data["errors"] = result.errors
                all_results.append(data)

        self.results = all_results

        if self.output_path:
            self._save(all_results)

        return all_results

    def _resolve_model_path(self, model_cfg, precision: str,
                             framework_key: str) -> str:
        """
        Attempt to resolve a model path from common locations.
        Users should place converted models in a 'models/' directory.
        """
        ext_map = {
            "tflite":      f"{model_cfg.keras_name}_{'quant' if precision=='INT8' else 'fp16' if precision=='FP16' else 'fp32'}.tflite",
            "tensorrt":    f"{model_cfg.keras_name}_{precision.lower()}.engine",
            "openvino":    f"{model_cfg.keras_name}_{precision.lower()}.xml",
            "armnn":       f"{model_cfg.keras_name}_{precision.lower()}.armnn",
            "pyarmnn":     f"{model_cfg.keras_name}_{precision.lower()}.armnn",
            "autoschedule":f"{model_cfg.keras_name}.onnx",
            "ksnn":        f"{model_cfg.keras_name}.nb",
            "rknn":        f"{model_cfg.keras_name}.rknn",
        }
        fname = ext_map.get(framework_key, f"{model_cfg.keras_name}.model")
        model_dir = os.path.join(os.path.dirname(__file__), "..", "results", "models")
        return os.path.join(model_dir, fname)

    def _save(self, results: List[Dict]):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "w") as f:
            json.dump(results, f, indent=2)
        self._log(f"Results saved → {self.output_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Bang for the Buck — benchmark runner (SEC '23 §5.1)"
    )
    parser.add_argument("--model",     type=str, default="mobilenetv2",
                        choices=list(MODEL_CONFIGS),
                        help="NN model to benchmark")
    parser.add_argument("--framework", type=str, default="tflite",
                        choices=list(FRAMEWORK_RUNNER_MAP),
                        help="Inference framework")
    parser.add_argument("--processor", type=str, default="M1_CPU",
                        help="Target processor name (e.g. M1_CPU, NX_GPU)")
    parser.add_argument("--precision", type=str, default="FP32",
                        choices=["FP32", "FP16", "INT8"])
    parser.add_argument("--sweep",     action="store_true",
                        help="Run all combos for the detected platform")
    parser.add_argument("--platform",  type=str, default=None,
                        choices=list(PLATFORM_CONFIGS),
                        help="Override platform detection")
    parser.add_argument("--imagenet-dir", type=str, default=None,
                        help="Path to ImageNet validation directory")
    parser.add_argument("--warmup",    type=int, default=20,
                        help="Warmup iterations (paper uses 20)")
    parser.add_argument("--iters",     type=int, default=100,
                        help="Measurement iterations (paper uses 100)")
    parser.add_argument("--output",    type=str, default=None,
                        help="JSON output path for results")
    args = parser.parse_args()

    runner = BenchmarkRunner(
        platform_key=args.platform,
        imagenet_dir=args.imagenet_dir,
        warmup_iters=args.warmup,
        measure_iters=args.iters,
        output_path=args.output,
    )

    if args.sweep:
        results = runner.run_sweep()
        print(f"\n=== Sweep complete: {len(results)} experiments ===")
    else:
        result = runner.run_single(
            args.model, args.framework, args.processor, args.precision
        )
        import pprint
        pprint.pprint(result.summary())


if __name__ == "__main__":
    main()
