"""
model_configs.py — NN model metadata and model loading helpers for the
benchmarks in "Bang for the Buck" (SEC '23).

Covers the 5 models from Table 1 of the paper used for empirical evaluation.
Models are loaded from Keras Applications and exported to TensorFlow SavedModel
format so that each runtime adapter (TFLite, TensorRT, OpenVINO…) can convert
them to its native format.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


@dataclass
class NNModelConfig:
    """Configuration for a single neural network model."""
    name: str
    keras_name: str             # Name in keras.applications
    input_shape: Tuple[int, int, int]  # (H, W, C)
    parameters_M: float
    depth: int
    model_size_mb: float        # Unoptimised FP32 size
    total_macs_M: float         # Million MACs at 224×224
    task: str                   # "classification" or "detection"
    top1_accuracy_fp32: float   # ImageNet top-1 (paper Table result)
    notes: str = ""

    # OI per precision (from Table 1)
    oi: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Table 1: models benchmarked in the paper
# ---------------------------------------------------------------------------
MODEL_CONFIGS: Dict[str, NNModelConfig] = {
    "mobilenetv2": NNModelConfig(
        name="MobileNet-V2",
        keras_name="MobileNetV2",
        input_shape=(224, 224, 3),
        parameters_M=3.5,
        depth=105,
        model_size_mb=14.0,
        total_macs_M=300.0,
        task="classification",
        top1_accuracy_fp32=71.8,
        oi={"FP32": 1.33, "FP16": 1.63, "INT8": 1.93},
        notes="Depthwise separable convolutions — highly memory-efficient"
    ),
    "resnet101v2": NNModelConfig(
        name="ResNet101-V2",
        keras_name="ResNet101V2",
        input_shape=(224, 224, 3),
        parameters_M=44.7,
        depth=205,
        model_size_mb=171.0,
        total_macs_M=7800.0,
        task="classification",
        top1_accuracy_fp32=77.2,
        oi={"FP32": 1.60, "FP16": 1.90, "INT8": 2.20},
        notes="Deep residual network — most parameters of the classification set"
    ),
    "densenet121": NNModelConfig(
        name="DenseNet-121",
        keras_name="DenseNet121",
        input_shape=(224, 224, 3),
        parameters_M=8.1,
        depth=242,
        model_size_mb=33.0,
        total_macs_M=2900.0,
        task="classification",
        top1_accuracy_fp32=75.0,
        oi={"FP32": 1.94, "FP16": 2.24, "INT8": 2.54},
        notes="Densely connected — deepest of the classification models"
    ),
    "xception": NNModelConfig(
        name="Xception",
        keras_name="Xception",
        input_shape=(299, 299, 3),    # Xception uses 299×299 natively
        parameters_M=22.9,
        depth=81,
        model_size_mb=88.0,
        total_macs_M=8400.0,
        task="classification",
        top1_accuracy_fp32=79.0,
        oi={"FP32": 1.96, "FP16": 2.26, "INT8": 2.56},
        notes="Extreme Inception — separable convs throughout; primary benchmark model"
    ),
    "yolov3": NNModelConfig(
        name="YOLOv3",
        keras_name=None,               # Not in keras.applications — use darknet weights
        input_shape=(416, 416, 3),
        parameters_M=62.2,
        depth=106,
        model_size_mb=246.6,
        total_macs_M=65900.0,
        task="detection",
        top1_accuracy_fp32=0.0,        # Detection uses mAP/AP, not top-1
        oi={"FP32": 2.42, "FP16": 2.72, "INT8": 3.02},
        notes="Object detection — used in application case study (§7)"
    ),
}

# Shorthand aliases
ALL_CLASSIFICATION_MODELS = [k for k, v in MODEL_CONFIGS.items()
                               if v.task == "classification"]
ALL_MODELS = list(MODEL_CONFIGS.keys())


def get_model_config(model_key: str) -> NNModelConfig:
    """Get NNModelConfig by key (case-insensitive)."""
    key = model_key.lower().replace("-", "").replace("_", "")
    for k, v in MODEL_CONFIGS.items():
        if k.replace("_", "").replace("-", "") == key:
            return v
    raise KeyError(f"Unknown model: {model_key}. Available: {list(MODEL_CONFIGS)}")


def load_keras_model(model_key: str):
    """
    Load and return the Keras model for a given key.
    Requires tensorflow ≥ 2.4.

    Args:
        model_key: Key from MODEL_CONFIGS (e.g. "mobilenetv2")

    Returns:
        tf.keras.Model instance (pre-trained on ImageNet, FP32)
    """
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("TensorFlow not installed. Run: pip install tensorflow")

    cfg = get_model_config(model_key)
    if cfg.keras_name is None:
        raise ValueError(f"{cfg.name} is not available from keras.applications")

    model_loader = getattr(tf.keras.applications, cfg.keras_name)
    model = model_loader(weights="imagenet", include_top=True,
                         input_shape=cfg.input_shape)
    return model


def export_saved_model(model_key: str, output_dir: str) -> str:
    """
    Export a Keras model to TensorFlow SavedModel format for further
    conversion to TFLite, TensorRT, OpenVINO etc.

    Args:
        model_key:   Key from MODEL_CONFIGS
        output_dir:  Directory to save the model

    Returns:
        Path to the saved model directory
    """
    model = load_keras_model(model_key)
    cfg = get_model_config(model_key)
    save_path = os.path.join(output_dir, cfg.keras_name)
    model.save(save_path)
    print(f"Saved {cfg.name} → {save_path}")
    return save_path


def print_model_table():
    """Print Table 1 from the paper."""
    print(f"{'Model':<16} {'Params(M)':>9} {'Depth':>6} {'Size(MB)':>9} "
          f"{'OI_FP32':>8} {'OI_FP16':>8} {'OI_INT8':>8}")
    print("-" * 75)
    for cfg in MODEL_CONFIGS.values():
        print(f"{cfg.name:<16} {cfg.parameters_M:>9.1f} {cfg.depth:>6} "
              f"{cfg.model_size_mb:>9.0f} "
              f"{cfg.oi.get('FP32', 0):>8.2f} "
              f"{cfg.oi.get('FP16', 0):>8.2f} "
              f"{cfg.oi.get('INT8', 0):>8.2f}")


if __name__ == "__main__":
    print("=== NN Model Configurations (Table 1, Bang for the Buck) ===\n")
    print_model_table()
    print()
    print(f"Classification models: {ALL_CLASSIFICATION_MODELS}")
    print(f"All models:            {ALL_MODELS}")
