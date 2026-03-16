"""
imagenet_loader.py — ImageNet validation dataset loader for edge inference
benchmarks in "Bang for the Buck" (SEC '23).

The paper uses ImageNet validation images resized to 224×224×3 (RGB).
This module provides a lightweight loader that:
  - Reads JPEG images from a validation directory
  - Applies standard ImageNet preprocessing per model family
  - Yields batches (batch_size=1 for latency benchmarking)
  - Optionally generates random synthetic data for throughput testing

Expected directory structure:
    imagenet/val/
        n01440764/   (synset ID)
            ILSVRC2012_val_00000293.JPEG
            ...
        n01443537/
            ...

Or flat:
    imagenet/val/
        *.JPEG
"""

import os
import random
from typing import Callable, Generator, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Preprocessing functions per model family
# ---------------------------------------------------------------------------

def preprocess_caffe(img: np.ndarray) -> np.ndarray:
    """
    Caffe-style preprocessing: BGR, subtract ImageNet mean.
    Used by VGG, ResNet v1.
    """
    # Already RGB from PIL; convert to BGR channel order
    img = img[:, :, ::-1].astype(np.float32)
    mean = np.array([103.939, 116.779, 123.68], dtype=np.float32)
    img -= mean
    return img


def preprocess_tf(img: np.ndarray) -> np.ndarray:
    """
    TF-style: scale to [-1, 1].
    Used by MobileNetV2, Xception, Inception.
    """
    img = img.astype(np.float32)
    img = (img / 127.5) - 1.0
    return img


def preprocess_torch(img: np.ndarray) -> np.ndarray:
    """
    PyTorch-style: divide by 255, subtract mean, divide by std.
    Used by DenseNet, ResNet v2 in some runtimes.
    """
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    return img


# Map from keras model name → preprocessing function
PREPROCESS_MAP: dict = {
    "MobileNetV2":  preprocess_tf,
    "Xception":     preprocess_tf,
    "InceptionV3":  preprocess_tf,
    "ResNet101V2":  preprocess_caffe,
    "DenseNet121":  preprocess_torch,
    "YOLOv3":       lambda x: x.astype(np.float32) / 255.0,
}


def get_preprocessor(model_name: str) -> Callable:
    """Return the correct preprocessing function for a model."""
    for key, fn in PREPROCESS_MAP.items():
        if key.lower() in model_name.lower():
            return fn
    # Default: tf-style normalize
    return preprocess_tf


# ---------------------------------------------------------------------------
# Image loaders
# ---------------------------------------------------------------------------

def _resize_and_load(path: str, target_h: int, target_w: int) -> np.ndarray:
    """
    Load an image file and resize to (target_h, target_w, 3).
    Uses PIL (preferred) or OpenCV as fallback.
    """
    try:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        img = img.resize((target_w, target_h), Image.BILINEAR)
        return np.array(img, dtype=np.uint8)
    except ImportError:
        pass

    try:
        import cv2
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"cv2 could not read: {path}")
        img = cv2.resize(img, (target_w, target_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    except ImportError:
        pass

    raise ImportError("Install Pillow or OpenCV: pip install Pillow opencv-python")


class ImageNetLoader:
    """
    Streams preprocessed ImageNet images for inference benchmarking.

    Args:
        val_dir:      Path to the ImageNet validation directory
        model_name:   Keras model name (determines preprocessing)
        input_shape:  Target (H, W, C) input shape
        max_images:   Maximum images to load (default: use all)
        shuffle:      Shuffle image order
    """

    def __init__(
        self,
        val_dir: str,
        model_name: str = "MobileNetV2",
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        max_images: int = 1000,
        shuffle: bool = False,
    ):
        self.val_dir = val_dir
        self.preprocess = get_preprocessor(model_name)
        self.input_h, self.input_w, self.input_c = input_shape
        self.max_images = max_images
        self.shuffle = shuffle
        self._image_paths: Optional[List[str]] = None

    def _discover_images(self) -> List[str]:
        """Walk val_dir and collect all JPEG/PNG paths."""
        paths = []
        for root, _, files in os.walk(self.val_dir):
            for fname in files:
                if fname.lower().endswith((".jpeg", ".jpg", ".png")):
                    paths.append(os.path.join(root, fname))
        if self.shuffle:
            random.shuffle(paths)
        return paths[:self.max_images]

    @property
    def image_paths(self) -> List[str]:
        if self._image_paths is None:
            self._image_paths = self._discover_images()
        return self._image_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __iter__(self) -> Generator[Tuple[np.ndarray, str], None, None]:
        """
        Yield (preprocessed_image_CHW, image_path) tuples.
        Shape: (1, C, H, W) for frameworks expecting NCHW,
               (1, H, W, C) for frameworks expecting NHWC.
        This yields NHWC by default; transpose in the runner if needed.
        """
        for path in self.image_paths:
            try:
                raw   = _resize_and_load(path, self.input_h, self.input_w)
                proc  = self.preprocess(raw)                        # (H, W, C)
                batch = proc[np.newaxis, ...]                       # (1, H, W, C)
                yield batch, path
            except Exception as e:
                print(f"[WARN] Skipping {path}: {e}")
                continue

    def get_random_batch(self, batch_size: int = 1) -> np.ndarray:
        """Return a random preprocessed batch — useful for warmup."""
        paths = random.sample(self.image_paths, min(batch_size, len(self.image_paths)))
        imgs = []
        for path in paths:
            raw  = _resize_and_load(path, self.input_h, self.input_w)
            proc = self.preprocess(raw)
            imgs.append(proc)
        return np.stack(imgs, axis=0)  # (B, H, W, C)


class SyntheticLoader:
    """
    Generates random synthetic data for throughput benchmarking
    without requiring the actual ImageNet dataset.
    Used for quick validation of runtime adapters.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        num_images: int = 200,
        seed: int = 42,
    ):
        self.input_shape = input_shape
        self.num_images = num_images
        rng = np.random.default_rng(seed)
        H, W, C = input_shape
        # Pre-generate all images (already preprocessed as normalised floats)
        self._data = rng.standard_normal((num_images, H, W, C)).astype(np.float32)

    def __len__(self) -> int:
        return self.num_images

    def __iter__(self) -> Generator[Tuple[np.ndarray, str], None, None]:
        for i, img in enumerate(self._data):
            yield img[np.newaxis, ...], f"synthetic_{i:05d}"

    def get_random_batch(self, batch_size: int = 1) -> np.ndarray:
        indices = np.random.choice(self.num_images, batch_size, replace=False)
        return self._data[indices]


def make_loader(
    val_dir: Optional[str],
    model_name: str = "MobileNetV2",
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    max_images: int = 1000,
    synthetic_fallback: bool = True,
) -> "ImageNetLoader | SyntheticLoader":
    """
    Factory: return an ImageNetLoader if val_dir exists, else SyntheticLoader.

    Args:
        val_dir:            Path to ImageNet val directory (can be None)
        model_name:         For preprocessing selection
        input_shape:        Target (H, W, C)
        max_images:         Maximum images to use
        synthetic_fallback: If True and val_dir missing, use synthetic data
    """
    if val_dir and os.path.isdir(val_dir):
        print(f"[INFO] Using ImageNet loader from {val_dir}")
        return ImageNetLoader(val_dir, model_name, input_shape, max_images)
    if synthetic_fallback:
        print(f"[INFO] ImageNet dir not found — using synthetic data "
              f"(accuracy metrics will be invalid)")
        return SyntheticLoader(input_shape, min(max_images, 200))
    raise FileNotFoundError(f"ImageNet val directory not found: {val_dir}")


if __name__ == "__main__":
    print("=== SyntheticLoader demo ===")
    loader = SyntheticLoader(input_shape=(224, 224, 3), num_images=10)
    print(f"  Total images: {len(loader)}")
    for batch, name in loader:
        print(f"  {name}: shape={batch.shape}, "
              f"min={batch.min():.3f}, max={batch.max():.3f}")
