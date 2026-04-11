"""
inference.py — YOLOv8 inference engine with a drop-in mock.

Switching between mock and real model:
  - Set MODEL_PATH=mock in .env  → uses MockYoloModel (no GPU, no large deps)
  - Set MODEL_PATH=yolov8n.pt    → loads the real Ultralytics YOLOv8 model

The real model block is kept behind a try/import so the service starts cleanly
even when `ultralytics` is not installed.
"""

from __future__ import annotations

import base64
import io
import random
import time
from abc import ABC, abstractmethod

import numpy as np
from PIL import Image
from loguru import logger

from config import get_settings
from schemas import BoundingBox, Detection

settings = get_settings()


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────────────────

class BaseYoloModel(ABC):
    """Common interface for both the mock and the real YOLOv8 model."""

    @abstractmethod
    def predict(self, image: Image.Image) -> tuple[list[Detection], float]:
        """
        Run inference on a PIL image.

        Returns:
            detections: list of raw Detection objects (is_inside_roi not yet set)
            inference_ms: wall-clock inference time in milliseconds
        """


# ─────────────────────────────────────────────────────────────────────────────
# Mock model — deterministic-ish results, no external deps
# ─────────────────────────────────────────────────────────────────────────────

# Class names from COCO + custom Falcon classes
_COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
    # Custom Falcon Security classes
    "gun", "pistol", "rifle", "weapon", "fight", "fire", "smoke",
]

# Weighted sampling — person is the most common detection in a surveillance feed
_CLASS_WEIGHTS = {
    "person": 40,
    "chair": 15,
    "cup": 10,
    "bottle": 10,
    "laptop": 8,
    "cell phone": 8,
    "knife": 3,
    "gun": 1,
    "fight": 2,
    "fire": 1,
}
_DEFAULT_WEIGHT = 1

_CLASSES_LIST = _COCO_CLASSES
_WEIGHTS_LIST = [_CLASS_WEIGHTS.get(c, _DEFAULT_WEIGHT) for c in _CLASSES_LIST]


class MockYoloModel(BaseYoloModel):
    """
    Mock YOLOv8 model for development and testing.

    Behaviour:
    - Always detects 1-4 objects per frame
    - Detection locations are random within the frame dimensions
    - When the environment variable MOCK_FORCE_ANOMALY=1 is set,
      the first detection is always a weapon/fight class
    - Inference time is simulated as 5–25 ms
    """

    def __init__(self) -> None:
        logger.info("🤖 MockYoloModel initialised (no real inference — development mode)")

    def predict(self, image: Image.Image) -> tuple[list[Detection], float]:
        t0 = time.perf_counter()

        w, h = image.size
        num_detections = random.randint(1, 4)
        detections: list[Detection] = []

        import os
        force_anomaly = os.getenv("MOCK_FORCE_ANOMALY", "0") == "1"

        for i in range(num_detections):
            # Force the first detection to be an alert class when requested
            if i == 0 and force_anomaly:
                class_name = random.choice(["knife", "gun", "fight", "weapon"])
                confidence = round(random.uniform(0.75, 0.98), 4)
            else:
                # Weighted random class selection
                class_name = random.choices(_CLASSES_LIST, weights=_WEIGHTS_LIST, k=1)[0]
                confidence = round(random.uniform(0.45, 0.97), 4)

            # Random bounding box within the frame
            bx = random.randint(0, max(0, w - 50))
            by = random.randint(0, max(0, h - 50))
            bw = random.randint(30, min(200, w - bx))
            bh = random.randint(30, min(200, h - by))

            detections.append(
                Detection(
                    class_name=class_name,
                    confidence=confidence,
                    bounding_box=BoundingBox(x=bx, y=by, w=bw, h=bh),
                )
            )

        # Simulate inference latency
        time.sleep(random.uniform(0.005, 0.025))
        inference_ms = (time.perf_counter() - t0) * 1000

        return detections, round(inference_ms, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Real YOLOv8 model (Ultralytics)
# ─────────────────────────────────────────────────────────────────────────────

class RealYoloModel(BaseYoloModel):
    """
    Wrapper around the Ultralytics YOLOv8 model.

    Install: pip install ultralytics
    Model:   download from https://github.com/ultralytics/assets/releases
             or use a custom fine-tuned .pt file.
    """

    def __init__(self, model_path: str) -> None:
        try:
            from ultralytics import YOLO  # type: ignore
            logger.info(f"🔥 Loading YOLOv8 model from: {model_path}")
            self._model = YOLO(model_path)
            logger.success(f"✅ YOLOv8 model loaded — {model_path}")
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is not installed. "
                "Run `pip install ultralytics` or set MODEL_PATH=mock in .env."
            ) from exc

    def predict(self, image: Image.Image) -> tuple[list[Detection], float]:
        t0 = time.perf_counter()

        results = self._model.predict(
            source=image,
            conf=settings.confidence_threshold,
            iou=settings.iou_threshold,
            verbose=False,
        )

        inference_ms = (time.perf_counter() - t0) * 1000
        detections: list[Detection] = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_idx = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                class_name = result.names.get(cls_idx, str(cls_idx))

                detections.append(
                    Detection(
                        class_name=class_name,
                        confidence=round(conf, 4),
                        bounding_box=BoundingBox(
                            x=int(x1),
                            y=int(y1),
                            w=int(x2 - x1),
                            h=int(y2 - y1),
                        ),
                    )
                )

        return detections, round(inference_ms, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def load_model() -> BaseYoloModel:
    """
    Load the appropriate model based on MODEL_PATH in .env.
    Called once at application startup.
    """
    if settings.model_path.lower() == "mock":
        return MockYoloModel()
    return RealYoloModel(settings.model_path)


# ─────────────────────────────────────────────────────────────────────────────
# Frame decoder
# ─────────────────────────────────────────────────────────────────────────────

def decode_frame(frame_base64: str, max_size: int = 1280) -> Image.Image:
    """
    Decode a base64-encoded image and optionally downscale it.

    Args:
        frame_base64: base64 string (no data URI prefix)
        max_size: if the longest side exceeds this, the image is resized
                  proportionally to keep inference fast on edge hardware

    Returns:
        PIL.Image in RGB mode
    """
    raw_bytes = base64.b64decode(frame_base64)
    image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")

    w, h = image.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), Image.LANCZOS)
        logger.debug(f"📐 Frame resized: {w}×{h} → {new_w}×{new_h}")

    return image


# ─────────────────────────────────────────────────────────────────────────────
# ROI filter
# ─────────────────────────────────────────────────────────────────────────────

def filter_detections_by_roi(
    detections: list[Detection],
    roi: BoundingBox,
    min_overlap: float = 0.30,
) -> list[Detection]:
    """
    Annotate each detection with is_inside_roi and overlap_ratio.
    A detection is considered "inside" if at least `min_overlap` of its area
    overlaps with the table's ROI bounding box.

    Args:
        detections: raw detections from the model
        roi: the table's registered bounding box
        min_overlap: minimum overlap fraction (0–1) to qualify

    Returns:
        The same list with is_inside_roi + overlap_ratio populated.
    """
    for det in detections:
        ratio = roi.overlap_ratio(det.bounding_box)
        det.overlap_ratio = round(ratio, 4)
        det.is_inside_roi = ratio >= min_overlap

    return detections
