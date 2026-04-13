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
from schemas import (
    BoundingBox,
    Detection,
    EmotionType,
    FaceDetection,
    DeskRoi,
    BestViewCandidate,
    BestViewResult,
    IRATE_EMOTIONS,
    IRATE_EMOTION_THRESHOLD,
)

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


# ─────────────────────────────────────────────────────────────────────────────
# Face Emotion Analyzer — Objective 13: Irate Customer
# ─────────────────────────────────────────────────────────────────────────────

# Realistic emotion probability weights for a customer-service environment
# (bank counter / telecom office). Mostly NEUTRAL; small tail of angry/irate.
_EMOTION_WEIGHTS: dict[str, float] = {
    EmotionType.NEUTRAL:    0.44,
    EmotionType.HAPPY:      0.18,
    EmotionType.FRUSTRATED: 0.15,
    EmotionType.CALM:       0.09,
    EmotionType.ANGRY:      0.07,
    EmotionType.SURPRISED:  0.04,
    EmotionType.DISGUSTED:  0.02,
    EmotionType.FEARFUL:    0.01,
}
_EMOTION_KEYS:   list[str]   = list(_EMOTION_WEIGHTS.keys())
_EMOTION_W_LIST: list[float] = list(_EMOTION_WEIGHTS.values())


def _compute_clarity_score(
    face_area_px: int,
    sharpness_score: float,
    detection_confidence: float,
    frame_area_px: int,
) -> float:
    """
    Composite face clarity score (0–1) used by BestViewSelector.

    Weights
    -------
    50%  face area relative to frame — bigger face = better / closer view
    30%  sharpness estimate (normalised Laplacian variance in real mode)
    20%  face detection confidence

    The area component is capped at 25% of the frame to prevent extreme values
    when a face fills most of the camera's FOV.
    """
    area_ratio = min(1.0, face_area_px / max(1, frame_area_px * 0.25))
    score = 0.50 * area_ratio + 0.30 * sharpness_score + 0.20 * detection_confidence
    return round(score, 4)


class BaseFaceAnalyzer(ABC):
    """Common interface for face detection + emotion analysis engines."""

    @abstractmethod
    def analyze(self, image: Image.Image, roi: BoundingBox) -> list[FaceDetection]:
        """
        Detect faces within `roi` and return per-face emotion analysis.

        Args:
            image: Full camera frame (PIL RGB).
            roi:   Desk bounding box — only faces overlapping this ROI matter.

        Returns:
            List of FaceDetection objects annotated with clarity metrics and
            ROI membership flags.
        """


class MockFaceAnalyzer(BaseFaceAnalyzer):
    """
    Mock face analyzer for development and CI.

    Behaviour
    ---------
    - Simulates 0–2 faces per ROI (weighted: 30% zero, 60% one, 10% two).
    - Emotion scores are sampled with realistic customer-service weights.
    - When ``MOCK_FORCE_ANOMALY=1`` is set, the first face in every ROI is
      guaranteed ANGRY with irate_confidence ≥ 0.68 — always triggers an alert.
    - Clarity score is computed from simulated face size and random sharpness.
    """

    def __init__(self) -> None:
        logger.info("😐 MockFaceAnalyzer initialised (development mode — no real inference)")

    def analyze(self, image: Image.Image, roi: BoundingBox) -> list[FaceDetection]:
        import os
        force_anomaly = os.getenv("MOCK_FORCE_ANOMALY", "0") == "1"

        frame_w, frame_h = image.size
        frame_area = frame_w * frame_h

        num_faces = random.choices([0, 1, 2], weights=[30, 60, 10], k=1)[0]
        if force_anomaly:
            num_faces = max(1, num_faces)

        detections: list[FaceDetection] = []
        for i in range(num_faces):
            # Place face box randomly within the desk ROI (min 40×50 px)
            fx = random.randint(roi.x, max(roi.x, roi.x2 - 80))
            fy = random.randint(roi.y, max(roi.y, roi.y2 - 80))
            fw = random.randint(40, min(120, max(41, roi.x2 - fx)))
            fh = random.randint(50, min(140, max(51, roi.y2 - fy)))
            face_box  = BoundingBox(x=fx, y=fy, w=fw, h=fh)
            face_area = fw * fh

            det_conf  = round(random.uniform(0.72, 0.99), 4)
            sharpness = round(random.uniform(0.30, 0.95), 4)

            if i == 0 and force_anomaly:
                dominant = EmotionType.ANGRY
                scores = {e: round(random.uniform(0.02, 0.15), 3) for e in _EMOTION_KEYS}
                scores[EmotionType.ANGRY]      = round(random.uniform(0.68, 0.95), 3)
                scores[EmotionType.FRUSTRATED] = round(random.uniform(0.45, 0.70), 3)
            else:
                dom_str  = random.choices(_EMOTION_KEYS, weights=_EMOTION_W_LIST, k=1)[0]
                dominant = EmotionType(dom_str)
                scores   = {e: round(random.uniform(0.01, 0.20), 3) for e in _EMOTION_KEYS}
                scores[dom_str] = round(random.uniform(0.45, 0.90), 3)

            clarity = _compute_clarity_score(face_area, sharpness, det_conf, frame_area)
            overlap = roi.overlap_ratio(face_box)

            detections.append(
                FaceDetection(
                    face_id=i,
                    bounding_box=face_box,
                    detection_confidence=det_conf,
                    emotion_scores=scores,
                    dominant_emotion=dominant,
                    face_area_px=face_area,
                    sharpness_score=sharpness,
                    clarity_score=clarity,
                    is_inside_roi=overlap >= 0.30,
                    roi_overlap=round(overlap, 4),
                )
            )

        return detections


class RealFaceAnalyzer(BaseFaceAnalyzer):
    """
    Production face analyzer using DeepFace for emotion recognition.

    Install
    -------
        pip install deepface tf-keras opencv-python

    Notes
    -----
    - Crops the desk ROI before running DeepFace to speed up detection and
      reduce false positives from outside the desk zone.
    - Region coordinates from DeepFace are relative to the ROI crop; they are
      translated back to full-frame coordinates here.
    - Emotion scores from DeepFace are in 0–100 range; normalised to 0–1.
    - Sharpness: normalised Laplacian variance of the cropped face region.
    - DeepFace "sad" is mapped to FRUSTRATED; "disgust" → DISGUSTED.
    """

    _EMOTION_MAP: dict[str, EmotionType] = {
        "angry":    EmotionType.ANGRY,
        "disgust":  EmotionType.DISGUSTED,
        "fear":     EmotionType.FEARFUL,
        "happy":    EmotionType.HAPPY,
        "sad":      EmotionType.FRUSTRATED,   # "sad" ≈ FRUSTRATED in this context
        "surprise": EmotionType.SURPRISED,
        "neutral":  EmotionType.NEUTRAL,
    }

    def __init__(self) -> None:
        try:
            from deepface import DeepFace as _DF  # type: ignore
            import cv2 as _cv2  # type: ignore  # noqa: F401
            self._DeepFace = _DF
            logger.success("✅ RealFaceAnalyzer ready (DeepFace + OpenCV)")
        except ImportError as exc:
            raise RuntimeError(
                "deepface or opencv-python not installed. "
                "Run: pip install deepface tf-keras opencv-python  "
                "or set USE_REAL_FACE_MODEL=false in .env to use mock mode."
            ) from exc

    def analyze(self, image: Image.Image, roi: BoundingBox) -> list[FaceDetection]:
        import cv2
        import numpy as np

        roi_crop   = image.crop((roi.x, roi.y, roi.x2, roi.y2))
        frame_area = image.width * image.height

        try:
            results = self._DeepFace.analyze(
                img_path=np.array(roi_crop),
                actions=["emotion"],
                enforce_detection=False,
                silent=True,
            )
        except Exception as exc:
            logger.warning(f"DeepFace analysis failed for roi={roi}: {exc}")
            return []

        if not isinstance(results, list):
            results = [results]

        detections: list[FaceDetection] = []
        for i, result in enumerate(results):
            reg = result.get("region", {})
            # Translate ROI-relative coords back to full-frame coords
            fx = roi.x + int(reg.get("x", 0))
            fy = roi.y + int(reg.get("y", 0))
            fw = max(1, int(reg.get("w", 60)))
            fh = max(1, int(reg.get("h", 80)))
            face_box  = BoundingBox(x=fx, y=fy, w=fw, h=fh)
            face_area = fw * fh

            raw_emotions: dict[str, float] = result.get("emotion", {})
            scores: dict[str, float] = {}
            for raw_key, raw_score in raw_emotions.items():
                mapped = self._EMOTION_MAP.get(raw_key.lower())
                if mapped:
                    scores[mapped.value] = round(float(raw_score) / 100.0, 4)

            dominant_raw = result.get("dominant_emotion", "neutral").lower()
            dominant     = self._EMOTION_MAP.get(dominant_raw, EmotionType.NEUTRAL)

            # Sharpness via Laplacian variance of the isolated face crop
            face_crop = image.crop((fx, fy, fx + fw, fy + fh))
            gray      = np.array(face_crop.convert("L"))
            lap_var   = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            sharpness = round(min(1.0, lap_var / 500.0), 4)

            det_conf = round(float(result.get("face_confidence", 0.80)), 4)
            clarity  = _compute_clarity_score(face_area, sharpness, det_conf, frame_area)
            overlap  = roi.overlap_ratio(face_box)

            detections.append(
                FaceDetection(
                    face_id=i,
                    bounding_box=face_box,
                    detection_confidence=det_conf,
                    emotion_scores=scores,
                    dominant_emotion=dominant,
                    face_area_px=face_area,
                    sharpness_score=sharpness,
                    clarity_score=clarity,
                    is_inside_roi=overlap >= 0.30,
                    roi_overlap=round(overlap, 4),
                )
            )

        return detections


def load_face_analyzer() -> BaseFaceAnalyzer:
    """
    Instantiate the appropriate face analyzer from the USE_REAL_FACE_MODEL env var.
    Called once at application startup alongside load_model().

    USE_REAL_FACE_MODEL=true   → RealFaceAnalyzer (requires deepface + opencv)
    USE_REAL_FACE_MODEL=false  → MockFaceAnalyzer (default; no extra deps)
    """
    import os
    if os.getenv("USE_REAL_FACE_MODEL", "false").lower() == "true":
        return RealFaceAnalyzer()
    return MockFaceAnalyzer()


# ─────────────────────────────────────────────────────────────────────────────
# Best View Selector — picks the clearest face across cameras for a desk
# ─────────────────────────────────────────────────────────────────────────────

class BestViewSelector:
    """
    Selects the highest-quality face view for a desk from multiple cameras.

    Algorithm
    ---------
    1. For each camera, find the face with the highest ``clarity_score``
       that is inside the desk ROI (``is_inside_roi=True``).
    2. Compare per-camera winners: the camera with the highest winner
       clarity_score is the **Best View**.
    3. Compute ``irate_confidence`` = max(ANGRY, FRUSTRATED, DISGUSTED)
       from the winning face's emotion_scores.
    4. Set ``is_irate = irate_confidence >= IRATE_EMOTION_THRESHOLD``.

    If no camera detected a face in the ROI, ``no_face_detected=True`` is
    returned and no alert is generated.
    """

    def select(
        self,
        desk_roi: DeskRoi,
        candidates: list[tuple[str, list[FaceDetection]]],
    ) -> BestViewResult:
        """
        Args:
            desk_roi:   The DeskRoi being evaluated (carries desk_id, desk_label,
                        table_id, and bounding_box).
            candidates: List of (camera_id, face_detections_for_this_desk).
                        Faces must already be annotated with is_inside_roi by
                        BaseFaceAnalyzer.analyze().

        Returns:
            BestViewResult with winner camera, expression analysis, and irate flag.
        """
        best_per_cam:  list[BestViewCandidate] = []
        best_face:     FaceDetection | None     = None
        best_camera_id: str                     = ""
        best_clarity:  float                    = -1.0

        for camera_id, faces in candidates:
            roi_faces = [f for f in faces if f.is_inside_roi]
            if not roi_faces:
                best_per_cam.append(
                    BestViewCandidate(
                        camera_id=camera_id,
                        desk_id=desk_roi.desk_id,
                        face=None,
                        clarity_score=0.0,
                    )
                )
                continue

            # Per-camera winner: highest clarity among ROI-inside faces
            cam_best = max(roi_faces, key=lambda f: f.clarity_score)
            best_per_cam.append(
                BestViewCandidate(
                    camera_id=camera_id,
                    desk_id=desk_roi.desk_id,
                    face=cam_best,
                    clarity_score=cam_best.clarity_score,
                )
            )
            if cam_best.clarity_score > best_clarity:
                best_clarity   = cam_best.clarity_score
                best_face      = cam_best
                best_camera_id = camera_id

        if best_face is None:
            return BestViewResult(
                desk_id=desk_roi.desk_id,
                desk_label=desk_roi.desk_label,
                table_id=desk_roi.table_id,
                winner_camera_id=best_camera_id or (candidates[0][0] if candidates else ""),
                winner_clarity_score=0.0,
                no_face_detected=True,
                dominant_emotion=EmotionType.NEUTRAL,
                emotion_scores={},
                irate_confidence=0.0,
                is_irate=False,
                candidates=best_per_cam,
            )

        emotion_scores = best_face.emotion_scores
        irate_conf = max(
            (emotion_scores.get(e, 0.0) for e in IRATE_EMOTIONS),
            default=0.0,
        )

        logger.debug(
            f"🎭 BestView  desk={desk_roi.desk_id}  "
            f"winner={best_camera_id}  clarity={best_clarity:.3f}  "
            f"emotion={best_face.dominant_emotion.value}  "
            f"irate_conf={irate_conf:.3f}"
        )

        return BestViewResult(
            desk_id=desk_roi.desk_id,
            desk_label=desk_roi.desk_label,
            table_id=desk_roi.table_id,
            winner_camera_id=best_camera_id,
            winner_clarity_score=round(best_clarity, 4),
            no_face_detected=False,
            dominant_emotion=best_face.dominant_emotion,
            emotion_scores=emotion_scores,
            irate_confidence=round(irate_conf, 4),
            is_irate=irate_conf >= IRATE_EMOTION_THRESHOLD,
            candidates=best_per_cam,
        )
