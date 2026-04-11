"""
schemas.py — Pydantic models for request / response bodies.
"""

from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────────────────────────────────────
# Shared primitives
# ─────────────────────────────────────────────────────────────────────────────

class BoundingBox(BaseModel):
    """
    Pixel coordinates of a region of interest inside a camera frame.
    Mirrors the BoundingBoxDto used in the NestJS Table model.
    """
    x: int = Field(..., ge=0, description="Top-left X coordinate (px)")
    y: int = Field(..., ge=0, description="Top-left Y coordinate (px)")
    w: int = Field(..., ge=1, description="Width (px)")
    h: int = Field(..., ge=1, description="Height (px)")

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    def contains_point(self, px: float, py: float) -> bool:
        """Returns True if point (px, py) lies inside this box."""
        return self.x <= px <= self.x2 and self.y <= py <= self.y2

    def iou(self, other: "BoundingBox") -> float:
        """Intersection-over-Union with another bounding box."""
        ix1 = max(self.x, other.x)
        iy1 = max(self.y, other.y)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)

        inter_w = max(0, ix2 - ix1)
        inter_h = max(0, iy2 - iy1)
        inter_area = inter_w * inter_h

        area_self = self.w * self.h
        area_other = other.w * other.h
        union_area = area_self + area_other - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def overlap_ratio(self, other: "BoundingBox") -> float:
        """
        Fraction of `other` box that overlaps with `self`.
        Used to check how much of a detected person falls inside the table ROI.
        """
        ix1 = max(self.x, other.x)
        iy1 = max(self.y, other.y)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)

        inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        other_area = other.w * other.h
        return inter_area / other_area if other_area > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# /analyze-frame  — request
# ─────────────────────────────────────────────────────────────────────────────

class AnalyzeFrameRequest(BaseModel):
    """
    Payload accepted by POST /analyze-frame.

    The image is base64-encoded so the endpoint works over plain JSON without
    multipart form-data.  For high-throughput production use, switch the
    endpoint to accept multipart/form-data (see the comment in main.py).
    """
    # ── Context ───────────────────────────────────────────────────────────────
    center_id: str = Field(..., description="CUID of the center this frame belongs to")
    camera_id: str = Field(..., description="CUID of the camera that captured the frame")
    table_id: str | None = Field(None, description="CUID of the table being monitored")

    # ── Image ─────────────────────────────────────────────────────────────────
    frame_base64: str = Field(
        ...,
        description=(
            "Base64-encoded JPEG or PNG frame.  "
            "Strip the data URI prefix (data:image/jpeg;base64,) before sending."
        ),
    )

    # ── Region of interest ────────────────────────────────────────────────────
    bounding_box: BoundingBox = Field(
        ...,
        description=(
            "The table's registered bounding box on this camera's frame.  "
            "Only detections that overlap this region trigger alerts."
        ),
    )

    # ── Optional metadata ─────────────────────────────────────────────────────
    timestamp: float | None = Field(
        None,
        description="Unix timestamp (seconds) when the frame was captured.  "
                    "Defaults to server time if omitted.",
    )

    @field_validator("frame_base64")
    @classmethod
    def strip_data_uri(cls, v: str) -> str:
        """Silently strip the data URI prefix if the client included it."""
        if "," in v and v.startswith("data:"):
            return v.split(",", 1)[1]
        return v


# ─────────────────────────────────────────────────────────────────────────────
# Detection result types
# ─────────────────────────────────────────────────────────────────────────────

class AlertSeverity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class AnomalyType(str, Enum):
    WEAPON_DETECTED = "WEAPON_DETECTED"
    FIGHT_DETECTED = "FIGHT_DETECTED"
    FALL_DETECTED = "FALL_DETECTED"
    FIRE_DETECTED = "FIRE_DETECTED"
    UNATTENDED_OBJECT = "UNATTENDED_OBJECT"
    UNKNOWN = "UNKNOWN"


class Detection(BaseModel):
    """A single object detected by YOLOv8 inside the frame."""
    class_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bounding_box: BoundingBox
    is_inside_roi: bool = Field(
        False,
        description="True if this detection overlaps the table's registered bounding box",
    )
    overlap_ratio: float = Field(
        0.0,
        description="Fraction of the detected object that falls inside the ROI (0–1)",
    )


class AnalyzeFrameResponse(BaseModel):
    """Response returned by POST /analyze-frame."""
    # ── Input echo ────────────────────────────────────────────────────────────
    center_id: str
    camera_id: str
    table_id: str | None

    # ── Result ────────────────────────────────────────────────────────────────
    person_detected_in_roi: bool = Field(
        description="True if at least one person was detected inside the table's bounding box"
    )
    anomaly_detected: bool
    anomaly_type: AnomalyType | None = None
    severity: AlertSeverity = AlertSeverity.INFO

    # ── All detections (full frame, not just ROI) ─────────────────────────────
    detections: list[Detection] = Field(default_factory=list)

    # ── Timing ────────────────────────────────────────────────────────────────
    inference_ms: float = Field(description="Model inference time in milliseconds")
    server_time: str = Field(description="ISO-8601 timestamp when this response was generated")

    # ── Alert dispatch ────────────────────────────────────────────────────────
    alert_dispatched: bool = Field(
        False,
        description="True if an alert was POSTed to the NestJS backend",
    )
    alert_dispatch_status: int | None = Field(
        None,
        description="HTTP status code returned by the NestJS /ingest/ai-alert endpoint",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Alert payload (sent to NestJS /ingest/ai-alert)
# ─────────────────────────────────────────────────────────────────────────────

class AlertPayload(BaseModel):
    """
    JSON body POSTed to the NestJS backend when an anomaly is detected.
    NestJS will publish this on MQTT topic falcon/center/{centerId}/ai-results
    so it flows through the normal real-time pipeline to the frontend.
    """
    center_id: str
    camera_id: str
    table_id: str | None
    anomaly_type: str
    severity: str
    primary_event: str
    detections: list[dict[str, Any]]
    timestamp: float
    source: str = "ai-service"


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic AI Configuration — per-camera feature toggles
# ─────────────────────────────────────────────────────────────────────────────

class AiFeature(str, Enum):
    """AI inference features that can be individually enabled/disabled per camera."""
    WEAPON = "WEAPON"  # knife, gun, pistol, rifle, weapon
    FIGHT  = "FIGHT"   # physical altercation / aggression
    FALL   = "FALL"    # person-fall detection
    FIRE   = "FIRE"    # fire and smoke
    CROWD  = "CROWD"   # crowd density / overcrowding


# Mapping: YOLO class name → AiFeature gate
CLASS_TO_FEATURE: dict[str, AiFeature] = {
    "knife":        AiFeature.WEAPON,
    "gun":          AiFeature.WEAPON,
    "pistol":       AiFeature.WEAPON,
    "rifle":        AiFeature.WEAPON,
    "weapon":       AiFeature.WEAPON,
    "fight":        AiFeature.FIGHT,
    "fall":         AiFeature.FALL,
    "fire":         AiFeature.FIRE,
    "smoke":        AiFeature.FIRE,
    "crowd":        AiFeature.CROWD,
    "overcrowding": AiFeature.CROWD,
}


class CameraConfigUpdate(BaseModel):
    """Payload accepted by POST /camera-config."""
    camera_id: str = Field(..., description="CUID of the camera to configure")
    enabled_features: list[AiFeature] = Field(
        default_factory=list,
        description="List of enabled AI features. Empty list → all features disabled.",
    )


class CameraConfigResponse(BaseModel):
    """Response from POST /camera-config."""
    camera_id: str
    enabled_features: list[AiFeature]
    message: str
