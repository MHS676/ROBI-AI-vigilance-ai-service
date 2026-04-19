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
    WEAPON_DETECTED   = "WEAPON_DETECTED"
    FIGHT_DETECTED    = "FIGHT_DETECTED"
    FALL_DETECTED     = "FALL_DETECTED"
    FIRE_DETECTED     = "FIRE_DETECTED"
    UNATTENDED_OBJECT = "UNATTENDED_OBJECT"
    IRATE_CUSTOMER    = "IRATE_CUSTOMER"   # Objective 13 — cross-camera face emotion
    IDLE_AGENT        = "IDLE_AGENT"       # Assigned agent absent > threshold
    GOSSIP_DETECTED   = "GOSSIP_DETECTED"  # Two+ agents chatting without customer
    UNKNOWN           = "UNKNOWN"


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


# ─────────────────────────────────────────────────────────────────────────────
# Cross-Camera Face Emotion Analysis — Objective 13: Irate Customer
# ─────────────────────────────────────────────────────────────────────────────

class EmotionType(str, Enum):
    """Emotion categories used for customer expression analysis."""
    ANGRY      = "ANGRY"       # primary irate indicator
    FRUSTRATED = "FRUSTRATED"  # secondary irate indicator
    NEUTRAL    = "NEUTRAL"     # baseline expression
    HAPPY      = "HAPPY"
    CALM       = "CALM"
    SURPRISED  = "SURPRISED"
    DISGUSTED  = "DISGUSTED"   # tertiary irate indicator
    FEARFUL    = "FEARFUL"


#: Emotion values that contribute to the IRATE_CUSTOMER classification
IRATE_EMOTIONS: frozenset[str] = frozenset({
    EmotionType.ANGRY,
    EmotionType.FRUSTRATED,
    EmotionType.DISGUSTED,
})

#: Minimum irate-emotion score (0–1) required to fire an IRATE_CUSTOMER alert
IRATE_EMOTION_THRESHOLD: float = 0.65


class FaceDetection(BaseModel):
    """
    A single face detected inside a camera frame, with emotion analysis.
    Produced by BaseFaceAnalyzer.analyze() and consumed by BestViewSelector.
    """
    face_id: int = Field(..., description="Index of this face within the frame (0-based)")
    bounding_box: BoundingBox
    detection_confidence: float = Field(..., ge=0.0, le=1.0)

    # Per-emotion probability scores (keys match EmotionType values)
    emotion_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Per-emotion probability scores (0–1). Keys are EmotionType enum values.",
    )
    dominant_emotion: EmotionType = Field(
        default=EmotionType.NEUTRAL,
        description="Highest-scoring emotion for this face",
    )

    # ── Clarity metrics — drive the Best View Selector ────────────────────────
    face_area_px: int = Field(
        ...,
        description="Face bounding-box area in pixels (width × height)",
    )
    sharpness_score: float = Field(
        ..., ge=0.0, le=1.0,
        description=(
            "Image sharpness estimate (0–1). "
            "Real mode: normalised Laplacian variance. Mock: simulated."
        ),
    )
    clarity_score: float = Field(
        ..., ge=0.0, le=1.0,
        description=(
            "Composite clarity score (0–1) combining face_area, sharpness, and "
            "detection_confidence. Higher = cleaner view. Used by BestViewSelector."
        ),
    )
    is_inside_roi: bool   = Field(default=False)
    roi_overlap: float    = Field(default=0.0, ge=0.0, le=1.0)


class DeskRoi(BaseModel):
    """Region of Interest covering one service desk on a given camera’s frame."""
    desk_id:     str            = Field(..., description="Stable desk ID, e.g. 'desk_1'")
    desk_label:  str            = Field(..., description="Human-readable name, e.g. 'Desk 1 — Teller A'")
    table_id:    str | None     = Field(None, description="CUID of the linked DB Table (once provisioned)")
    bounding_box: BoundingBox


class CameraRoiLayout(BaseModel):
    """All desk ROIs registered for one physical camera."""
    camera_id: str
    desk_rois: list[DeskRoi] = Field(default_factory=list)

    def get_roi_for_desk(self, desk_id: str) -> "DeskRoi | None":
        """Return the DeskRoi for the given desk_id, or None."""
        for roi in self.desk_rois:
            if roi.desk_id == desk_id:
                return roi
        return None

    @property
    def desk_ids(self) -> list[str]:
        return [r.desk_id for r in self.desk_rois]


class CrossCameraLayout(BaseModel):
    """
    A named multi-camera layout for a branch.

    Maps two or more cameras to the same set of service desks viewed from
    different physical angles. The primary_camera_id is the frontal / highest-
    resolution camera; secondary cameras provide face-angle coverage.
    """
    layout_id:         str                = Field(..., description="Unique ID, e.g. 'layout_cam1_cam2'")
    description:       str                = Field(default="")
    primary_camera_id: str                = Field(..., description="Frontal / highest-resolution camera ID")
    cameras:           list[CameraRoiLayout] = Field(
        ..., description="Per-camera desk ROI mappings. At least one camera required."
    )

    @property
    def all_desk_ids(self) -> list[str]:
        """Sorted deduplicated list of all desk IDs across all cameras."""
        ids: set[str] = set()
        for cam in self.cameras:
            ids.update(cam.desk_ids)
        return sorted(ids)

    def get_camera_layout(self, camera_id: str) -> "CameraRoiLayout | None":
        for cam in self.cameras:
            if cam.camera_id == camera_id:
                return cam
        return None


class CrossCameraFrameInput(BaseModel):
    """One camera’s frame contribution to a cross-camera analysis request."""
    camera_id:    str
    frame_base64: str
    timestamp:    float | None = None

    @field_validator("frame_base64")
    @classmethod
    def strip_data_uri(cls, v: str) -> str:
        """Silently strip the data URI prefix if the client included it."""
        if "," in v and v.startswith("data:"):
            return v.split(",", 1)[1]
        return v


class CrossCameraAnalyzeRequest(BaseModel):
    """POST /analyze-cross-camera — frames from all cameras in a named layout."""
    center_id:  str
    layout_id:  str = Field(
        ...,
        description=(
            "ID of a registered CrossCameraLayout. "
            "Use 'layout_cam1_cam2' for the built-in 3-desk layout."
        ),
    )
    frames: list[CrossCameraFrameInput] = Field(
        ..., min_length=1,
        description="Frames from one or more cameras listed in the layout.",
    )
    timestamp: float | None = None


class BestViewCandidate(BaseModel):
    """One camera’s best face candidate for a given desk."""
    camera_id:     str
    desk_id:       str
    face:          FaceDetection | None = Field(
        None,
        description="Best face found in this camera’s desk ROI (None = no face detected)",
    )
    clarity_score: float = Field(
        default=0.0,
        description="0.0 when no face detected; otherwise the best face’s clarity_score",
    )


class BestViewResult(BaseModel):
    """
    Cross-camera analysis result for one desk — Objective 13: Irate Customer.

    The Best View Selector picks the camera with the highest-clarity face.
    Emotion analysis (and the irate flag) come from that winning camera’s face.
    """
    desk_id:   str
    desk_label: str
    table_id:  str | None

    # ── Best View Selection ─────────────────────────────────────────────────
    winner_camera_id:    str   = Field(..., description="Camera that provided the best face view")
    winner_clarity_score: float = Field(..., ge=0.0, le=1.0)
    no_face_detected:    bool  = Field(default=False)

    # ── Expression analysis (from the winning camera) ──────────────────────
    dominant_emotion: EmotionType         = Field(default=EmotionType.NEUTRAL)
    emotion_scores:   dict[str, float]    = Field(default_factory=dict)
    irate_confidence: float               = Field(
        default=0.0, ge=0.0, le=1.0,
        description=(
            "max(ANGRY, FRUSTRATED, DISGUSTED) from the winning face. "
            "Triggers IRATE_CUSTOMER alert when ≥ IRATE_EMOTION_THRESHOLD (0.65)."
        ),
    )
    is_irate: bool = Field(
        default=False,
        description="True if irate_confidence ≥ IRATE_EMOTION_THRESHOLD",
    )

    # ── All candidates (audit trail / frontend debug panel) ─────────────────
    candidates: list[BestViewCandidate] = Field(default_factory=list)


class CrossCameraAnalyzeResponse(BaseModel):
    """Response from POST /analyze-cross-camera."""
    center_id:               str
    layout_id:               str
    desk_results:            list[BestViewResult]
    total_irate_detected:    int   = Field(default=0, description="Desks with irate customers")
    irate_alerts_dispatched: int   = Field(default=0, description="Alerts POSTed to NestJS")
    inference_ms:            float = Field(description="Total wall-clock inference time (ms)")
    server_time:             str


# ─────────────────────────────────────────────────────────────────────────────
# Agent Monitoring — Schemas for /register-table and /monitor-agent
# ─────────────────────────────────────────────────────────────────────────────

class TableRegistrationRequest(BaseModel):
    """
    POST /register-table

    Called by NestJS on startup (for every provisioned table) and whenever
    a Super Admin reassigns an agent.
    """
    table_id:       str           = Field(..., description="Prisma CUID of the Table")
    agent_id:       str           = Field(..., description="Prisma CUID of the assigned User")
    center_id:      str           = Field(..., description="Prisma CUID of the Center")
    camera_id:      str           = Field(..., description="Prisma CUID of the Camera")
    face_photo_path: str | None   = Field(
        None,
        description=(
            "Absolute path to the agent's reference face JPEG on the local server, "
            "e.g. /data/agent-faces/clxyz123.jpg.  "
            "Matches User.facePhotoPath in the Prisma schema."
        ),
    )


class TableRegistrationResponse(BaseModel):
    """Response from POST /register-table."""
    table_id:        str
    agent_id:        str
    embedding_loaded: bool   = Field(description="True if the reference face embedding was loaded successfully")
    message:         str


class AgentMonitorFrameRequest(BaseModel):
    """
    POST /monitor-agent

    Sent by the NestJS camera-poller for each table on every polling cycle.
    Contains the current frame + pre-computed YOLO detection count + table context.
    """
    center_id:      str        = Field(..., description="CUID of the center")
    camera_id:      str        = Field(..., description="CUID of the camera")
    table_id:       str        = Field(..., description="CUID of the table")
    agent_id:       str        = Field(..., description="CUID of the assigned agent")

    frame_base64:   str        = Field(
        ...,
        description="Base64-encoded JPEG frame from the table's camera (no data URI prefix).",
    )
    bounding_box:   BoundingBox = Field(..., description="Table ROI on the camera frame")

    # YOLO-derived person count (already computed upstream by /analyze-frame)
    persons_in_roi: int        = Field(0, ge=0, description="Number of persons detected in the table ROI by YOLO")

    timestamp: float | None    = Field(None, description="Unix epoch when the frame was captured")

    @field_validator("frame_base64")
    @classmethod
    def strip_data_uri(cls, v: str) -> str:
        if "," in v and v.startswith("data:"):
            return v.split(",", 1)[1]
        return v


class SHISampleOut(BaseModel):
    """One timestamped SHI reading in the API response."""
    ts:      float = Field(description="Monotonic timestamp of this sample")
    shi:     float = Field(description="Service Happiness Index [0–100]")
    emotion: str   = Field(description="Dominant emotion at sample time")


class AgentMonitorReport(BaseModel):
    """
    Response from POST /monitor-agent.

    Summarises all three monitoring results for the current frame.
    """
    table_id:   str
    agent_id:   str
    server_time: str

    # ── Presence ────────────────────────────────────────────────────────────
    agent_present:          bool  = False
    idle_seconds:           float = 0.0
    idle_alert_triggered:   bool  = False

    # ── Gossip ───────────────────────────────────────────────────────────
    gossip_active:           bool  = False
    gossip_duration_seconds: float = 0.0
    gossip_alert_triggered:  bool  = False

    # ── Service Happiness Index ──────────────────────────────────────────────
    shi_sampled:    bool              = False
    latest_shi:     float             = 50.0
    rolling_shi:    float             = 50.0
    latest_emotion: str               = "NEUTRAL"
    shi_history:    list[SHISampleOut] = Field(default_factory=list)

    # ── Activity ───────────────────────────────────────────────────────────
    active_minutes:       float = 0.0
    agent_faces_in_roi:   int   = 0
    unknown_faces_in_roi: int   = 0
