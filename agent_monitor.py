"""
agent_monitor.py
================
Falcon Security — Agent Monitoring Engine
-----------------------------------------
Three-in-one per-table monitoring pipeline:

  1. AGENT PRESENCE VERIFICATION
     On each frame, detected faces inside the table ROI are matched against
     the assigned agent's reference ArcFace embedding.
     • Agent absent > IDLE_THRESHOLD_MINUTES during work hours
       → IDLE_AGENT alert dispatched to NestJS

  2. GOSSIP DETECTION
     Counts how many known-agent faces appear in the same table's ROI.
     • ≥2 agent faces with no unmatched customer face for >
       GOSSIP_THRESHOLD_SECONDS → GOSSIP_DETECTED alert

  3. SERVICE HAPPINESS INDEX (SHI)
     Samples the agent's dominant facial expression every
     SHI_SAMPLE_INTERVAL_SECONDS (default 30 s).
     Emotion scores → weighted raw score [-1, +1] → normalised SHI [0, 100].
     Rolling average over SHI_WINDOW_SIZE samples is PATCHed to NestJS
     /ingest/agent-activity (updates the AgentActivity DB record).

Identity back-end
-----------------
Uses DeepFace ArcFace + RetinaFace via detect_and_embed().
Falls back to a random-vector mock when DeepFace is not installed
(development mode — USE_REAL_FACE_MODEL=false).

Integrates with the existing FaceDetection pipeline:
  • The existing BaseFaceAnalyzer already extracts emotion scores;
    those FaceDetection objects are passed directly to process_frame().
  • Identity matching runs on the same detected bounding boxes but calls
    DeepFace.represent() with detector_backend="skip" (crop already known).
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, time as dtime
from typing import TYPE_CHECKING, Optional

import numpy as np
from loguru import logger

from config import get_settings

if TYPE_CHECKING:
    from schemas import FaceDetection

settings = get_settings()


# ─────────────────────────────────────────────────────────────────────────────
# SHI emotion → score weights
# Range [-1, +1]; normalised to [0, 100] for the public API
# ─────────────────────────────────────────────────────────────────────────────

_SHI_WEIGHTS: dict[str, float] = {
    "HAPPY":       1.00,   # radiant service; customers feel welcomed
    "CALM":        0.65,   # composed, professional
    "NEUTRAL":     0.35,   # acceptable baseline
    "SURPRISED":   0.10,   # ambiguous — slight positive lean
    "FEARFUL":    -0.25,   # stress indicator
    "FRUSTRATED": -0.55,   # likely poor service interaction
    "DISGUSTED":  -0.75,   # very negative signal
    "ANGRY":      -1.00,   # worst-case; escalation risk
}


def emotion_scores_to_shi(emotion_scores: dict[str, float]) -> tuple[float, float]:
    """
    Convert a per-emotion probability dict → (raw_score, shi_index).

    raw_score : weighted average of SHI weights, range [-1, +1]
    shi_index : normalised to [0, 100] — the public SHI value
    """
    total_w = 0.0
    total_s = 0.0
    for emotion, prob in emotion_scores.items():
        weight = _SHI_WEIGHTS.get(emotion.upper(), 0.0)
        total_s += weight * float(prob)
        total_w += float(prob)

    raw = total_s / total_w if total_w > 0.001 else 0.0
    shi = round((raw + 1.0) / 2.0 * 100.0, 1)
    return round(raw, 4), shi


# ─────────────────────────────────────────────────────────────────────────────
# Work hours gate
# ─────────────────────────────────────────────────────────────────────────────

def _is_work_hours() -> bool:
    """Return True if the current local time falls within configured work hours."""
    now = datetime.now().time()
    try:
        sh, sm = map(int, settings.work_hours_start.split(":"))
        eh, em = map(int, settings.work_hours_end.split(":"))
        return dtime(sh, sm) <= now <= dtime(eh, em)
    except Exception:
        return True  # fail-open — treat as work hours if config is broken


# ─────────────────────────────────────────────────────────────────────────────
# Cosine distance helper
# ─────────────────────────────────────────────────────────────────────────────

def _cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 2.0
    return float(1.0 - np.dot(a, b) / (na * nb))


# ─────────────────────────────────────────────────────────────────────────────
# DeepFace face-embedding helpers
# Falls back to random mock vectors when DeepFace is unavailable
# ─────────────────────────────────────────────────────────────────────────────

_deepface_available = False
try:
    from deepface import DeepFace as _DeepFace  # type: ignore
    _deepface_available = True
    logger.info("🧠 AgentMonitor — DeepFace available (ArcFace identity mode)")
except ImportError:
    logger.warning(
        "⚠️  AgentMonitor — DeepFace not installed; identity verification in MOCK mode. "
        "Install with: pip install deepface retina-face"
    )


def load_reference_embedding(image_path: str) -> Optional[np.ndarray]:
    """
    Compute an ArcFace embedding from an agent's reference photo.
    Called once when a table is registered.

    Returns:
        512-d float32 numpy array, or None on failure.
    """
    if not _deepface_available:
        logger.debug(f"[mock] load_reference_embedding({image_path}) → random 512-d vector")
        return np.random.default_rng(abs(hash(image_path)) % (2**32)).standard_normal(512).astype(np.float32)

    try:
        result = _DeepFace.represent(
            img_path=image_path,
            model_name="ArcFace",
            detector_backend="retinaface",
            enforce_detection=True,
            align=True,
        )
        emb = np.array(result[0]["embedding"], dtype=np.float32)
        logger.debug(f"✓ Embedding loaded for {image_path} (dim={len(emb)})")
        return emb
    except Exception as exc:
        logger.warning(f"Could not embed reference photo '{image_path}': {exc}")
        return None


def embed_face_crop(
    frame_bgr: np.ndarray,
    x: int, y: int, w: int, h: int,
) -> Optional[np.ndarray]:
    """
    Extract an ArcFace embedding from a pre-located face crop within a BGR frame.
    Uses detector_backend='skip' since the bounding box is already known
    (avoids double-detecting which would add ~50-150 ms per face).

    Returns:
        512-d float32 array, or None on failure.
    """
    if not _deepface_available:
        return np.random.standard_normal(512).astype(np.float32)  # mock

    try:
        crop = frame_bgr[y : y + h, x : x + w]
        if crop.size == 0:
            return None
        result = _DeepFace.represent(
            img_path=crop,
            model_name="ArcFace",
            detector_backend="skip",   # face already located
            enforce_detection=False,
            align=False,
        )
        if result:
            return np.array(result[0]["embedding"], dtype=np.float32)
        return None
    except Exception as exc:
        logger.debug(f"embed_face_crop failed: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Internal state dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SHISample:
    """One timestamped Service Happiness Index reading."""
    ts:      float   # monotonic time
    raw:     float   # weighted emotion score in [-1, +1]
    shi:     float   # normalised index in [0, 100]
    emotion: str     # dominant emotion label at sample time


@dataclass
class TableMonitorState:
    """All per-table monitoring state — lives in memory for the lifetime of the process."""

    table_id:  str
    agent_id:  str
    center_id: str
    camera_id: str

    # ── Face identity reference ───────────────────────────────────────────────
    agent_embedding:  Optional[np.ndarray] = None
    face_photo_path:  Optional[str]        = None

    # ── Idle / presence tracking ──────────────────────────────────────────────
    last_agent_seen_at:  Optional[float] = None   # monotonic; None = never seen
    idle_alert_sent_at:  Optional[float] = None   # throttle: one alert per idle period

    # ── Gossip tracking ───────────────────────────────────────────────────────
    multi_agent_since:    Optional[float] = None  # when ≥2 agents first detected
    gossip_alert_sent_at: Optional[float] = None  # throttle
    gossip_count:         int             = 0     # lifetime gossip events at this table

    # ── SHI tracking ─────────────────────────────────────────────────────────
    shi_samples:        deque = field(default_factory=lambda: deque(maxlen=20))
    last_shi_sample_at: float = 0.0
    latest_shi:         float = 50.0   # start at neutral

    # ── Activity accumulator ──────────────────────────────────────────────────
    active_minutes:      float          = 0.0
    last_active_tick_at: Optional[float] = None  # monotonic

    @property
    def idle_seconds(self) -> float:
        """Seconds since the assigned agent was last seen (0 if seen this frame)."""
        if self.last_agent_seen_at is None:
            return 0.0
        return time.monotonic() - self.last_agent_seen_at

    @property
    def gossip_duration_seconds(self) -> float:
        """How long the multi-agent condition has been active (0 if not active)."""
        if self.multi_agent_since is None:
            return 0.0
        return time.monotonic() - self.multi_agent_since

    @property
    def rolling_shi(self) -> float:
        """Rolling average SHI over the last SHI_WINDOW_SIZE samples."""
        if not self.shi_samples:
            return 50.0
        return round(sum(s.shi for s in self.shi_samples) / len(self.shi_samples), 1)


# ─────────────────────────────────────────────────────────────────────────────
# Report — returned per frame by AgentMonitor.process_frame()
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentMonitorFrameResult:
    """Per-frame output from the monitoring engine."""
    table_id:   str
    agent_id:   str
    timestamp:  float

    # ── Presence ──────────────────────────────────────────────────────────────
    agent_present:        bool  = False
    idle_seconds:         float = 0.0
    idle_alert_triggered: bool  = False

    # ── Gossip ────────────────────────────────────────────────────────────────
    gossip_active:          bool  = False
    gossip_duration_seconds: float = 0.0
    gossip_alert_triggered:  bool  = False

    # ── SHI ───────────────────────────────────────────────────────────────────
    shi_sampled:    bool  = False
    latest_shi:     float = 50.0
    rolling_shi:    float = 50.0
    latest_emotion: str   = "NEUTRAL"

    # ── Activity ──────────────────────────────────────────────────────────────
    active_minutes:       float = 0.0
    activity_patched:     bool  = False   # True when PATCH sent to NestJS

    # ── Matched face count ────────────────────────────────────────────────────
    agent_faces_in_roi:   int = 0
    unknown_faces_in_roi: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# AgentMonitor — the main singleton
# ─────────────────────────────────────────────────────────────────────────────

class AgentMonitor:
    """
    Singleton that tracks monitoring state for every registered table.

    Usage
    -----
    1.  Call `await monitor.register_table(...)` when a table config arrives
        (on server start or when Super Admin updates the mapping).

    2.  On every camera frame, call `await monitor.process_frame(...)` with
        the decoded frame + detected faces from the existing FaceAnalyzer.

    3.  Dispatch functions in alert_client.py are called internally as
        asyncio.create_task() so they never block the response.
    """

    def __init__(self) -> None:
        self._tables:      dict[str, TableMonitorState] = {}
        self._all_agents:  dict[str, np.ndarray]        = {}  # agent_id → embedding (global DB)
        self._lock = asyncio.Lock()
        logger.info("🕵️  AgentMonitor initialised")

    # ── Table registration ────────────────────────────────────────────────────

    async def register_table(
        self,
        table_id:        str,
        agent_id:        str,
        face_photo_path: Optional[str],
        center_id:       str,
        camera_id:       str,
    ) -> bool:
        """
        Register (or re-register) a table and pre-load the agent's face embedding.

        Returns True if the reference embedding was loaded successfully.
        Called during startup (for all provisioned tables) and on-demand when
        a Super Admin reassigns an agent.
        """
        embedding = None
        if face_photo_path:
            embedding = load_reference_embedding(face_photo_path)

        async with self._lock:
            # Preserve accumulated metrics if re-registering an existing table
            existing = self._tables.get(table_id)
            if existing is None:
                state = TableMonitorState(
                    table_id=table_id,
                    agent_id=agent_id,
                    center_id=center_id,
                    camera_id=camera_id,
                    agent_embedding=embedding,
                    face_photo_path=face_photo_path,
                )
            else:
                # Update identity + config; preserve running counters
                existing.agent_id         = agent_id
                existing.center_id        = center_id
                existing.camera_id        = camera_id
                existing.agent_embedding  = embedding
                existing.face_photo_path  = face_photo_path
                # Reset idle/gossip timers on re-registration
                existing.idle_alert_sent_at  = None
                existing.multi_agent_since   = None
                state = existing

            self._tables[table_id] = state

            if embedding is not None:
                self._all_agents[agent_id] = embedding

        status = "✓" if embedding is not None else "✗ (no photo)"
        logger.info(
            f"📋 Table registered — table={table_id}  agent={agent_id}  "
            f"center={center_id}  embedding={status}"
        )
        return embedding is not None

    async def unregister_table(self, table_id: str) -> None:
        async with self._lock:
            self._tables.pop(table_id, None)

    @property
    def registered_tables(self) -> list[str]:
        return list(self._tables.keys())

    # ── Main per-frame entry point ────────────────────────────────────────────

    async def process_frame(
        self,
        table_id:       str,
        frame_bgr:      Optional[np.ndarray],   # OpenCV BGR array for identity crops
        face_detections: list["FaceDetection"],  # from BaseFaceAnalyzer.analyze()
        persons_in_roi: int,                     # from YOLO detection count
        frame_ts:       float,                   # unix timestamp
    ) -> Optional[AgentMonitorFrameResult]:
        """
        Run all three monitoring checks for one table on one frame.

        Returns an AgentMonitorFrameResult, or None if the table is not registered.
        Alert dispatches are fired as background tasks (non-blocking).
        """
        async with self._lock:
            state = self._tables.get(table_id)
        if state is None:
            logger.debug(f"[AgentMonitor] Unknown table_id={table_id} — skipping")
            return None

        now = time.monotonic()
        result = AgentMonitorFrameResult(
            table_id=table_id,
            agent_id=state.agent_id,
            timestamp=frame_ts,
        )

        # ── 1. AGENT IDENTITY VERIFICATION ───────────────────────────────────
        agent_faces_in_roi   = 0
        unknown_faces_in_roi = 0

        roi_faces = [f for f in face_detections if f.is_inside_roi]

        if roi_faces and frame_bgr is not None:
            for face in roi_faces:
                bb = face.bounding_box
                emb = embed_face_crop(frame_bgr, bb.x, bb.y, bb.w, bb.h)
                if emb is None:
                    unknown_faces_in_roi += 1
                    continue

                # Check against the table's own agent first (fast path)
                if (
                    state.agent_embedding is not None
                    and _cosine_dist(emb, state.agent_embedding) <= settings.face_match_threshold
                ):
                    agent_faces_in_roi += 1
                    continue

                # Check against the global agent DB (for gossip detection)
                matched_other = False
                for other_agent_id, other_emb in self._all_agents.items():
                    if other_agent_id == state.agent_id:
                        continue
                    if _cosine_dist(emb, other_emb) <= settings.face_match_threshold:
                        agent_faces_in_roi += 1  # another known agent in this ROI
                        matched_other = True
                        break

                if not matched_other:
                    unknown_faces_in_roi += 1  # unrecognised → likely a customer

        elif roi_faces and frame_bgr is None:
            # No BGR frame supplied — optimistic: assume agent is present if any face detected
            agent_faces_in_roi = len(roi_faces)

        agent_present = agent_faces_in_roi > 0

        # ── 2. IDLE DETECTION ─────────────────────────────────────────────────
        idle_alert_triggered = await self._check_idle(state, agent_present, now)

        # ── 3. GOSSIP DETECTION ───────────────────────────────────────────────
        gossip_alert_triggered = await self._check_gossip(
            state, agent_faces_in_roi, unknown_faces_in_roi, now
        )

        # ── 4. SHI SAMPLING (every SHI_SAMPLE_INTERVAL_SECONDS) ───────────────
        shi_sampled, current_shi, current_emotion = await self._maybe_sample_shi(
            state, roi_faces, now
        )

        # ── 5. ACTIVE-MINUTE ACCUMULATION ────────────────────────────────────
        if agent_present:
            if state.last_active_tick_at is not None:
                elapsed_s = now - state.last_active_tick_at
                state.active_minutes += elapsed_s / 60.0
            state.last_active_tick_at = now
        else:
            state.last_active_tick_at = None

        # ── 6. Periodic PATCH to NestJS /ingest/agent-activity ────────────────
        # Every 60 s (to avoid hammering the DB on each frame)
        activity_patched = False
        if (
            not hasattr(state, "_last_activity_patch_at")
            or (now - getattr(state, "_last_activity_patch_at", 0)) >= 60.0
        ):
            asyncio.create_task(
                _patch_agent_activity_task(
                    table_id=state.table_id,
                    user_id=state.agent_id,
                    active_minutes=round(state.active_minutes, 2),
                    gossip_count=state.gossip_count,
                    avg_sentiment_score=round((state.rolling_shi - 50.0) / 50.0, 4),
                )
            )
            object.__setattr__(state, "_last_activity_patch_at", now)
            activity_patched = True

        # ── Build result ──────────────────────────────────────────────────────
        result.agent_present          = agent_present
        result.idle_seconds           = round(state.idle_seconds, 1)
        result.idle_alert_triggered   = idle_alert_triggered
        result.gossip_active          = state.multi_agent_since is not None
        result.gossip_duration_seconds = round(state.gossip_duration_seconds, 1)
        result.gossip_alert_triggered  = gossip_alert_triggered
        result.shi_sampled             = shi_sampled
        result.latest_shi              = current_shi
        result.rolling_shi             = state.rolling_shi
        result.latest_emotion          = current_emotion
        result.active_minutes          = round(state.active_minutes, 2)
        result.activity_patched        = activity_patched
        result.agent_faces_in_roi      = agent_faces_in_roi
        result.unknown_faces_in_roi    = unknown_faces_in_roi

        return result

    # ── Idle detection ────────────────────────────────────────────────────────

    async def _check_idle(
        self,
        state: TableMonitorState,
        agent_present: bool,
        now: float,
    ) -> bool:
        """
        Update idle state and fire IDLE_AGENT alert if threshold crossed.

        Returns True if an alert was dispatched this frame.
        """
        if agent_present:
            state.last_agent_seen_at = now
            state.idle_alert_sent_at = None   # reset — agent is back
            return False

        # Agent absent; only alert during work hours
        if not _is_work_hours():
            return False

        if state.last_agent_seen_at is None:
            # Haven't established a baseline yet
            state.last_agent_seen_at = now
            return False

        idle_min = state.idle_seconds / 60.0
        threshold_min = settings.idle_threshold_minutes

        if idle_min < threshold_min:
            return False

        # Throttle: don't re-fire within the same idle period
        if state.idle_alert_sent_at is not None:
            return False

        # ── Fire IDLE_AGENT alert ─────────────────────────────────────────────
        state.idle_alert_sent_at = now
        logger.warning(
            f"🚨 IDLE_AGENT — table={state.table_id}  agent={state.agent_id}  "
            f"absent={idle_min:.1f} min  center={state.center_id}"
        )
        asyncio.create_task(
            _dispatch_alert_task(
                center_id=state.center_id,
                camera_id=state.camera_id,
                table_id=state.table_id,
                agent_id=state.agent_id,
                anomaly_type="IDLE_AGENT",
                severity="MEDIUM",
                extra={
                    "idleMinutes":        round(idle_min, 1),
                    "thresholdMinutes":   threshold_min,
                    "lastSeenAt":         datetime.fromtimestamp(
                        time.time() - (now - state.last_agent_seen_at)
                    ).isoformat(),
                },
            )
        )
        return True

    # ── Gossip detection ──────────────────────────────────────────────────────

    async def _check_gossip(
        self,
        state: TableMonitorState,
        agent_faces_in_roi: int,
        unknown_faces_in_roi: int,
        now: float,
    ) -> bool:
        """
        Start/stop the gossip timer and fire GOSSIP_DETECTED alert.

        Gossip condition:
          ≥2 known-agent faces in ROI AND no unrecognised customer face.

        Returns True if an alert was dispatched this frame.
        """
        # Two or more agents AND no customer detected → gossip condition
        gossip_condition = (agent_faces_in_roi >= 2) and (unknown_faces_in_roi == 0)

        if not gossip_condition:
            state.multi_agent_since = None   # condition lifted — reset timer
            return False

        if state.multi_agent_since is None:
            state.multi_agent_since = now    # start the clock

        duration = time.monotonic() - state.multi_agent_since
        threshold = settings.gossip_threshold_seconds

        if duration < threshold:
            return False

        # Throttle: cooldown = threshold (fire once per sustained gossip window)
        if state.gossip_alert_sent_at is not None:
            if (now - state.gossip_alert_sent_at) < threshold:
                return False

        # ── Fire GOSSIP_DETECTED alert ────────────────────────────────────────
        state.gossip_count       += 1
        state.gossip_alert_sent_at = now
        logger.warning(
            f"💬 GOSSIP_DETECTED — table={state.table_id}  agents_detected={agent_faces_in_roi}  "
            f"duration={duration:.0f}s  count=#{state.gossip_count}"
        )
        asyncio.create_task(
            _dispatch_alert_task(
                center_id=state.center_id,
                camera_id=state.camera_id,
                table_id=state.table_id,
                agent_id=state.agent_id,
                anomaly_type="GOSSIP_DETECTED",
                severity="LOW",
                extra={
                    "agentFacesDetected": agent_faces_in_roi,
                    "durationSeconds":    round(duration, 1),
                    "gossipCount":        state.gossip_count,
                },
            )
        )
        return True

    # ── SHI sampling ──────────────────────────────────────────────────────────

    async def _maybe_sample_shi(
        self,
        state: TableMonitorState,
        roi_faces: list["FaceDetection"],
        now: float,
    ) -> tuple[bool, float, str]:
        """
        Sample the agent's SHI every SHI_SAMPLE_INTERVAL_SECONDS.

        Returns:
            (sampled: bool, current_shi: float, emotion_label: str)
        """
        interval = settings.shi_sample_interval_seconds
        if (now - state.last_shi_sample_at) < interval:
            return False, state.latest_shi, "NEUTRAL"

        if not roi_faces:
            return False, state.latest_shi, "NEUTRAL"

        # Pick the highest-clarity face in the ROI (the agent's face)
        best_face = max(roi_faces, key=lambda f: f.clarity_score)
        emotion_scores = best_face.emotion_scores
        dominant_emotion = best_face.dominant_emotion.value if hasattr(best_face.dominant_emotion, "value") else str(best_face.dominant_emotion)

        if not emotion_scores:
            return False, state.latest_shi, dominant_emotion

        raw, shi = emotion_scores_to_shi(emotion_scores)
        sample = SHISample(ts=now, raw=raw, shi=shi, emotion=dominant_emotion)
        state.shi_samples.append(sample)
        state.last_shi_sample_at = now
        state.latest_shi = shi

        logger.debug(
            f"📊 SHI sample — table={state.table_id}  emotion={dominant_emotion}  "
            f"raw={raw:.3f}  shi={shi:.1f}  rolling={state.rolling_shi:.1f}"
        )
        return True, shi, dominant_emotion

    # ── Snapshot for API ──────────────────────────────────────────────────────

    def get_table_snapshot(self, table_id: str) -> Optional[dict]:
        """Return a JSON-serialisable summary of a table's current monitoring state."""
        state = self._tables.get(table_id)
        if state is None:
            return None
        return {
            "tableId":              state.table_id,
            "agentId":              state.agent_id,
            "centerId":             state.center_id,
            "embeddingLoaded":      state.agent_embedding is not None,
            "idleSeconds":          round(state.idle_seconds, 1),
            "idleAlertPending":     state.idle_alert_sent_at is not None,
            "gossipActive":         state.multi_agent_since is not None,
            "gossipDurationSeconds": round(state.gossip_duration_seconds, 1),
            "gossipCount":          state.gossip_count,
            "latestShi":            state.latest_shi,
            "rollingShi":           state.rolling_shi,
            "activeMinutes":        round(state.active_minutes, 2),
            "shiSamples": [
                {"ts": s.ts, "shi": s.shi, "emotion": s.emotion}
                for s in list(state.shi_samples)
            ],
        }

    def all_snapshots(self) -> list[dict]:
        return [self.get_table_snapshot(tid) for tid in self._tables]


# ─────────────────────────────────────────────────────────────────────────────
# Background task helpers
# (asyncio.create_task() wrappers so they never block process_frame)
# ─────────────────────────────────────────────────────────────────────────────

async def _dispatch_alert_task(
    center_id:    str,
    camera_id:    str,
    table_id:     str,
    agent_id:     str,
    anomaly_type: str,    # "IDLE_AGENT" | "GOSSIP_DETECTED"
    severity:     str,
    extra:        dict,
) -> None:
    """POST an IDLE_AGENT or GOSSIP_DETECTED alert to NestJS /ingest/ai-alert."""
    from alert_client import get_http_client

    payload = {
        "center_id":    center_id,
        "camera_id":    camera_id,
        "table_id":     table_id,
        "anomaly_type": anomaly_type,
        "severity":     severity,
        "primary_event": anomaly_type,
        "detections":   [],
        "timestamp":    time.time(),
        "source":       "agent-monitor",
        "tech":         "CCTV",
        # Agent-specific metadata folded into the detections-like field
        "metadata": {"agentId": agent_id, **extra},
    }

    client = get_http_client()
    try:
        resp = await client.post("/ingest/ai-alert", json=payload)
        if resp.is_success:
            logger.success(f"✅ {anomaly_type} dispatched — table={table_id} agent={agent_id}")
        else:
            logger.warning(f"⚠️  {anomaly_type} dispatch HTTP {resp.status_code}: {resp.text[:120]}")
    except Exception as exc:
        logger.error(f"❌ {anomaly_type} dispatch failed: {exc}")


async def _patch_agent_activity_task(
    table_id:            str,
    user_id:             str,
    active_minutes:      float,
    gossip_count:        int,
    avg_sentiment_score: float,
) -> None:
    """PATCH running AgentActivity metrics to NestJS /ingest/agent-activity."""
    from alert_client import get_http_client

    payload = {
        "table_id":            table_id,
        "user_id":             user_id,
        "active_minutes":      active_minutes,
        "gossip_count":        gossip_count,
        "avg_sentiment_score": avg_sentiment_score,
    }
    client = get_http_client()
    try:
        resp = await client.patch("/ingest/agent-activity", json=payload)
        if not resp.is_success:
            logger.debug(f"AgentActivity PATCH {resp.status_code}: {resp.text[:80]}")
    except Exception as exc:
        logger.debug(f"AgentActivity PATCH failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton — imported by main.py
# ─────────────────────────────────────────────────────────────────────────────

agent_monitor = AgentMonitor()
