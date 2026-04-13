"""
alert_client.py — Async HTTP client for dispatching anomaly alerts to NestJS.

Flow:
  MqttController (NestJS) already handles inbound MQTT from ESP32 nodes.
  The AI service uses a *separate* REST call back to NestJS so the alert
  goes through the same enrichment + WebSocket broadcast pipeline:

  AI-service ──POST /ingest/ai-alert──► NestJS backend
                                            │
                                   publishes on MQTT topic
                                   falcon/center/{id}/ai-results
                                            │
                                   MqttController picks it up
                                            │
                                   EventsGateway.emitToCenterAndSuperAdmin()
                                            │
                                   ┌────────────────────┐
                                   │  room:super_admin  │
                                   │  room:center:{id}  │
                                   └────────────────────┘

If the NestJS /ingest/ai-alert endpoint is not yet implemented, the alert is
simply logged with a warning and `alert_dispatched` is set to False.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import httpx
from loguru import logger

from config import get_settings
from schemas import AlertPayload, Detection, AnomalyType, AlertSeverity

if TYPE_CHECKING:
    pass

settings = get_settings()

# Reuse a single async client for connection pooling.
# Created at module import time; closed on app shutdown via lifespan.
_http_client: httpx.AsyncClient | None = None


def get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            base_url=settings.nestjs_api_url,
            headers={
                "Content-Type": "application/json",
                "X-Service-Key": settings.nestjs_service_key,
            },
            timeout=httpx.Timeout(5.0),
        )
    return _http_client


async def close_http_client() -> None:
    global _http_client
    if _http_client and not _http_client.is_closed:
        await _http_client.aclose()
        logger.info("🔌 HTTP client closed")


# ─────────────────────────────────────────────────────────────────────────────
# Anomaly classification helpers
# ─────────────────────────────────────────────────────────────────────────────

_WEAPON_CLASSES = {"knife", "gun", "pistol", "rifle", "weapon"}
_FIGHT_CLASSES = {"fight", "aggression", "violence"}
_FIRE_CLASSES = {"fire", "smoke"}

def classify_anomaly(class_name: str) -> tuple[AnomalyType, AlertSeverity]:
    """Map a YOLO class name to an AnomalyType + AlertSeverity."""
    name = class_name.lower()
    if name in _WEAPON_CLASSES:
        return AnomalyType.WEAPON_DETECTED, AlertSeverity.CRITICAL
    if name in _FIGHT_CLASSES:
        return AnomalyType.FIGHT_DETECTED, AlertSeverity.HIGH
    if name in _FIRE_CLASSES:
        return AnomalyType.FIRE_DETECTED, AlertSeverity.HIGH
    if name == "fall":
        return AnomalyType.FALL_DETECTED, AlertSeverity.CRITICAL
    return AnomalyType.UNKNOWN, AlertSeverity.MEDIUM


# ─────────────────────────────────────────────────────────────────────────────
# Main dispatch function (called as a FastAPI BackgroundTask)
# ─────────────────────────────────────────────────────────────────────────────

async def dispatch_alert(
    center_id: str,
    camera_id: str,
    table_id: str | None,
    anomaly_type: AnomalyType,
    severity: AlertSeverity,
    detections: list[Detection],
) -> tuple[bool, int | None]:
    """
    POST an alert to the NestJS /ingest/ai-alert endpoint.

    Returns:
        (dispatched: bool, http_status: int | None)
    """
    payload = AlertPayload(
        center_id=center_id,
        camera_id=camera_id,
        table_id=table_id,
        anomaly_type=anomaly_type.value,
        severity=severity.value,
        primary_event=anomaly_type.value,
        detections=[
            {
                "class_name": d.class_name,
                "confidence": d.confidence,
                "bounding_box": d.bounding_box.model_dump(),
                "is_inside_roi": d.is_inside_roi,
                "overlap_ratio": d.overlap_ratio,
            }
            for d in detections
        ],
        timestamp=time.time(),
    )

    client = get_http_client()
    url = "/ingest/ai-alert"

    try:
        response = await client.post(url, json=payload.model_dump())
        status = response.status_code

        if response.is_success:
            logger.success(
                f"✅ Alert dispatched → NestJS {url} "
                f"[{status}] center={center_id} anomaly={anomaly_type.value} severity={severity.value}"
            )
        else:
            logger.warning(
                f"⚠️  Alert dispatch returned {status} → NestJS {url} "
                f"center={center_id}  body={response.text[:200]}"
            )

        return True, status

    except httpx.ConnectError:
        logger.error(
            f"❌ Cannot reach NestJS backend at {settings.nestjs_api_url}{url} "
            f"— is the backend running?"
        )
        return False, None

    except httpx.TimeoutException:
        logger.error(
            f"⏱️  Alert dispatch timed out → NestJS {url} center={center_id}"
        )
        return False, None

    except Exception as exc:  # noqa: BLE001
        logger.exception(f"❌ Unexpected error dispatching alert: {exc}")
        return False, None


# ─────────────────────────────────────────────────────────────────────────────
# Irate Customer alert (cross-camera face emotion pipeline)
# ─────────────────────────────────────────────────────────────────────────────

async def dispatch_face_alert(
    center_id:        str,
    camera_id:        str,
    table_id:         str | None,
    desk_id:          str,
    dominant_emotion: str,
    irate_confidence: float,
) -> tuple[bool, int | None]:
    """
    POST an IRATE_CUSTOMER alert to the NestJS /ingest/ai-alert endpoint.

    Called by POST /analyze-cross-camera as a BackgroundTask whenever the
    BestViewSelector determines that irate_confidence >= IRATE_EMOTION_THRESHOLD
    for a desk.

    The ``detections`` list carries face-emotion metadata instead of YOLO boxes
    so NestJS can enrich the WS event with emotion context for the dashboard.

    Returns:
        (dispatched: bool, http_status: int | None)
    """
    payload = AlertPayload(
        center_id=center_id,
        camera_id=camera_id,
        table_id=table_id,
        anomaly_type=AnomalyType.IRATE_CUSTOMER.value,
        severity=AlertSeverity.HIGH.value,
        primary_event=AnomalyType.IRATE_CUSTOMER.value,
        detections=[
            {
                "source":           "cross-camera-face-analysis",
                "desk_id":          desk_id,
                "dominant_emotion": dominant_emotion,
                "irate_confidence": round(irate_confidence, 4),
            }
        ],
        timestamp=time.time(),
    )

    client = get_http_client()
    url    = "/ingest/ai-alert"

    try:
        response = await client.post(url, json=payload.model_dump())
        status   = response.status_code

        if response.is_success:
            logger.success(
                f"✅ IRATE_CUSTOMER alert dispatched → NestJS [{status}]  "
                f"center={center_id}  desk={desk_id}  "
                f"emotion={dominant_emotion}  conf={irate_confidence:.2f}"
            )
        else:
            logger.warning(
                f"⚠️  IRATE_CUSTOMER dispatch returned {status} → NestJS {url}  "
                f"center={center_id}  body={response.text[:200]}"
            )

        return True, status

    except httpx.ConnectError:
        logger.error(
            f"❌ IRATE_CUSTOMER alert: cannot reach NestJS at "
            f"{settings.nestjs_api_url}{url} — is the backend running?"
        )
        return False, None

    except httpx.TimeoutException:
        logger.error(
            f"⏱️  IRATE_CUSTOMER alert timed out → NestJS {url}  "
            f"center={center_id}  desk={desk_id}"
        )
        return False, None

    except Exception as exc:  # noqa: BLE001
        logger.exception(f"❌ Unexpected error dispatching IRATE_CUSTOMER alert: {exc}")
        return False, None
