"""
main.py — Falcon Security AI Microservice (Edge Worker)
=======================================================

Responsibilities:
  1. Accept a camera frame + table bounding box over POST /analyze-frame
  2. Run YOLOv8 inference (real or mock) on the full frame
  3. Filter detections to those that overlap the table's ROI
  4. Classify anomalies (weapon, fight, fire, fall…)
  5. If an anomaly is detected in the ROI → dispatch an alert to the NestJS API
     via a FastAPI BackgroundTask (non-blocking, doesn't delay the response)

The service is stateless — no database, no persistent state.
It does not talk to MQTT directly; alerts flow via NestJS which re-publishes
them on the MQTT broker so the existing EventsGateway WebSocket pipeline
broadcasts them to the frontend.

Run:
  uvicorn main:app --reload --port 8000
  # or:
  python main.py
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, BackgroundTasks, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from alert_client import classify_anomaly, dispatch_alert, close_http_client
from config import get_settings
from inference import decode_frame, filter_detections_by_roi, load_model, BaseYoloModel
from schemas import (
    AlertSeverity,
    AnomalyType,
    AnalyzeFrameRequest,
    AnalyzeFrameResponse,
    Detection,
)

settings = get_settings()

# ─────────────────────────────────────────────────────────────────────────────
# Model singleton — loaded once at startup, reused for every request
# ─────────────────────────────────────────────────────────────────────────────

model: BaseYoloModel | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model at startup; release HTTP client at shutdown."""
    global model

    logger.info("🚀 Falcon AI Service starting up…")
    logger.info(f"   Environment : {settings.app_env}")
    logger.info(f"   Model path  : {settings.model_path}")
    logger.info(f"   NestJS URL  : {settings.nestjs_api_url}")
    logger.info(f"   Alert classes: {settings.alert_classes}")

    model = load_model()
    logger.success("✅ Model ready — service is live")

    yield  # ← application runs here

    logger.info("🛑 Shutting down — releasing resources…")
    await close_http_client()
    logger.info("✅ Shutdown complete")


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Falcon Security — AI Microservice",
    description=(
        "Edge worker that runs YOLOv8 inference on camera frames and dispatches "
        "anomaly alerts back to the Falcon Security NestJS backend.\n\n"
        "**No database.** Stateless. Designed to run on-site (edge) or in the cloud."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health() -> dict:
    return {
        "status": "ok",
        "service": "falcon-ai-service",
        "model": settings.model_path,
        "env": settings.app_env,
        "server_time": datetime.now(timezone.utc).isoformat(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# POST /analyze-frame — THE CORE ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────

@app.post(
    "/analyze-frame",
    response_model=AnalyzeFrameResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
    summary="Analyze a camera frame for anomalies within a table's bounding box",
    description="""
Accepts a base64-encoded image frame and the table's registered bounding box.

**Processing pipeline:**
1. Decode and optionally downscale the frame
2. Run YOLOv8 inference on the full frame
3. For each detection, calculate overlap with the table's ROI bounding box
4. Determine if a person is present inside the ROI
5. Check if any alert-class object (weapon, fight, fire…) is inside the ROI
6. If an anomaly is detected → dispatch a `BackgroundTask` that POSTs an alert
   to the NestJS `/ingest/ai-alert` endpoint (non-blocking)
7. Return the full analysis result immediately

**Alert classes** are configured via the `ALERT_CLASSES` environment variable.

**Switching from mock to real YOLOv8:** set `MODEL_PATH=yolov8n.pt` in `.env`
and install `ultralytics` (`pip install ultralytics`).
    """,
)
async def analyze_frame(
    request: AnalyzeFrameRequest,
    background_tasks: BackgroundTasks,
) -> AnalyzeFrameResponse:
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded — service is still starting up",
        )

    request_time = time.time() if request.timestamp is None else request.timestamp
    server_time = datetime.now(timezone.utc).isoformat()

    # ── 1. Decode frame ───────────────────────────────────────────────────────
    try:
        image = decode_frame(request.frame_base64, max_size=settings.max_image_size)
    except Exception as exc:
        logger.warning(f"Frame decode failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not decode frame_base64: {exc}",
        ) from exc

    logger.info(
        f"📸 Analyzing frame  center={request.center_id}  "
        f"camera={request.camera_id}  table={request.table_id}  "
        f"size={image.size}"
    )

    # ── 2. Run inference ──────────────────────────────────────────────────────
    raw_detections, inference_ms = model.predict(image)

    logger.info(
        f"🤖 Inference done in {inference_ms:.1f} ms  — "
        f"{len(raw_detections)} raw detections"
    )

    # ── 3. Filter by ROI ──────────────────────────────────────────────────────
    detections = filter_detections_by_roi(raw_detections, request.bounding_box)

    roi_detections = [d for d in detections if d.is_inside_roi]
    person_in_roi = any(
        d.class_name.lower() == "person" and d.is_inside_roi for d in detections
    )

    logger.info(
        f"   {len(roi_detections)}/{len(detections)} detections inside ROI  |  "
        f"person_in_roi={person_in_roi}"
    )

    # ── 4. Anomaly detection ──────────────────────────────────────────────────
    alert_class_set = settings.alert_class_set
    anomaly_detection: Detection | None = None

    # Only anomalies INSIDE the table's ROI trigger an alert
    for det in roi_detections:
        if det.class_name.lower() in alert_class_set:
            # Pick the highest-confidence alert-class detection
            if anomaly_detection is None or det.confidence > anomaly_detection.confidence:
                anomaly_detection = det

    anomaly_detected = anomaly_detection is not None
    anomaly_type: AnomalyType | None = None
    severity = AlertSeverity.INFO
    alert_dispatched = False
    alert_dispatch_status: int | None = None

    if anomaly_detected and anomaly_detection is not None:
        anomaly_type, severity = classify_anomaly(anomaly_detection.class_name)

        logger.warning(
            f"🚨 ANOMALY in ROI  center={request.center_id}  "
            f"class={anomaly_detection.class_name}  "
            f"confidence={anomaly_detection.confidence:.2f}  "
            f"anomaly_type={anomaly_type.value}  severity={severity.value}"
        )

        # ── 5. Dispatch alert (non-blocking BackgroundTask) ───────────────────
        # The response is returned immediately; the HTTP call to NestJS happens
        # in the background so the caller is not blocked on network I/O.
        async def _do_dispatch():
            nonlocal alert_dispatched, alert_dispatch_status
            dispatched, http_status = await dispatch_alert(
                center_id=request.center_id,
                camera_id=request.camera_id,
                table_id=request.table_id,
                anomaly_type=anomaly_type,  # type: ignore[arg-type]
                severity=severity,
                detections=roi_detections,
            )
            # Note: these assignments won't affect the already-sent response.
            # They are here only for completeness / local logging.
            alert_dispatched = dispatched
            alert_dispatch_status = http_status

        background_tasks.add_task(_do_dispatch)

        # We optimistically set these in the response before the task runs.
        # The actual HTTP result is logged by alert_client.dispatch_alert.
        alert_dispatched = True
        alert_dispatch_status = None  # will be populated asynchronously

    else:
        logger.debug(
            f"✅ No anomaly  center={request.center_id}  "
            f"camera={request.camera_id}  roi_detections={len(roi_detections)}"
        )

    # ── 6. Build and return response ──────────────────────────────────────────
    return AnalyzeFrameResponse(
        center_id=request.center_id,
        camera_id=request.camera_id,
        table_id=request.table_id,
        person_detected_in_roi=person_in_roi,
        anomaly_detected=anomaly_detected,
        anomaly_type=anomaly_type,
        severity=severity,
        detections=detections,
        inference_ms=inference_ms,
        server_time=server_time,
        alert_dispatched=alert_dispatched,
        alert_dispatch_status=alert_dispatch_status,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Optional: multipart/form-data endpoint for high-throughput use
# ─────────────────────────────────────────────────────────────────────────────
# Uncomment this block for production use.  Multipart avoids the ~33% base64
# overhead and is faster for large frames.
#
# from fastapi import File, Form, UploadFile
#
# @app.post("/analyze-frame/upload", tags=["Inference"])
# async def analyze_frame_upload(
#     center_id: str = Form(...),
#     camera_id: str = Form(...),
#     table_id: str | None = Form(None),
#     bounding_box: str = Form(...),   # JSON string of BoundingBox
#     frame: UploadFile = File(...),
#     background_tasks: BackgroundTasks = BackgroundTasks(),
# ):
#     import json, base64
#     raw = await frame.read()
#     b64 = base64.b64encode(raw).decode()
#     bb = BoundingBox(**json.loads(bounding_box))
#     req = AnalyzeFrameRequest(
#         center_id=center_id, camera_id=camera_id, table_id=table_id,
#         frame_base64=b64, bounding_box=bb,
#     )
#     return await analyze_frame(req, background_tasks)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.app_port,
        reload=settings.app_env == "development",
        workers=1 if settings.app_env == "development" else settings.workers,
        log_level="info",
    )
