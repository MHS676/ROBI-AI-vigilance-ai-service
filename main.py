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

from fastapi import FastAPI, BackgroundTasks, HTTPException, status, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from alert_client import classify_anomaly, dispatch_alert, dispatch_face_alert, close_http_client
from agent_monitor import agent_monitor
from audio import (
    load_whisper_model,
    get_whisper_model,
    transcribe_audio_bytes,
    SUPPORTED_AUDIO_TYPES,
    MAX_AUDIO_BYTES,
)
from config import get_settings
from inference import (
    decode_frame,
    filter_detections_by_roi,
    load_model,
    BaseYoloModel,
    BaseFaceAnalyzer,
    BestViewSelector,
    load_face_analyzer,
)
import cross_camera
from schemas import (
    AlertSeverity,
    AnomalyType,
    AnalyzeFrameRequest,
    AnalyzeFrameResponse,
    CameraConfigUpdate,
    CameraConfigResponse,
    AiFeature,
    CLASS_TO_FEATURE,
    Detection,
    # Cross-camera face analysis
    CrossCameraLayout,
    CrossCameraAnalyzeRequest,
    CrossCameraAnalyzeResponse,
    FaceDetection,
    DeskRoi,
    BestViewResult,
    # Agent monitoring
    TableRegistrationRequest,
    TableRegistrationResponse,
    AgentMonitorFrameRequest,
    AgentMonitorReport,
    SHISampleOut,
)

settings = get_settings()

# ─────────────────────────────────────────────────────────────────────────────
# Model singleton — loaded once at startup, reused for every request
# ─────────────────────────────────────────────────────────────────────────────

model: BaseYoloModel | None = None

# Face emotion analyzer singleton — loaded once at startup alongside `model`
face_analyzer: BaseFaceAnalyzer | None = None

# ─────────────────────────────────────────────────────────────────────────────
# Per-camera AI feature configuration (in-memory)
#
# Populated by POST /camera-config (called by NestJS when a Super Admin
# toggles a feature on the Frontend).
#
# Structure: { camera_id: set_of_enabled_AiFeature_strings }
# Default (key not present): ALL features are enabled.
# ─────────────────────────────────────────────────────────────────────────────

camera_configs: dict[str, set[str]] = {}

# All feature values — used as the "all enabled" default
_ALL_FEATURES: set[str] = {f.value for f in AiFeature}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model at startup; release HTTP client at shutdown."""
    global model, face_analyzer

    logger.info("🚀 Falcon AI Service starting up…")
    logger.info(f"   Environment : {settings.app_env}")
    logger.info(f"   Model path  : {settings.model_path}")
    logger.info(f"   NestJS URL  : {settings.nestjs_api_url}")
    logger.info(f"   Alert classes: {settings.alert_classes}")

    model         = load_model()
    face_analyzer = load_face_analyzer()
    logger.success("✅ Model + FaceAnalyzer ready — service is live")
    logger.info(f"🕵️  AgentMonitor ready — {len(agent_monitor.registered_tables)} tables pre-registered")

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
        "configured_cameras": len(camera_configs),
    }


# ─────────────────────────────────────────────────────────────────────────────
# POST /camera-config — Dynamic AI feature toggle (called by NestJS backend)
# ─────────────────────────────────────────────────────────────────────────────

@app.post(
    "/camera-config",
    response_model=CameraConfigResponse,
    status_code=status.HTTP_200_OK,
    tags=["Configuration"],
    summary="Update enabled AI features for a specific camera",
    description="""
Called by the NestJS backend whenever a Super Admin toggles an AI feature
on the Hardware Mapping page.

Stores the enabled feature set in memory. On the **next** `/analyze-frame`
call for this camera, only detections that belong to an enabled feature will
be considered alert candidates.

An empty `enabled_features` list disables all AI detection for that camera.
Sending all five features re-enables everything.
    """,
)
async def update_camera_config(request: CameraConfigUpdate) -> CameraConfigResponse:
    camera_configs[request.camera_id] = {f.value for f in request.enabled_features}
    logger.info(
        f"⚙️  Camera config updated  camera={request.camera_id}  "
        f"features={[f.value for f in request.enabled_features]}"
    )
    return CameraConfigResponse(
        camera_id=request.camera_id,
        enabled_features=request.enabled_features,
        message=f"Config updated — {len(request.enabled_features)} feature(s) enabled",
    )


@app.get(
    "/camera-config/{camera_id}",
    response_model=CameraConfigResponse,
    tags=["Configuration"],
    summary="Get current AI feature config for a camera",
)
async def get_camera_config(camera_id: str) -> CameraConfigResponse:
    enabled_set = camera_configs.get(camera_id, _ALL_FEATURES)
    features = [AiFeature(f) for f in enabled_set if f in _ALL_FEATURES]
    return CameraConfigResponse(
        camera_id=camera_id,
        enabled_features=features,
        message="default (all enabled)" if camera_id not in camera_configs else "custom config",
    )


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

    # Per-camera enabled feature gate:
    # If the camera has a custom config, only consider detections whose class
    # maps to an enabled AiFeature. Default (no config): all features allowed.
    enabled_features: set[str] = camera_configs.get(request.camera_id, _ALL_FEATURES)

    anomaly_detection: Detection | None = None

    # Only anomalies INSIDE the table's ROI trigger an alert
    for det in roi_detections:
        if det.class_name.lower() in alert_class_set:
            # Check if the feature gate allows this class
            required_feature = CLASS_TO_FEATURE.get(det.class_name.lower())
            if required_feature and required_feature.value not in enabled_features:
                logger.debug(
                    f"🔇 Skipping {det.class_name} — feature "
                    f"{required_feature.value} disabled for camera {request.camera_id}"
                )
                continue
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
# Layout management — Cross-Camera ROI Registry
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/layouts",
    tags=["Cross-Camera"],
    summary="List all registered cross-camera layouts",
)
async def list_layouts() -> dict:
    return {
        "layouts": [
            {
                "layout_id":         lid,
                "description":       layout.description,
                "primary_camera_id": layout.primary_camera_id,
                "cameras":           [c.camera_id for c in layout.cameras],
                "desks":             layout.all_desk_ids,
            }
            for lid, layout in cross_camera.LAYOUT_REGISTRY.items()
        ]
    }


@app.get(
    "/layouts/{layout_id}",
    response_model=CrossCameraLayout,
    tags=["Cross-Camera"],
    summary="Get a registered cross-camera layout by ID",
)
async def get_layout(layout_id: str) -> CrossCameraLayout:
    layout = cross_camera.get_layout(layout_id)
    if layout is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Layout '{layout_id}' not found. "
                   f"Available: {list(cross_camera.LAYOUT_REGISTRY.keys())}",
        )
    return layout


@app.post(
    "/layouts",
    response_model=CrossCameraLayout,
    status_code=status.HTTP_201_CREATED,
    tags=["Cross-Camera"],
    summary="Register a new cross-camera layout",
    description=(
        "Registers a custom cross-camera layout in memory. "
        "Layouts are reset on service restart. "
        "To make a layout permanent, add it to cross_camera.py."
    ),
)
async def register_layout(layout: CrossCameraLayout) -> CrossCameraLayout:
    cross_camera.register_layout(layout)
    logger.info(
        f"✇️  Layout registered  id={layout.layout_id}  "
        f"cameras={[c.camera_id for c in layout.cameras]}"
    )
    return layout


# ─────────────────────────────────────────────────────────────────────────────
# POST /analyze-cross-camera — Objective 13: Irate Customer
# ─────────────────────────────────────────────────────────────────────────────

@app.post(
    "/analyze-cross-camera",
    response_model=CrossCameraAnalyzeResponse,
    status_code=status.HTTP_200_OK,
    tags=["Cross-Camera"],
    summary="Objective 13: Irate Customer — Cross-camera face emotion analysis",
    description="""
Accepts frames from two or more cameras that share a named cross-camera layout.

**Processing pipeline for each desk in the layout:**
```
1.  Decode all incoming frames
2.  For every camera × every desk ROI:
      Run BaseFaceAnalyzer.analyze(frame, desk_roi)
      └──► list[FaceDetection] with emotion scores + clarity metrics
3.  For every desk:
      Collect (camera_id, faces) from every camera that provided a frame
      Run BestViewSelector.select(desk_roi, candidates)
      └──► Picks the camera with the highest-clarity face in that desk’s ROI
      └──► Uses that face’s emotion scores for the final Expression Score
4.  For each desk where is_irate=True:
      Dispatch BackgroundTask → POST /ingest/ai-alert (NestJS)
      └──► NestJS broadcasts ‘irate_customer’ WS event to room:super_admin
         and room:center:{centerId}
5.  Return CrossCameraAnalyzeResponse with per-desk BestViewResult
```

**Built-in layout:** `layout_cam1_cam2` — Camera 1 (frontal, 3 desk thirds)
+ Camera 2 (side-angle face view, mirrored desk positions).
    """,
)
async def analyze_cross_camera(
    request:          CrossCameraAnalyzeRequest,
    background_tasks: BackgroundTasks,
) -> CrossCameraAnalyzeResponse:
    if face_analyzer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="FaceAnalyzer not loaded — service is still starting up",
        )

    t0          = time.time()
    server_time = datetime.now(timezone.utc).isoformat()

    # ── 1. Look up layout ─────────────────────────────────────────────────────
    layout = cross_camera.get_layout(request.layout_id)
    if layout is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Layout '{request.layout_id}' not registered. "
                f"Available: {list(cross_camera.LAYOUT_REGISTRY.keys())}"
            ),
        )

    logger.info(
        f"🎥 Cross-camera analysis  center={request.center_id}  "
        f"layout={request.layout_id}  cameras={[f.camera_id for f in request.frames]}"
    )

    # ── 2. Decode all frames ───────────────────────────────────────────────
    from PIL import Image as _PILImage
    frame_map: dict[str, _PILImage.Image] = {}
    for frame_input in request.frames:
        try:
            img = decode_frame(frame_input.frame_base64, max_size=settings.max_image_size)
            frame_map[frame_input.camera_id] = img
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Could not decode frame for camera '{frame_input.camera_id}': {exc}",
            ) from exc

    # ── 3. Run face analysis: for each camera × each desk ROI ─────────────
    # Structure: face_analyses[camera_id][desk_id] = list[FaceDetection]
    face_analyses: dict[str, dict[str, list[FaceDetection]]] = {}

    for camera_id, image in frame_map.items():
        cam_layout = layout.get_camera_layout(camera_id)
        if cam_layout is None:
            logger.warning(
                f"⚠️  Camera '{camera_id}' not in layout '{request.layout_id}' — skipping"
            )
            continue

        face_analyses[camera_id] = {}
        for desk_roi in cam_layout.desk_rois:
            faces = face_analyzer.analyze(image, desk_roi.bounding_box)
            face_analyses[camera_id][desk_roi.desk_id] = faces
            logger.info(
                f"🎞️  Face analysis  camera={camera_id}  desk={desk_roi.desk_id}  "
                f"roi={desk_roi.bounding_box}  faces_found={len(faces)}"
            )

    # ── 4. Cross-camera correlation + Best View Selection per desk ─────────
    selector: BestViewSelector = BestViewSelector()
    desk_results: list[BestViewResult] = []

    for desk_id in layout.all_desk_ids:
        # Canonical DeskRoi definition for this desk (use first camera that has it)
        desk_roi_def: DeskRoi | None = None
        for cam in layout.cameras:
            roi = cam.get_roi_for_desk(desk_id)
            if roi:
                desk_roi_def = roi
                break
        if desk_roi_def is None:
            continue

        # Collect (camera_id, faces) for this desk from every camera that
        # sent a frame.  Cameras that did not send a frame are excluded.
        candidates: list[tuple[str, list[FaceDetection]]] = [
            (
                cam.camera_id,
                face_analyses.get(cam.camera_id, {}).get(desk_id, []),
            )
            for cam in layout.cameras
            if cam.camera_id in frame_map
        ]

        result = selector.select(desk_roi_def, candidates)
        desk_results.append(result)

        if result.is_irate:
            logger.warning(
                f"🔴 IRATE_CUSTOMER  center={request.center_id}  "
                f"desk={desk_id}  table={result.table_id}  "
                f"winner_cam={result.winner_camera_id}  "
                f"clarity={result.winner_clarity_score:.3f}  "
                f"emotion={result.dominant_emotion.value}  "
                f"irate_conf={result.irate_confidence:.2f}"
            )

    # ── 5. Dispatch IRATE_CUSTOMER alerts as BackgroundTasks ──────────────
    irate_dispatched = 0
    for result in desk_results:
        if not result.is_irate:
            continue

        # Capture loop variables in default args to avoid late-binding closure bug
        async def _dispatch(r: BestViewResult = result, cam: str = result.winner_camera_id):
            await dispatch_face_alert(
                center_id=request.center_id,
                camera_id=cam,
                table_id=r.table_id,
                desk_id=r.desk_id,
                dominant_emotion=r.dominant_emotion.value,
                irate_confidence=r.irate_confidence,
            )

        background_tasks.add_task(_dispatch)
        irate_dispatched += 1

    inference_ms = round((time.time() - t0) * 1000, 2)

    logger.info(
        f"✅ Cross-camera done  center={request.center_id}  "
        f"desks={len(desk_results)}  irate={sum(1 for r in desk_results if r.is_irate)}  "
        f"alerts_queued={irate_dispatched}  total_ms={inference_ms:.1f}"
    )

    return CrossCameraAnalyzeResponse(
        center_id=request.center_id,
        layout_id=request.layout_id,
        desk_results=desk_results,
        total_irate_detected=sum(1 for r in desk_results if r.is_irate),
        irate_alerts_dispatched=irate_dispatched,
        inference_ms=inference_ms,
        server_time=server_time,
    )




# ─────────────────────────────────────────────────────────────────────────────
# POST /transcribe-audio — Whisper speech-to-text
# ─────────────────────────────────────────────────────────────────────────────

import os as _os

@app.post("/transcribe-audio", tags=["Audio"])
async def transcribe_audio(
    file: UploadFile     = File(..., description="Audio file (wav/mp3/m4a/ogg/flac/webm)"),
    table_id: str        = Form("", description="Table ID that captured this audio (optional)"),
    language: str        = Form("", description="ISO-639-1 hint e.g. 'en'. Leave blank to auto-detect."),
) -> JSONResponse:
    """
    Transcribe an audio buffer using OpenAI Whisper.

    - **file**: raw audio upload (wav / mp3 / m4a / ogg / flac / webm).
    - **table_id**: optional — the Table that owns the microphone.
    - **language**: optional ISO-639-1 code to skip language detection.

    Returns `transcribed_text`, `detected_language`, `confidence_score`,
    `duration_s`, `inference_ms`, and `model_size`.
    """
    # Guard: Whisper loaded?
    if get_whisper_model() is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Whisper model is not loaded. "
                "Install openai-whisper and restart the service."
            ),
        )

    # Guard: file present?
    if file is None or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No audio file provided.",
        )

    # Guard: content-type / extension
    content_type = (file.content_type or "").lower().split(";")[0].strip()
    ext = _os.path.splitext(file.filename or "")[-1].lower()
    valid_exts = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".mp4"}
    if content_type not in SUPPORTED_AUDIO_TYPES and ext not in valid_exts:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported audio format '{content_type or ext}'. "
                "Accepted: wav, mp3, m4a, ogg, flac, webm."
            ),
        )

    # Read body
    audio_bytes = await file.read()

    if len(audio_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded audio file is empty.",
        )

    if len(audio_bytes) > MAX_AUDIO_BYTES:
        mb_limit = MAX_AUDIO_BYTES // (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Audio file exceeds the {mb_limit} MB limit.",
        )

    # Transcribe
    try:
        result = await transcribe_audio_bytes(
            audio_bytes=audio_bytes,
            filename=file.filename or "upload.wav",
            language=language.strip() or None,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except Exception as exc:
        logger.exception(f"Transcription error for table={table_id!r}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Audio transcription failed. Check service logs.",
        )

    preview = result["transcribed_text"]
    logger.info(
        f"Transcription | table={table_id!r} | lang={result['detected_language']} | "
        f"conf={result['confidence_score']:.2f} | '{preview[:80]}{'...' if len(preview) > 80 else ''}'"
    )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "table_id":          table_id or None,
            "transcribed_text":  result["transcribed_text"],
            "detected_language": result["detected_language"],
            "confidence_score":  result["confidence_score"],
            "duration_s":        result["duration_s"],
            "inference_ms":      result["inference_ms"],
            "model_size":        result["model_size"],
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# POST /register-table — Pre-load agent reference embedding
# ─────────────────────────────────────────────────────────────────────────────

@app.post(
    "/register-table",
    response_model=TableRegistrationResponse,
    status_code=status.HTTP_200_OK,
    tags=["Agent Monitoring"],
    summary="Register a table and pre-load the agent's face embedding",
    description="""
Called by NestJS on startup (once per provisioned table) and whenever a Super
Admin reassigns an agent to a table.

Loads the ArcFace reference embedding from `face_photo_path` so that subsequent
`/monitor-agent` calls can perform fast cosine-distance identity checks without
re-loading the model.
""",
)
async def register_table(req: TableRegistrationRequest) -> TableRegistrationResponse:
    loaded = await agent_monitor.register_table(
        table_id=req.table_id,
        agent_id=req.agent_id,
        face_photo_path=req.face_photo_path,
        center_id=req.center_id,
        camera_id=req.camera_id,
    )
    return TableRegistrationResponse(
        table_id=req.table_id,
        agent_id=req.agent_id,
        embedding_loaded=loaded,
        message=(
            "Table registered — reference embedding loaded."
            if loaded
            else "Table registered — no face photo supplied (identity check disabled)."
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET /monitor-agent/status — snapshot of all registered tables
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/monitor-agent/status",
    tags=["Agent Monitoring"],
    summary="Live snapshot of all registered table monitoring states",
)
async def agent_monitor_status() -> dict:
    return {
        "registered_tables": len(agent_monitor.registered_tables),
        "tables": agent_monitor.all_snapshots(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# POST /monitor-agent — Per-frame agent monitoring (idle + gossip + SHI)
# ─────────────────────────────────────────────────────────────────────────────

@app.post(
    "/monitor-agent",
    response_model=AgentMonitorReport,
    status_code=status.HTTP_200_OK,
    tags=["Agent Monitoring"],
    summary="Run agent presence, gossip, and SHI checks on one camera frame",
    description="""
The primary per-frame agent monitoring endpoint.

**Three checks are performed on every call:**

1. **Agent Presence Verification**
   Detected faces inside the table ROI are compared (ArcFace cosine distance)
   against the assigned agent's reference embedding.  If the agent is absent
   for > `IDLE_THRESHOLD_MINUTES` (default 10 min) during work hours, an
   `IDLE_AGENT` alert is dispatched to NestJS.

2. **Gossip Detection**
   If ≥2 known-agent faces are present in the same ROI with no unrecognised
   customer face for > `GOSSIP_THRESHOLD_SECONDS` (default 300 s), a
   `GOSSIP_DETECTED` alert is dispatched.

3. **Service Happiness Index (SHI)**
   The agent's dominant emotion is sampled every 30 s.  Emotion scores are
   mapped to a weighted score (happy=100 … angry=0) and accumulated into a
   rolling SHI [0–100].  Metrics are PATCHed to NestJS /ingest/agent-activity
   for database persistence.
""",
)
async def monitor_agent(
    req: AgentMonitorFrameRequest,
    bg: BackgroundTasks,
) -> AgentMonitorReport:
    if face_analyzer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Face analyzer not initialised — is the service starting up?",
        )

    # ── 1. Decode frame ───────────────────────────────────────────────────────
    frame_pil = decode_frame(req.frame_base64, settings.max_image_size)

    # Convert PIL → numpy BGR for identity cropping (DeepFace uses BGR/numpy)
    import numpy as np
    import cv2
    frame_rgb = np.array(frame_pil)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    # ── 2. Run face emotion analysis on the table ROI ─────────────────────────
    face_detections = face_analyzer.analyze(frame_pil, req.bounding_box)

    # ── 3. Auto-register table if not already known ───────────────────────────
    if req.table_id not in agent_monitor.registered_tables:
        logger.warning(
            f"Table {req.table_id} not pre-registered — auto-registering without face photo. "
            "Call POST /register-table with face_photo_path to enable identity checks."
        )
        await agent_monitor.register_table(
            table_id=req.table_id,
            agent_id=req.agent_id,
            face_photo_path=None,
            center_id=req.center_id,
            camera_id=req.camera_id,
        )

    # ── 4. Run monitoring engine ──────────────────────────────────────────────
    ts = req.timestamp or time.time()
    result = await agent_monitor.process_frame(
        table_id=req.table_id,
        frame_bgr=frame_bgr,
        face_detections=face_detections,
        persons_in_roi=req.persons_in_roi,
        frame_ts=ts,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Table '{req.table_id}' not found in agent monitor.",
        )

    # ── 5. Build the API response ─────────────────────────────────────────────
    snapshot = agent_monitor.get_table_snapshot(req.table_id) or {}
    shi_history = [
        SHISampleOut(ts=s["ts"], shi=s["shi"], emotion=s["emotion"])
        for s in snapshot.get("shiSamples", [])
    ]

    return AgentMonitorReport(
        table_id=result.table_id,
        agent_id=result.agent_id,
        server_time=datetime.now(timezone.utc).isoformat(),
        agent_present=result.agent_present,
        idle_seconds=result.idle_seconds,
        idle_alert_triggered=result.idle_alert_triggered,
        gossip_active=result.gossip_active,
        gossip_duration_seconds=result.gossip_duration_seconds,
        gossip_alert_triggered=result.gossip_alert_triggered,
        shi_sampled=result.shi_sampled,
        latest_shi=result.latest_shi,
        rolling_shi=result.rolling_shi,
        latest_emotion=result.latest_emotion,
        shi_history=shi_history,
        active_minutes=result.active_minutes,
        agent_faces_in_roi=result.agent_faces_in_roi,
        unknown_faces_in_roi=result.unknown_faces_in_roi,
    )


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
